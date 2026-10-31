"""Bounded neural training, frozen evaluation, and lazy runtime assembly."""

from __future__ import annotations

from spire_learn.neural.profiling import observed

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import time
from types import SimpleNamespace
from typing import Any, Protocol
import uuid

from spire_learn.experiments.budget import Budget
from spire_learn.neural.artifacts import NeuralWriter
from spire_learn.neural.checkpoint import (
    identity_payload,
    load_checkpoint,
    preflight_checkpoint,
    save_checkpoint,
)
from spire_learn.neural.collector import Collector, CollectorRecovery
from spire_learn.neural.config import NeuralConfig
from spire_learn.neural.contracts import (
    CollectionDiagnostics,
    CombatFrame,
    NeuralIdentity,
    Rollout,
    StepRecord,
)
from spire_learn.neural.device import (
    RuntimeDevicePlan,
    plan_runtime_device,
    qualify_runtime_device,
    require_runtime_execution_gate,
    validate_resume_device,
)
from spire_learn.neural.evaluation import REFERENCE_POLICIES, choose_reference
from spire_learn.neural.ppo import UpdateFault, UpdateMetrics, ppo_update


_RNG_001 = (
    "Project-owned reproducible randomness is not seed-for-seed or bit-for-bit "
    "original-game RNG parity; no original-runtime transfer is established."
)
_TIME_001 = (
    "The full-run Secret Portal elapsed-time assumption is retained; this "
    "evaluation does not validate it."
)
_DIAGNOSTIC_INTEGER_FIELDS = (
    "attempted_environment_slots", "forwarded_environment_slots",
    "known_accepted_environment_slots", "delivered_environment_slots",
    "uncertain_environment_slots", "reset_operations", "reset_slots",
    "attempted_reset_slots", "forwarded_reset_slots", "known_accepted_reset_slots",
    "delivered_reset_slots", "uncertain_reset_slots", "learned_decisions",
    "fixed_decisions", "tasks_started", "task_terminals", "tasks_completed",
    "victories", "deaths", "faults", "incomplete_decisions",
    "incomplete_environment_slots", "undispatched_actor_decisions",
    "undispatched_fixed_decisions",
)
_PROGRESS_FIELDS = frozenset({
    "attempted_environment_slots", "forwarded_environment_slots",
    "known_accepted_environment_slots", "delivered_environment_slots",
    "uncertain_environment_slots", "attempted_reset_slots", "forwarded_reset_slots",
    "known_accepted_reset_slots", "delivered_reset_slots", "uncertain_reset_slots",
    "learned_decisions", "fixed_decisions", "optimizer_attempted_steps",
    "optimizer_accepted_steps", "optimizer_uncertain_steps", "update_count",
    "incomplete_decisions",
})


class TrainingRuntime(Protocol):
    """Narrow runner boundary; tests provide deterministic scripted runtimes."""

    collector: Any

    def update(self, rollout: Rollout) -> UpdateMetrics: ...

    def save(self, path: Path, purpose: str) -> None: ...

    def close(self) -> None: ...


@dataclass
class NeuralReport:
    summary: Mapping[str, object]
    exit_code: int
    run_dir: Path | None


@dataclass(frozen=True)
class RuntimeOperationLimits:
    """Per-invocation controls used by held, explicitly bounded operations."""

    max_updates: int | None = None
    monotonic_deadline: float | None = None
    max_environment_actions: int | None = None

    def __post_init__(self) -> None:
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 0):
            raise ValueError("max_updates must be a non-negative integer or None")
        if self.max_environment_actions is not None and (
            type(self.max_environment_actions) is not int or self.max_environment_actions < 0
        ):
            raise ValueError("max_environment_actions must be a non-negative integer or None")
        if self.monotonic_deadline is not None:
            if type(self.monotonic_deadline) not in (int, float):
                raise TypeError("monotonic_deadline must be a primitive number or None")
            if not math.isfinite(float(self.monotonic_deadline)):
                raise ValueError("monotonic_deadline must be finite")


class _RunnerBudget(Budget):
    """PPO-compatible wall budget with an optional shared absolute deadline."""

    def __init__(self, duration: float, clock: Callable[[], float], deadline: float | None):
        if type(duration) not in (int, float) or not math.isfinite(float(duration)):
            raise ValueError("wall allowance must be finite")
        if duration < 0:
            raise ValueError("wall allowance cannot be negative")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if deadline is not None and (
            type(deadline) not in (int, float) or not math.isfinite(float(deadline))
        ):
            raise ValueError("shared deadline must be finite")
        effective = float(duration)
        if effective == 0.0:
            # Budget intentionally requires positive constructor input. Its
            # public allowance is narrowed to exact zero immediately below.
            super().__init__(1.0, clock)
        else:
            super().__init__(effective, clock)
        if deadline is not None:
            effective = min(effective, max(0.0, float(deadline) - self._start))
        self._duration = effective
        self._max_wall_seconds = effective

    @property
    def allowed_seconds(self) -> float:
        return self._duration


def _operation_limits(runtime: Any) -> RuntimeOperationLimits:
    limits = getattr(runtime, "operation_limits", RuntimeOperationLimits())
    if not isinstance(limits, RuntimeOperationLimits):
        raise TypeError("runtime.operation_limits must be RuntimeOperationLimits")
    return limits


class _CountingOptimizer:
    """Expose optimizer prefix counts while preserving the accepted Torch API."""

    def __init__(self, optimizer: Any):
        self.inner = optimizer
        self.attempted = 0
        self.accepted = 0

    @property
    def param_groups(self):
        return self.inner.param_groups

    def zero_grad(self, *args, **kwargs):
        return self.inner.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.attempted += 1
        result = self.inner.step(*args, **kwargs)
        self.accepted += 1
        return result

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state):
        return self.inner.load_state_dict(state)


class _DeterministicPolicy:
    """Keep learned attribution while forcing the model's deterministic mode."""

    def __init__(self, policy: Any):
        self._policy = policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def act(self, frames, *, generator, deterministic=False):
        return self._policy.act(frames, generator=generator, deterministic=True)

    def values(self, frames):
        return self._policy.values(frames)


class _ReferencePolicy:
    def __init__(self, identity: NeuralIdentity, schema: Any, kind: str):
        if kind not in REFERENCE_POLICIES:
            raise ValueError(f"unsupported reference policy {kind!r}")
        self.identity = identity
        self.schema = schema
        self.kind = kind

    def act(self, frames, *, generator, deterministic=False):
        from spire_learn.neural.contracts import PolicyChoice

        return tuple(
            PolicyChoice(choose_reference(frame, self.schema, kind=self.kind, rng=generator))
            for frame in frames
        )


def _error_value(error: BaseException) -> dict[str, object]:
    try:
        message = str(error).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    except BaseException:
        message = "exception message unavailable"
    return {
        "exception_type": type(error).__name__,
        "message": message[:512],
    }


def _is_process_control(error: BaseException) -> bool:
    return not isinstance(error, Exception)


def _promote_finalization_control(
    primary: BaseException | None, control: BaseException, label: str,
) -> BaseException:
    """Promote the first finalization control while retaining exception context."""
    if primary is None:
        return control
    if _is_process_control(primary):
        _note(primary, f"later {label} process control also raised: {type(control).__name__}: {_error_value(control)['message']}")
        return primary
    _note(control, f"{label} process control superseded prior {type(primary).__name__}: {_error_value(primary)['message']}")
    return control


def _note(primary: BaseException, detail: str) -> None:
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(detail)


def _close_resource(
    primary: BaseException | None, resource: Any, label: str,
) -> tuple[BaseException | None, BaseException | None]:
    close = getattr(resource, "close", None)
    if not callable(close):
        return primary, None
    try:
        close()
    except BaseException as error:
        if primary is not None:
            if _is_process_control(error) and not _is_process_control(primary):
                _note(error, f"{label} close followed an operational failure: {type(primary).__name__}: {_error_value(primary)['message']}")
                return error, error
            _note(primary, f"{label} close also failed: {type(error).__name__}: {_error_value(error)['message']}")
            return primary, error
        return error, error
    return primary, None


def _close_preserving(primary: BaseException | None, resource: Any, label: str) -> BaseException | None:
    return _close_resource(primary, resource, label)[0]


def _preflight_close(runtime: Any, report: NeuralReport, label: str = "runtime") -> NeuralReport:
    primary, close_error = _close_resource(None, runtime, label)
    if primary is not None and _is_process_control(primary):
        raise primary
    if close_error is None:
        return report
    summary = dict(report.summary)
    summary.update({
        "status": "operational_failure",
        "stop_reason": "preflight_close_failure",
        "fatal_error": _error_value(close_error),
        "exit_code": 1,
    })
    return NeuralReport(summary, 1, report.run_dir)


@observed("frame_hash")
def _frame_hash(frame: CombatFrame) -> str:
    digest = hashlib.sha256()
    digest.update(frame.schema_fingerprint.encode("ascii"))
    for name, buffer in sorted(frame.tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(buffer.dtype.encode("ascii"))
        digest.update(struct.pack("!I", len(buffer.shape)))
        for dim in buffer.shape:
            digest.update(struct.pack("!Q", dim))
        digest.update(buffer.data)
    return digest.hexdigest()


def _config_projection(config: NeuralConfig) -> dict[str, object]:
    return {
        "schema": "spire-learn-neural-config-v1",
        "config_sha256": config.config_sha256,
        "mode": config.mode,
        "operation": config.operation,
        "split": config.split,
        "partition": config.partition,
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": config.manifest_sha256,
        "output_root": str(config.output_root),
        "environment_revision": config.environment_revision,
        "common_view_id": config.common_view_id,
        "common_view_fingerprint": config.common_view_fingerprint,
        "run_observation_fingerprint": config.run_observation_fingerprint,
        "profile_id": config.profile_id,
        "observation_version": config.observation_version,
        "replay_version": config.replay_version,
        "device": config.device,
        "runtime_gpu_uuid": config.runtime_gpu_uuid,
        "seed_streams": {
            "policy": config.seed_streams.policy,
            "optimizer": config.seed_streams.optimizer,
            "curriculum": config.seed_streams.curriculum,
            "noncombat": config.seed_streams.noncombat,
        },
        "model": {
            "embedding_width": config.model.embedding_width,
            "hidden_width": config.model.hidden_width,
            "max_parameters": config.model.max_parameters,
        },
        "ppo": {
            "learning_rate": config.ppo.learning_rate,
            "gamma": config.ppo.gamma,
            "gae_lambda": config.ppo.gae_lambda,
            "clip_epsilon": config.ppo.clip_epsilon,
            "value_coefficient": config.ppo.value_coefficient,
            "entropy_coefficient": config.ppo.entropy_coefficient,
            "max_grad_norm": config.ppo.max_grad_norm,
            "epochs": config.ppo.epochs,
            "minibatch_rows": config.ppo.minibatch_rows,
            "adam_eps": config.ppo.adam_eps,
        },
        "rollout": {"rows": config.rollout_rows, "steps": config.rollout_steps},
        "budgets": {
            "max_optimizer_steps": config.budgets.max_optimizer_steps,
            "max_training_decisions": config.budgets.max_training_decisions,
            "max_environment_actions": config.budgets.max_environment_actions,
            "max_wall_seconds": config.budgets.max_wall_seconds,
        },
    }


def _manifest_projection(
    config: NeuralConfig,
    runtime: Any,
    *,
    policy_kind: str,
    action_selection: str,
    allow_final: bool,
) -> dict[str, object]:
    schema = getattr(getattr(runtime, "collector", None), "driver", None)
    schema = getattr(schema, "schema", None)
    cases = [
        {
            "case_id": case.case_id,
            "lineage_id": case.lineage_id,
            "mode": case.mode,
            "environment_seed": case.environment_seed,
            "cohort": case.cohort,
            "entry_id": case.entry_id,
            "profile_id": case.profile_id,
            "character": case.character,
            "ascension": case.ascension,
        }
        for case in config.cases.for_mode(config.mode)
    ]
    return {
        "operation": config.operation,
        "mode": config.mode,
        "split": config.split,
        "partition": config.partition,
        "case_ids": [case["case_id"] for case in cases],
        "cases": cases,
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": config.manifest_sha256,
        "environment_revision": config.environment_revision,
        "learner_dirty_state": "recorded_by_neural_writer_provenance",
        "environment_dirty_state": "unobserved_source_checkout_not_mounted",
        "environment_dirty_state_qualification": (
            "the loaded native build is checked against environment_revision; "
            "no Gym Git checkout is mounted for a dirty-state query"
        ),
        "profile_id": config.profile_id or "ironclad-combat-v1",
        "content_id": "spire-gym-ironclad-combat-content-v1" if config.mode == "standalone" else "profile-defined-m10-ironclad-policy",
        "rules_id": "spire-gym-native-m10-rules-v1",
        "reward_id": "spire-gym-combat-outcome-reward-v1" if config.mode == "standalone" else "spire-gym-floor-key-terminal-reward-v1",
        "common_view_id": config.common_view_id,
        "common_view_fingerprint": config.common_view_fingerprint,
        "run_observation_fingerprint": config.run_observation_fingerprint,
        "schema_descriptor_digest": getattr(schema, "descriptor_digest", None),
        "learner_identity": identity_payload(_selected_identity(config)),
        "policy_kind": policy_kind,
        "controller_id": getattr(runtime, "controller_id", policy_kind),
        "action_selection": action_selection,
        "runtime_device_report": (
            dict(runtime.device_report)
            if isinstance(getattr(runtime, "device_report", None), Mapping) else None
        ),
        "run_noncombat_controller_id": "legal-random-v1" if config.mode == "run" else None,
        "parent_checkpoint": str(getattr(runtime, "parent_checkpoint", None)) if getattr(runtime, "parent_checkpoint", None) is not None else None,
        "allow_final": allow_final,
        "rng_001": _RNG_001,
        "time_001": _TIME_001,
        "resource_budgets": _config_projection(config)["budgets"],
        "python_version": platform.python_version(),
    }


def _selected_identity(config: NeuralConfig) -> NeuralIdentity:
    identities = tuple(
        identity for identity in config.accepted_identities
        if config.mode in identity.accepted_raw_mode_mappings
    )
    if len(identities) != 1:
        raise ValueError("neural config must select exactly one identity for its mode")
    return identities[0]


def _diagnostic_projection(diagnostics: CollectionDiagnostics | None) -> dict[str, object]:
    if diagnostics is None:
        return {name: 0 for name in _DIAGNOSTIC_INTEGER_FIELDS} | {
            "elapsed_seconds": 0.0,
            "wall_overshoot_seconds": 0.0,
            "next_seed_ordinal": 0,
            "stop_reason": "no_collection",
            "fault": None,
        }
    result = {name: getattr(diagnostics, name) for name in _DIAGNOSTIC_INTEGER_FIELDS}
    result.update({
        "elapsed_seconds": diagnostics.elapsed_seconds,
        "wall_overshoot_seconds": diagnostics.wall_overshoot_seconds,
        "next_seed_ordinal": diagnostics.next_seed_ordinal,
        "stop_reason": diagnostics.stop_reason,
        "elapsed_seconds_is_fallback": diagnostics.elapsed_seconds_is_fallback,
        "fault": None if diagnostics.fault is None else {
            "phase": diagnostics.fault.phase,
            "exception_type": diagnostics.fault.exception_type,
            "message": diagnostics.fault.message,
            "operation_slots": diagnostics.fault.operation_slots,
            "known_accepted_slots": diagnostics.fault.known_accepted_slots,
            "uncertain_slots": diagnostics.fault.uncertain_slots,
        },
        "incomplete_tasks": [
            {"stream_id": item.stream_id, "task_id": item.task_id, "decisions": item.decisions}
            for item in diagnostics.incomplete_tasks
        ],
    })
    return result


def _write_collection_events(
    writer: NeuralWriter,
    rollout: Rollout,
    *,
    cursors: dict[str, int],
) -> None:
    for record in rollout.records:
        writer.event({
            "event": "accepted_action",
            "stream_id": record.stream_id,
            "task_id": record.task_id,
            "decision_id": record.decision_id,
            "action_index": record.choice.original_action_index,
            "actor_training_mask": record.actor_training_mask,
            "native_reward": record.native_reward,
            "task_reward": record.task_signal.reward,
            "task_terminal": record.task_signal.task_terminal,
            "bootstrap": record.task_signal.bootstrap,
            "continuation": record.task_signal.continuation,
            "terminated": record.terminated,
            "truncated": record.truncated,
            "source_public_sha256": _frame_hash(record.source),
            "successor_public_sha256": _frame_hash(record.successor),
        })
    diagnostics = rollout.collection
    if diagnostics is None:
        return
    for allocation in diagnostics.seed_allocations[cursors["allocations"]:]:
        writer.event({
            "event": "seed_allocation",
            "stream_id": allocation.stream_id,
            "slot": allocation.slot,
            "case_id": allocation.case_id,
            "entry_id": allocation.entry_id,
            "authored_reference_seed": allocation.authored_reference_seed,
            "actual_seed": allocation.actual_seed,
            "ordinal": allocation.ordinal,
            "allocation_policy_id": allocation.allocation_policy_id,
            "delivery": allocation.delivery,
        })
    cursors["allocations"] = len(diagnostics.seed_allocations)
    for checkpoint in diagnostics.resource_checkpoints[cursors["checkpoints"]:]:
        writer.event({
            "event": "combat_settlement",
            "stream_id": checkpoint.stream_id,
            "task_id": checkpoint.task_id,
            "visit": checkpoint.visit,
            "outcome": checkpoint.outcome,
            "encounter": checkpoint.encounter,
            "hp_before_healing": checkpoint.hp_before_healing,
            "hp_after_cleanup": checkpoint.hp_after_cleanup,
            "public_frame_sha256": _frame_hash(checkpoint.frame),
        })
    cursors["checkpoints"] = len(diagnostics.resource_checkpoints)
    writer.event({"event": "collection_progress", "collection": _diagnostic_projection(diagnostics)})


def _current_collection(runtime: Any) -> CollectionDiagnostics | None:
    partial = getattr(runtime.collector, "partial_rollout", None)
    return partial.collection if isinstance(partial, Rollout) else None


def _current_policy_version(runtime: Any) -> str | int:
    policy = getattr(runtime.collector, "policy", None)
    version = getattr(policy, "_behavior_policy_version", None)
    if type(version) is int and version >= 0:
        return version
    if isinstance(version, str) and version:
        return version
    if getattr(runtime.collector, "learned_combat_policy", True) is False:
        return "reference-policy-v1"
    raise ValueError("collector policy does not expose a valid behavior policy version")


def _set_budget(runtime: Any, budget: Any) -> None:
    if hasattr(runtime, "set_budget") and callable(runtime.set_budget):
        runtime.set_budget(budget)
    else:
        runtime.budget = budget
        if hasattr(runtime.collector, "budget"):
            runtime.collector.budget = budget


def _counter_total(runtime: Any, field: str) -> int:
    if field in {
        "optimizer_attempted_steps", "optimizer_accepted_steps",
        "optimizer_uncertain_steps", "update_count",
    } and hasattr(runtime, field):
        value = getattr(runtime, field)
        if type(value) is not int or value < 0:
            raise ValueError(f"runtime progress {field} is invalid")
        return value
    prefix = getattr(runtime, "recovered_progress", {})
    if not isinstance(prefix, Mapping):
        return 0
    value = prefix.get(field, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"runtime recovered progress {field} is invalid")
    diagnostics = _current_collection(runtime)
    current = getattr(diagnostics, field, 0) if diagnostics is not None else 0
    if type(current) is not int or current < 0:
        raise ValueError(f"collector progress {field} is invalid")
    return value + current


def _max_optimizer_steps_per_rollout(config: NeuralConfig, max_actor_rows: int) -> int:
    if max_actor_rows <= 0:
        return 0
    return config.ppo.epochs * math.ceil(max_actor_rows / config.ppo.minibatch_rows)


def _actor_rows(rollout: Rollout) -> int:
    return sum(record.actor_training_mask for record in rollout.records)


def _recovery_state(runtime: Any) -> CollectorRecovery:
    state = getattr(runtime.collector, "recovery_state", None)
    if not isinstance(state, CollectorRecovery):
        raise ValueError("collector does not expose a valid fresh-environment recovery state")
    return state


def _collection_budget_stop(config: NeuralConfig, runtime: Any) -> str | None:
    prefix = getattr(runtime, "recovered_progress", {})
    if not isinstance(prefix, Mapping):
        raise ValueError("runtime recovered progress must be a mapping")
    env_prefix = prefix.get("attempted_environment_slots", 0)
    learned_prefix = prefix.get("learned_decisions", 0)
    for name, value in (("attempted_environment_slots", env_prefix), ("learned_decisions", learned_prefix)):
        if type(value) is not int or value < 0:
            raise ValueError(f"runtime recovered progress {name} is invalid")
    limits = _operation_limits(runtime)
    if limits.max_updates is not None and _counter_total(runtime, "update_count") >= limits.max_updates:
        return "update_count_limit"
    local = _current_collection(runtime)
    local_env = local.attempted_environment_slots if local is not None else 0
    local_learned = local.learned_decisions if local is not None else 0
    remaining_environment = _collection_environment_ceiling(config, runtime) - local_env
    remaining_learned = config.budgets.max_training_decisions - learned_prefix - local_learned
    if remaining_environment < config.rollout_rows:
        return "environment_action_budget"
    if remaining_learned < config.rollout_rows:
        return "learned_decision_budget"
    attempted_steps = _counter_total(runtime, "optimizer_attempted_steps")
    remaining_optimizer = config.budgets.max_optimizer_steps - attempted_steps
    if remaining_optimizer <= 0:
        return "optimizer_step_budget"
    max_actor_rows = min(
        config.rollout_rows * config.rollout_steps,
        remaining_learned,
        (remaining_optimizer // config.ppo.epochs) * config.ppo.minibatch_rows,
    )
    if max_actor_rows < 1:
        return "optimizer_step_budget"
    return None


def _collection_environment_ceiling(config: NeuralConfig, runtime: Any) -> int:
    """Collector-local attempt ceiling after subtracting the restored prefix."""
    prefix = getattr(runtime, "recovered_progress", {})
    if not isinstance(prefix, Mapping):
        raise ValueError("runtime recovered progress must be a mapping")
    restored = prefix.get("attempted_environment_slots", 0)
    if type(restored) is not int or restored < 0:
        raise ValueError("runtime recovered environment-action prefix is invalid")
    ceiling = config.budgets.max_environment_actions
    operational = _operation_limits(runtime).max_environment_actions
    if operational is not None:
        ceiling = min(ceiling, operational)
    return max(0, ceiling - restored)


def _collection_learned_ceiling(config: NeuralConfig, runtime: Any) -> int:
    """Collector-local cumulative ceiling that fits remaining decision/step caps."""
    prefix = getattr(runtime, "recovered_progress", {})
    if not isinstance(prefix, Mapping):
        raise ValueError("runtime recovered progress must be a mapping")
    prefix_learned = prefix.get("learned_decisions", 0)
    if type(prefix_learned) is not int or prefix_learned < 0:
        raise ValueError("runtime recovered learned-decision prefix is invalid")
    diagnostics = _current_collection(runtime)
    local_learned = diagnostics.learned_decisions if diagnostics is not None else 0
    remaining_learned = config.budgets.max_training_decisions - prefix_learned - local_learned
    remaining_steps = config.budgets.max_optimizer_steps - _counter_total(runtime, "optimizer_attempted_steps")
    allowed_rows = min(
        config.rollout_rows * config.rollout_steps,
        remaining_learned,
        (remaining_steps // config.ppo.epochs) * config.ppo.minibatch_rows,
    )
    return local_learned + max(0, allowed_rows)


def _record_update_counts(metrics: UpdateMetrics, config: NeuralConfig, runtime: Any) -> None:
    del config
    record = getattr(runtime, "record_update", None)
    if callable(record):
        record(metrics)
    elif hasattr(runtime, "optimizer_attempted_steps"):
        runtime.optimizer_attempted_steps += metrics.attempted_steps
        runtime.optimizer_accepted_steps += metrics.accepted_steps


def _metric_event(metrics: UpdateMetrics) -> dict[str, object]:
    return {
        "event": "optimizer_update",
        "policy_loss": metrics.policy_loss,
        "value_loss": metrics.value_loss,
        "entropy": metrics.entropy,
        "attempted_steps": metrics.attempted_steps,
        "accepted_steps": metrics.accepted_steps,
        "uncertain_steps": 0,
        "stop_reason": metrics.stop_reason,
        "admitted_behavior_policy_version": metrics.admitted_behavior_policy_version,
        "resulting_behavior_policy_version": metrics.resulting_behavior_policy_version,
        "consumed_decisions": metrics.consumed_decisions,
    }


def _fault_update_event(error: UpdateFault) -> dict[str, object]:
    return {
        "event": "optimizer_update_fault",
        "attempted_steps": error.attempted_steps,
        "accepted_steps": error.accepted_steps,
        "uncertain_steps": error.uncertain_steps,
        "phase": error.phase,
        "admitted_behavior_policy_version": error.admitted_behavior_policy_version,
        "resulting_behavior_policy_version": error.resulting_behavior_policy_version,
        "consumed_decisions": error.consumed_decisions,
        "exception": _error_value(error.__cause__ or error),
    }


def _case_accounting(
    config: NeuralConfig,
    records: Sequence[StepRecord],
    diagnostics: CollectionDiagnostics | None,
    recovery: CollectorRecovery,
) -> dict[str, object]:
    cases = config.cases.for_mode(config.mode)
    allocation_by_task: dict[str, str] = {}
    started: set[str] = set()
    next_episode = list(recovery.next_episode_ordinals)
    if diagnostics is not None:
        for allocation in diagnostics.seed_allocations:
            if allocation.delivery != "delivered":
                continue
            slot = allocation.slot
            if not 0 <= slot < len(next_episode):
                continue
            episode = next_episode[slot]
            if config.mode == "standalone":
                allocation_by_task[f"task-{slot}-{episode}"] = allocation.case_id
            else:
                allocation_by_task[f"task-{slot}-run-{episode}"] = allocation.case_id
            # A delivered reset starts a fixed case even if a wall or action
            # ceiling prevents its first accepted action.
            started.add(allocation.case_id)
            next_episode[slot] += 1

    completed: set[str] = set()
    for record in records:
        if config.mode == "standalone":
            case_id = allocation_by_task.get(record.task_id)
            if case_id is not None:
                started.add(case_id)
                if record.task_signal.task_terminal:
                    completed.add(case_id)
        else:
            episode_key = "-".join(record.task_id.split("-")[:4])
            case_id = allocation_by_task.get(episode_key)
            if case_id is not None:
                started.add(case_id)
                if record.terminated:
                    completed.add(case_id)
    if diagnostics is not None and config.mode == "run":
        for checkpoint in diagnostics.resource_checkpoints:
            episode_key = "-".join(checkpoint.task_id.split("-")[:4])
            case_id = allocation_by_task.get(episode_key)
            if case_id is not None:
                started.add(case_id)

    rows = []
    for case in cases:
        if case.case_id in completed:
            status = "complete"
        elif case.case_id in started:
            status = "incomplete"
        else:
            status = "not_started"
        rows.append({
            "case_id": case.case_id,
            "lineage_id": case.lineage_id,
            "environment_seed": case.environment_seed,
            "status": status,
        })
    return {
        "scheduled": len(rows),
        "complete": sum(item["status"] == "complete" for item in rows),
        "incomplete": sum(item["status"] == "incomplete" for item in rows),
        "not_started": sum(item["status"] == "not_started" for item in rows),
        "cases": rows,
    }


def _public_scalar(frame: CombatFrame, schema: Any, group: str, names: tuple[str, ...]) -> int | None:
    spec = schema.tensors.get(group)
    values = frame.tensors.get(group)
    if spec is None or values is None or len(spec.shape) != 1 or values.dtype != "int64":
        return None
    for name in names:
        index = next((i for i, column in enumerate(spec.columns) if column.field == name), None)
        if index is None:
            continue
        if spec.known_key is not None:
            known = frame.tensors.get(spec.known_key)
            if known is None or known.data[index] != 1:
                continue
        return struct.unpack_from("=q", values.data, index * 8)[0]
    return None


def _fight_summary(config: NeuralConfig, runtime: Any, records: Sequence[StepRecord], diagnostics: CollectionDiagnostics | None) -> dict[str, object]:
    schema = getattr(getattr(runtime.collector, "driver", None), "schema", None)
    resources = diagnostics.resource_checkpoints if diagnostics is not None else ()
    records_by_task: dict[str, list[StepRecord]] = defaultdict(list)
    for record in records:
        records_by_task[record.task_id].append(record)
    hp_fractions: list[float] = []
    fight_rows = []
    for checkpoint in resources:
        maximum = _public_scalar(checkpoint.frame, schema, "player", ("max_hp", "player_max_hp")) if schema is not None else None
        retention = checkpoint.hp_after_cleanup / maximum if maximum is not None and maximum > 0 else None
        if retention is not None and math.isfinite(retention):
            hp_fractions.append(retention)
        task_records = records_by_task.get(checkpoint.task_id, [])
        turns: int | None = None
        if schema is not None:
            observed = {
                _public_scalar(record.source, schema, "combat_state", ("turn", "turn_number", "turn_index"))
                for record in task_records
            }
            observed.discard(None)
            turns = len(observed) if observed else None
        fight_rows.append({
            "task_id": checkpoint.task_id,
            "visit": checkpoint.visit,
            "encounter": checkpoint.encounter,
            "outcome": checkpoint.outcome,
            "actions": len(task_records),
            "turns": turns,
            "hp_before_healing": checkpoint.hp_before_healing,
            "hp_after_cleanup": checkpoint.hp_after_cleanup,
            "max_hp": maximum,
            "hp_retention": retention,
            "public_frame_sha256": _frame_hash(checkpoint.frame),
        })
    completed = len(fight_rows)
    wins = sum(item["outcome"] == "victory" for item in fight_rows)
    deaths = sum(item["outcome"] == "death" for item in fight_rows)
    tasks_started = diagnostics.tasks_started if diagnostics is not None else 0
    run_terminated = any(record.terminated for record in records) if config.mode == "run" else False
    return {
        "standalone_fights": {
            "reached_combat_denominator": tasks_started if config.mode == "standalone" else 0,
            "completed": completed if config.mode == "standalone" else 0,
            "victories": wins if config.mode == "standalone" else 0,
            "deaths": deaths if config.mode == "standalone" else 0,
            "mean_hp_retention": (sum(hp_fractions) / len(hp_fractions) if hp_fractions and config.mode == "standalone" else None),
            "fights": fight_rows if config.mode == "standalone" else [],
        },
        "in_run_fights": {
            "reached_combat_denominator": tasks_started if config.mode == "run" else 0,
            "completed": completed if config.mode == "run" else 0,
            "victories": wins if config.mode == "run" else 0,
            "deaths": deaths if config.mode == "run" else 0,
            "mean_hp_retention": (sum(hp_fractions) / len(hp_fractions) if hp_fractions and config.mode == "run" else None),
            "fights": fight_rows if config.mode == "run" else [],
        },
        "full_run_progress": {
            "run_cases_scheduled": len(config.cases.for_mode("run")) if config.mode == "run" else 0,
            "combat_visits_settled": completed if config.mode == "run" else 0,
            "run_terminal_observed": run_terminated,
            "status": "complete" if run_terminated else "incomplete" if config.mode == "run" else "not_applicable",
            "progress_source": "public combat visit checkpoints and terminal flags",
        },
        "environment_actions": len(records),
    }


def _base_summary(
    config: NeuralConfig,
    runtime: Any,
    *,
    status: str,
    stop_reason: str,
    run_dir: Path | None,
    elapsed: float,
    overshoot: float,
    records: Sequence[StepRecord],
    diagnostics: CollectionDiagnostics | None,
    recovery: CollectorRecovery,
    fatal_error: BaseException | None,
    policy_kind: str,
    action_selection: str,
    allow_final: bool,
) -> dict[str, object]:
    prefix = getattr(runtime, "recovered_progress", {})
    prefix = prefix if isinstance(prefix, Mapping) else {}
    counters = _diagnostic_projection(diagnostics)
    total_optimizer_attempted = _counter_total(runtime, "optimizer_attempted_steps") if hasattr(runtime, "collector") else 0
    total_optimizer_accepted = _counter_total(runtime, "optimizer_accepted_steps") if hasattr(runtime, "collector") else 0
    total_optimizer_uncertain = _counter_total(runtime, "optimizer_uncertain_steps") if hasattr(runtime, "collector") else 0
    accounting = _case_accounting(config, records, diagnostics, recovery)
    return {
        "schema_id": "spire-learn-neural-report-v1",
        "operation": config.operation,
        "mode": config.mode,
        "split": config.split,
        "partition": config.partition,
        "status": status,
        "stop_reason": stop_reason,
        "run_directory": str(run_dir) if run_dir is not None else None,
        "policy_kind": policy_kind,
        "controller_id": getattr(runtime, "controller_id", policy_kind),
        "action_selection": action_selection,
        "allow_final": allow_final,
        "config_sha256": config.config_sha256,
        "manifest_sha256": config.manifest_sha256,
        "environment_revision": config.environment_revision,
        "runtime_device_report": (
            dict(runtime.device_report)
            if isinstance(getattr(runtime, "device_report", None), Mapping) else None
        ),
        "common_view_fingerprint": config.common_view_fingerprint,
        "run_observation_fingerprint": config.run_observation_fingerprint,
        "case_accounting": accounting,
        "collection": counters,
        "cumulative_prefix": {
            "environment_slots": prefix.get("attempted_environment_slots", 0),
            "learned_decisions": prefix.get("learned_decisions", 0),
            "optimizer_attempted_steps": prefix.get("optimizer_attempted_steps", 0),
            "optimizer_accepted_steps": prefix.get("optimizer_accepted_steps", 0),
            "elapsed_seconds": prefix.get("elapsed_seconds", 0.0),
        },
        "optimizer": {
            "attempted_steps": total_optimizer_attempted,
            "accepted_steps": total_optimizer_accepted,
            "uncertain_steps": total_optimizer_uncertain,
            "updates": getattr(runtime, "update_count", 0),
            "policy_version": _runtime_policy_version(runtime),
        },
        "fights": _fight_summary(config, runtime, records, diagnostics),
        "elapsed_seconds": elapsed,
        "wall_overshoot_seconds": overshoot,
        "total_elapsed_seconds_with_resume_prefix": float(prefix.get("elapsed_seconds", 0.0)) + elapsed,
        "fault": _error_value(fatal_error) if fatal_error is not None else None,
        "rng_001": _RNG_001,
        "time_001": _TIME_001,
    }


def _runtime_policy_version(runtime: Any) -> str | int | None:
    try:
        return _current_policy_version(runtime)
    except (ValueError, AttributeError):
        return None


def _preflight_report(config: NeuralConfig, *, reason: str, policy_kind: str, action_selection: str, allow_final: bool) -> NeuralReport:
    return NeuralReport({
        "schema_id": "spire-learn-neural-report-v1",
        "operation": config.operation,
        "mode": config.mode,
        "split": config.split,
        "status": "setup_blocked",
        "stop_reason": reason,
        "case_accounting": {
            "scheduled": len(config.cases.for_mode(config.mode)),
            "complete": 0,
            "incomplete": 0,
            "not_started": len(config.cases.for_mode(config.mode)),
            "cases": [
                {"case_id": case.case_id, "lineage_id": case.lineage_id,
                 "environment_seed": case.environment_seed, "status": "not_started"}
                for case in config.cases.for_mode(config.mode)
            ],
        },
        "policy_kind": policy_kind,
        "action_selection": action_selection,
        "allow_final": allow_final,
        "fatal_error": {"phase": "preflight", "message": reason},
    }, 2, None)


def _run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex


def _writer_start(
    config: NeuralConfig,
    runtime: Any,
    *,
    operation: str,
    policy_kind: str,
    action_selection: str,
    allow_final: bool,
    run_id: str | None = None,
) -> tuple[NeuralWriter, Path]:
    writer = NeuralWriter.create(config.output_root, run_id or _run_id())
    manifest = _manifest_projection(
        config, runtime, policy_kind=policy_kind, action_selection=action_selection,
        allow_final=allow_final,
    )
    manifest["operation"] = operation
    try:
        writer.start(_config_projection(config), manifest)
    except BaseException as error:
        primary = _close_preserving(error, writer, "artifact writer")
        if primary is not error:
            raise primary
        raise
    return writer, writer.run_dir


def _write_lifecycle_receipt(writer: Any, run_dir: Path | None, receipt: Mapping[str, object]) -> None:
    writer_method = getattr(writer, "write_lifecycle_receipt", None)
    if callable(writer_method):
        writer_method(receipt)
        return
    if run_dir is None:
        raise OSError("cannot write lifecycle receipt without an owned run directory")
    path = Path(run_dir) / "lifecycle.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(receipt), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(Path(run_dir), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mark_receipt_pending(summary: dict[str, object], budget: _RunnerBudget) -> None:
    summary["exit_code"] = 1
    summary["lifecycle_receipt_required"] = True
    summary["lifecycle"] = {
        "status": "receipt_pending",
        "exit_code": 1,
        "elapsed_seconds_before_summary_and_writer_close": budget.elapsed(),
        "timing_boundary": "after operation and runtime close; before summary and writer close",
        "receipt_path": "lifecycle.json",
        "authority": "lifecycle receipt is authoritative for final status, elapsed time, and exit code",
    }


def _update_once(
    config: NeuralConfig,
    runtime: TrainingRuntime,
    writer: NeuralWriter,
    rollout: Rollout,
) -> tuple[bool, BaseException | None]:
    count = _actor_rows(rollout)
    if count == 0:
        return False, None
    limits = _operation_limits(runtime)
    if limits.max_updates is not None and _counter_total(runtime, "update_count") >= limits.max_updates:
        return True, None
    max_steps = _max_optimizer_steps_per_rollout(config, count)
    attempted = _counter_total(runtime, "optimizer_attempted_steps")
    if config.budgets.max_optimizer_steps - attempted < max_steps:
        return False, None
    expected_version = _current_policy_version(runtime)
    try:
        metrics = runtime.update(rollout)
    except UpdateFault as error:
        record = getattr(runtime, "record_update_fault", None)
        if callable(record):
            record(error)
        writer.event(_fault_update_event(error))
        return True, error
    except BaseException as error:
        prefix = getattr(runtime, "last_update_prefix", None)
        if _is_process_control(error):
            if isinstance(prefix, Mapping):
                try:
                    writer.event({"event": "optimizer_update_interrupted", **dict(prefix)})
                except BaseException as artifact_error:
                    _note(error, f"optimizer interruption event also failed: {type(artifact_error).__name__}: {_error_value(artifact_error)['message']}")
            raise
        if isinstance(prefix, Mapping):
            writer.event({"event": "optimizer_update_interrupted", **dict(prefix)})
        writer.event({"event": "optimizer_update_failure", "exception": _error_value(error)})
        return True, error
    if not isinstance(metrics, UpdateMetrics):
        error = TypeError("TrainingRuntime.update must return UpdateMetrics")
        writer.event({"event": "optimizer_update_failure", "exception": _error_value(error)})
        return True, error
    if metrics.admitted_behavior_policy_version != expected_version:
        error = RuntimeError("update admitted a stale behavior policy version")
        writer.event({"event": "optimizer_update_failure", "exception": _error_value(error)})
        return True, error
    _record_update_counts(metrics, config, runtime)
    if _current_policy_version(runtime) != metrics.resulting_behavior_policy_version:
        error = RuntimeError("runtime policy version disagrees with UpdateMetrics")
        writer.event({"event": "optimizer_update_failure", "exception": _error_value(error)})
        return True, error
    writer.event(_metric_event(metrics))
    if metrics.stop_reason is not None:
        return True, None
    return False, None


def train(
    config: NeuralConfig,
    *,
    runtime: TrainingRuntime,
    clock: Callable[[], float] = time.monotonic,
) -> NeuralReport:
    """Train under configured lifetime caps and persist a complete report."""
    if not isinstance(config, NeuralConfig):
        primary, close_error = _close_resource(None, runtime, "runtime")
        if primary is not None and _is_process_control(primary):
            raise primary
        reason = "config must be NeuralConfig"
        if close_error is not None:
            reason += f"; runtime close failed: {_error_value(close_error)['message']}"
        return NeuralReport({"status": "operational_failure" if close_error else "setup_blocked",
                             "reason": reason}, 1 if close_error else 2, None)
    if config.scenario_stream is not None:
        from .stream_resume import stream_identity
        try:
            stream_identity(config)
        except Exception as error:
            return _preflight_close(runtime, _preflight_report(
                config, reason=f"stream admission failed: {_error_value(error)['message']}",
                policy_kind="checkpoint", action_selection="seeded-sampling", allow_final=False,
            ))
    if config.operation != "train" or config.split != "train" or config.partition != "train":
        return _preflight_close(runtime, _preflight_report(
            config, reason="training requires config.operation=train and split=partition=train",
            policy_kind="checkpoint", action_selection="seeded-sampling", allow_final=False,
        ))
    if not callable(clock):
        return _preflight_close(runtime, _preflight_report(
            config, reason="clock must be callable", policy_kind="checkpoint",
            action_selection="seeded-sampling", allow_final=False,
        ))

    prefix = getattr(runtime, "recovered_progress", {})
    prefix_elapsed = prefix.get("elapsed_seconds", 0.0) if isinstance(prefix, Mapping) else 0.0
    if type(prefix_elapsed) not in (int, float) or not math.isfinite(prefix_elapsed) or prefix_elapsed < 0:
        return _preflight_close(runtime, _preflight_report(
            config, reason="recovered elapsed budget is invalid", policy_kind="checkpoint",
            action_selection="seeded-sampling", allow_final=False,
        ))
    remaining_wall = max(0.0, config.budgets.max_wall_seconds - float(prefix_elapsed))
    try:
        limits = _operation_limits(runtime)
        budget = _RunnerBudget(remaining_wall, clock, limits.monotonic_deadline)
    except BaseException as error:
        report = _preflight_report(config, reason=f"runtime budget setup failed: {_error_value(error)['message']}",
                                   policy_kind="checkpoint", action_selection="seeded-sampling", allow_final=False)
        closed, close_error = _close_resource(error, runtime, "runtime")
        if closed is not error and closed is not None and _is_process_control(closed):
            raise closed
        if _is_process_control(error):
            raise
        if close_error is not None:
            summary = dict(report.summary)
            summary["runtime_close_error"] = _error_value(close_error)
            report = NeuralReport(summary, report.exit_code, report.run_dir)
        return report
    try:
        _set_budget(runtime, budget)
        initial_recovery = _recovery_state(runtime)
    except BaseException as error:
        report = _preflight_report(config, reason=f"runtime setup failed: {_error_value(error)['message']}",
                                   policy_kind="checkpoint", action_selection="seeded-sampling", allow_final=False)
        closed, close_error = _close_resource(error, runtime, "runtime")
        if closed is not error and closed is not None and _is_process_control(closed):
            raise closed
        if _is_process_control(error):
            raise
        if close_error is not None:
            summary = dict(report.summary)
            summary["runtime_close_error"] = _error_value(close_error)
            report = NeuralReport(summary, report.exit_code, report.run_dir)
        return report

    writer: NeuralWriter | None = None
    run_dir: Path | None = None
    primary: BaseException | None = None
    records: list[StepRecord] = []
    cursors = {"allocations": 0, "checkpoints": 0}
    latest_diagnostics: CollectionDiagnostics | None = None
    status = "complete"
    stop_reason = "bounded_run_complete"
    fatal: BaseException | None = None
    runtime_close_attempted = False
    close_errors: list[BaseException] = []
    summary: dict[str, object] | None = None
    report: NeuralReport | None = None
    checkpoint_publication: dict[str, object] | None = None
    try:
        writer, run_dir = _writer_start(
            config, runtime, operation="train", policy_kind="checkpoint",
            action_selection="seeded-sampling", allow_final=False,
        )
        while True:
            if budget.exhausted():
                stop_reason = "wall_budget"
                break
            stop = _collection_budget_stop(config, runtime)
            if stop is not None:
                stop_reason = stop
                break
            try:
                rollout = runtime.collector.collect(
                    max_learned_decisions=_collection_learned_ceiling(config, runtime),
                    max_env_actions=_collection_environment_ceiling(config, runtime),
                    policy_version=_current_policy_version(runtime),
                )
            except BaseException as error:
                partial = getattr(runtime.collector, "partial_rollout", None)
                if isinstance(partial, Rollout):
                    records.extend(partial.records)
                    latest_diagnostics = partial.collection
                    if _is_process_control(error) or config.scenario_stream is not None:
                        try:
                            _write_collection_events(writer, partial, cursors=cursors)
                        except BaseException as artifact_error:
                            if _is_process_control(artifact_error):
                                error = _promote_finalization_control(error, artifact_error, "partial collection event")
                            else:
                                _note(error, f"partial collection event also failed: {type(artifact_error).__name__}: {_error_value(artifact_error)['message']}")
                    else:
                        _write_collection_events(writer, partial, cursors=cursors)
                        _update_once(config, runtime, writer, partial)
                if _is_process_control(error):
                    raise error
                fatal = error
                status = "operational_failure"
                stop_reason = "collection_fault"
                break

            records.extend(rollout.records)
            latest_diagnostics = rollout.collection
            _write_collection_events(writer, rollout, cursors=cursors)
            diagnostics = rollout.collection
            if diagnostics is not None and diagnostics.fault is not None:
                fatal = RuntimeError(f"collector reported {diagnostics.fault.phase} fault")
                status = "operational_failure"
                stop_reason = "collection_fault"
                break
            stop_after_update, update_error = _update_once(config, runtime, writer, rollout)
            if update_error is not None:
                fatal = update_error
                status = "operational_failure"
                stop_reason = "update_fault"
                break
            if stop_after_update:
                stop_reason = "update_budget_stop"
                break
            if diagnostics is not None and diagnostics.stop_reason in {
                "seed_domain_exhausted", "fault", "wall_budget", "inactive_reset_boundary",
            }:
                stop_reason = diagnostics.stop_reason
                break
            if not rollout.records:
                stop_reason = diagnostics.stop_reason if diagnostics is not None else "no_progress"
                break
            if budget.exhausted():
                stop_reason = "wall_budget"
                break

        if status == "complete":
            checkpoint_path = run_dir / "checkpoint-final"
            checkpoint_publication = {
                "checkpoint_path": str(checkpoint_path),
                "purpose": "final",
                "publication_state": "not_published",
            }
            try:
                runtime.save(checkpoint_path, "final")
            except BaseException:
                checkpoint_publication["publication_state"] = "possibly_published"
                raise
            checkpoint_publication["publication_state"] = "published"
            writer.event({"event": "checkpoint_published", "path": str(checkpoint_path), "purpose": "final"})
        runtime_close_attempted = True
        close_primary, runtime_close_error = _close_resource(None, runtime, "runtime")
        if runtime_close_error is not None:
            close_errors.append(runtime_close_error)
        if close_primary is not None and _is_process_control(close_primary):
            primary = close_primary
            raise close_primary
        if runtime_close_error is not None:
            fatal = runtime_close_error
            status = "operational_failure"
            stop_reason = "runtime_close_failure"
        elapsed = budget.elapsed()
        overshoot = max(0.0, elapsed - budget.allowed_seconds)
        summary = _base_summary(
            config, runtime, status=status, stop_reason=stop_reason, run_dir=run_dir,
            elapsed=elapsed, overshoot=overshoot, records=records,
            diagnostics=latest_diagnostics, recovery=initial_recovery, fatal_error=fatal,
            policy_kind="checkpoint", action_selection="seeded-sampling", allow_final=False,
        )
        if checkpoint_publication is not None:
            summary["checkpoint_publication"] = dict(checkpoint_publication)
        _mark_receipt_pending(summary, budget)
        writer.finalize(summary)
        report = NeuralReport(summary, 1, run_dir)
        return report
    except BaseException as error:
        primary = error
        if not _is_process_control(error):
            fatal = error
            status = "artifact_failure" if "writer" in str(error).casefold() or isinstance(error, OSError) else "operational_failure"
            stop_reason = "artifact_or_runner_failure"
            if not runtime_close_attempted:
                runtime_close_attempted = True
                close_primary, runtime_close_error = _close_resource(error, runtime, "runtime")
                if runtime_close_error is not None:
                    close_errors.append(runtime_close_error)
                if close_primary is not error and close_primary is not None and _is_process_control(close_primary):
                    primary = close_primary
                    raise close_primary
            if writer is not None:
                try:
                    elapsed = budget.elapsed()
                    summary = _base_summary(
                        config, runtime, status=status, stop_reason=stop_reason, run_dir=run_dir,
                        elapsed=elapsed, overshoot=max(0.0, elapsed - budget.allowed_seconds), records=records,
                        diagnostics=latest_diagnostics, recovery=initial_recovery, fatal_error=fatal,
                        policy_kind="checkpoint", action_selection="seeded-sampling", allow_final=False,
                    )
                    if checkpoint_publication is not None:
                        summary["checkpoint_publication"] = dict(checkpoint_publication)
                    _mark_receipt_pending(summary, budget)
                    writer.finalize(summary)
                except BaseException as summary_error:
                    _note(error, f"failure report finalization also failed: {type(summary_error).__name__}: {_error_value(summary_error)['message']}")
            if summary is None:
                summary = {"status": status, "stop_reason": stop_reason,
                           "fatal_error": _error_value(error), "run_directory": str(run_dir) if run_dir else None}
                if checkpoint_publication is not None:
                    summary["checkpoint_publication"] = dict(checkpoint_publication)
            try:
                _mark_receipt_pending(summary, budget)
            except BaseException as summary_error:
                _note(error, f"failure timing snapshot also failed: {type(summary_error).__name__}: {_error_value(summary_error)['message']}")
            report = NeuralReport(summary, 1, run_dir)
            return report
        raise
    finally:
        if not runtime_close_attempted:
            runtime_close_attempted = True
            primary, runtime_close_error = _close_resource(primary, runtime, "runtime")
            if runtime_close_error is not None:
                close_errors.append(runtime_close_error)
        if writer is not None:
            primary, writer_close_error = _close_resource(primary, writer, "artifact writer")
            if writer_close_error is not None:
                close_errors.append(writer_close_error)
        final_elapsed: float | None = None
        final_clock_error: BaseException | None = None
        try:
            final_elapsed = budget.elapsed()
        except BaseException as error:
            final_clock_error = error
            if primary is None:
                primary = error
            elif _is_process_control(error):
                primary = _promote_finalization_control(primary, error, "final lifecycle clock")
            else:
                _note(primary, f"final lifecycle clock read failed: {type(error).__name__}: {_error_value(error)['message']}")
        overall_status = status
        if close_errors or (final_clock_error is not None and not _is_process_control(final_clock_error)):
            if overall_status == "complete":
                overall_status = "operational_failure"
        if primary is not None and _is_process_control(primary):
            overall_status = "interrupted"
        if final_elapsed is None:
            try:
                final_elapsed = float(summary.get("elapsed_seconds", 0.0)) if summary else 0.0
            except (TypeError, ValueError):
                final_elapsed = 0.0
        receipt: dict[str, object] = {
            "schema_id": "spire-learn-neural-lifecycle-v1",
            "status": overall_status,
            "exit_code": 0 if overall_status == "complete" else 1,
            "elapsed_seconds": final_elapsed,
            "wall_overshoot_seconds": max(0.0, final_elapsed - budget.allowed_seconds),
            "timing_boundary": "after runtime and artifact writer close; before lifecycle receipt serialization and fsync",
            "close_errors": [_error_value(item) for item in close_errors],
            "checkpoint_publication": checkpoint_publication,
            "receipt_path": "lifecycle.json",
        }
        if final_clock_error is not None:
            receipt["clock_error"] = _error_value(final_clock_error)
        if fatal is not None:
            receipt["operation_error"] = _error_value(fatal)
        receipt_error: BaseException | None = None
        if writer is not None and run_dir is not None:
            try:
                _write_lifecycle_receipt(writer, run_dir, receipt)
            except BaseException as error:
                receipt_error = error
                if _is_process_control(error):
                    primary = _promote_finalization_control(primary, error, "lifecycle receipt write")
                    overall_status = "interrupted"
                    receipt["status"] = overall_status
                    receipt["exit_code"] = 1
                elif primary is not None:
                    _note(primary, f"lifecycle receipt write failed: {type(error).__name__}: {_error_value(error)['message']}")
                else:
                    overall_status = "artifact_failure"
                    receipt["status"] = overall_status
                    receipt["exit_code"] = 1
        if summary is not None:
            summary["elapsed_seconds"] = final_elapsed
            summary["wall_overshoot_seconds"] = max(0.0, final_elapsed - budget.allowed_seconds)
            summary["total_elapsed_seconds_with_resume_prefix"] = (
                float(getattr(runtime, "recovered_progress", {}).get("elapsed_seconds", 0.0)) + final_elapsed
                if isinstance(getattr(runtime, "recovered_progress", {}), Mapping) else final_elapsed
            )
            summary["status"] = overall_status
            summary["exit_code"] = receipt["exit_code"]
            summary["lifecycle_receipt_required"] = True
            summary["lifecycle"] = receipt
            if receipt_error is not None:
                summary["lifecycle_receipt_error"] = _error_value(receipt_error)
            if report is not None:
                report.exit_code = int(receipt["exit_code"])
        elif primary is not None and _is_process_control(primary) and run_dir is not None and writer is not None:
            try:
                _write_lifecycle_receipt(writer, run_dir, receipt)
            except BaseException as error:
                _note(primary, f"interruption lifecycle receipt write failed: {type(error).__name__}: {_error_value(error)['message']}")
        if primary is not None and _is_process_control(primary):
            raise primary


def _eval_complete(
    config: NeuralConfig,
    records: Sequence[StepRecord],
    diagnostics: CollectionDiagnostics | None,
    recovery: CollectorRecovery,
) -> bool:
    if config.mode == "run":
        return any(record.terminated for record in records)
    accounting = _case_accounting(config, records, diagnostics, recovery)
    return accounting["complete"] >= accounting["scheduled"]


def evaluate_policy(
    config: NeuralConfig,
    *,
    runtime: TrainingRuntime,
    clock: Callable[[], float] = time.monotonic,
    allow_final: bool = False,
) -> NeuralReport:
    """Evaluate one frozen policy or reference controller on its fixed manifest."""
    if not isinstance(config, NeuralConfig):
        primary, close_error = _close_resource(None, runtime, "runtime")
        if primary is not None and _is_process_control(primary):
            raise primary
        reason = "config must be NeuralConfig"
        if close_error is not None:
            reason += f"; runtime close failed: {_error_value(close_error)['message']}"
        return NeuralReport({"status": "operational_failure" if close_error else "setup_blocked",
                             "reason": reason}, 1 if close_error else 2, None)
    if config.operation != "evaluate" or config.split not in {"selection", "final"}:
        return _preflight_close(runtime, _preflight_report(
            config, reason="evaluation requires operation=evaluate and selection/final split",
            policy_kind=getattr(runtime, "policy_kind", "checkpoint"),
            action_selection=getattr(runtime, "action_selection", "deterministic"),
            allow_final=allow_final,
        ))
    if type(allow_final) is not bool:
        return _preflight_close(runtime, _preflight_report(
            config, reason="allow_final must be boolean",
            policy_kind=getattr(runtime, "policy_kind", "checkpoint"),
            action_selection=getattr(runtime, "action_selection", "deterministic"),
            allow_final=allow_final is True,
        ))
    policy_kind = getattr(runtime, "policy_kind", "checkpoint")
    action_selection = getattr(runtime, "action_selection", "deterministic")
    if config.split == "final" and not allow_final:
        return _preflight_close(runtime, _preflight_report(
            config, reason="final split requires --allow-final", policy_kind=policy_kind,
            action_selection=action_selection, allow_final=False,
        ))
    if config.rollout_rows != 1 or config.rollout_steps != 1:
        return _preflight_close(runtime, _preflight_report(
            config, reason="fixed-case evaluation requires rollout.rows=1 and rollout.steps=1 to avoid case rescheduling",
            policy_kind=policy_kind, action_selection=action_selection, allow_final=allow_final,
        ))
    if not callable(clock):
        return _preflight_close(runtime, _preflight_report(
            config, reason="clock must be callable", policy_kind=policy_kind,
            action_selection=action_selection, allow_final=allow_final,
        ))
    try:
        limits = _operation_limits(runtime)
        budget = _RunnerBudget(
            config.budgets.max_wall_seconds, clock, limits.monotonic_deadline,
        )
    except BaseException as error:
        report = _preflight_report(config, reason=f"runtime budget setup failed: {_error_value(error)['message']}",
                                   policy_kind=policy_kind, action_selection=action_selection,
                                   allow_final=allow_final)
        closed, close_error = _close_resource(error, runtime, "runtime")
        if closed is not error and closed is not None and _is_process_control(closed):
            raise closed
        if _is_process_control(error):
            raise
        if close_error is not None:
            summary = dict(report.summary)
            summary["runtime_close_error"] = _error_value(close_error)
            report = NeuralReport(summary, report.exit_code, report.run_dir)
        return report

    try:
        _set_budget(runtime, budget)
        initial_recovery = _recovery_state(runtime)
    except BaseException as error:
        report = _preflight_report(config, reason=f"runtime setup failed: {_error_value(error)['message']}",
                                   policy_kind=policy_kind, action_selection=action_selection,
                                   allow_final=allow_final)
        closed, close_error = _close_resource(error, runtime, "runtime")
        if closed is not error and closed is not None and _is_process_control(closed):
            raise closed
        if _is_process_control(error):
            raise
        if close_error is not None:
            summary = dict(report.summary)
            summary["runtime_close_error"] = _error_value(close_error)
            report = NeuralReport(summary, report.exit_code, report.run_dir)
        return report

    writer: NeuralWriter | None = None
    run_dir: Path | None = None
    primary: BaseException | None = None
    records: list[StepRecord] = []
    cursors = {"allocations": 0, "checkpoints": 0}
    latest_diagnostics: CollectionDiagnostics | None = None
    status = "complete"
    stop_reason = "fixed_cases_completed"
    fatal: BaseException | None = None
    runtime_close_attempted = False
    close_errors: list[BaseException] = []
    summary: dict[str, object] | None = None
    report: NeuralReport | None = None
    try:
        writer, run_dir = _writer_start(
            config, runtime, operation="evaluate", policy_kind=policy_kind,
            action_selection=action_selection, allow_final=allow_final,
        )
        while not _eval_complete(config, records, latest_diagnostics, initial_recovery):
            if budget.exhausted():
                stop_reason = "wall_time"
                break
            diagnostics = _current_collection(runtime)
            learned_cap = 0 if getattr(runtime.collector, "learned_combat_policy", True) is False else config.budgets.max_training_decisions
            try:
                rollout = runtime.collector.collect(
                    max_learned_decisions=learned_cap,
                    max_env_actions=config.budgets.max_environment_actions,
                    policy_version=_runtime_policy_version(runtime) or "reference-policy-v1",
                )
            except BaseException as error:
                partial = getattr(runtime.collector, "partial_rollout", None)
                if isinstance(partial, Rollout):
                    records.extend(partial.records)
                    latest_diagnostics = partial.collection
                    if _is_process_control(error):
                        try:
                            _write_collection_events(writer, partial, cursors=cursors)
                        except BaseException as artifact_error:
                            _note(error, f"partial evaluation event also failed: {type(artifact_error).__name__}: {_error_value(artifact_error)['message']}")
                    else:
                        _write_collection_events(writer, partial, cursors=cursors)
                if _is_process_control(error):
                    raise
                fatal = error
                status = "operational_failure"
                stop_reason = "collection_fault"
                break
            records.extend(rollout.records)
            latest_diagnostics = rollout.collection
            _write_collection_events(writer, rollout, cursors=cursors)
            if rollout.collection is not None and rollout.collection.fault is not None:
                fatal = RuntimeError(f"collector reported {rollout.collection.fault.phase} fault")
                status = "operational_failure"
                stop_reason = "collection_fault"
                break
            if _eval_complete(config, records, latest_diagnostics, initial_recovery):
                stop_reason = "fixed_cases_completed"
                break
            if config.mode == "run" and any(record.terminated for record in rollout.records):
                stop_reason = "full_run_terminal"
                break
            if rollout.collection is not None and rollout.collection.stop_reason in {
                "environment_action_budget", "learned_decision_budget", "seed_domain_exhausted", "wall_budget",
            }:
                stop_reason = rollout.collection.stop_reason
                break
            if not rollout.records:
                stop_reason = rollout.collection.stop_reason if rollout.collection is not None else "no_progress"
                status = "incomplete"
                break
            if budget.exhausted():
                stop_reason = "wall_time"
                break
        if status == "complete" and _case_accounting(
            config, records, latest_diagnostics, initial_recovery
        )["complete"] < len(config.cases.for_mode(config.mode)) and config.mode == "standalone":
            status = "incomplete"
            stop_reason = "fixed_case_budget_incomplete"
        runtime_close_attempted = True
        close_primary, runtime_close_error = _close_resource(None, runtime, "runtime")
        if runtime_close_error is not None:
            close_errors.append(runtime_close_error)
        if close_primary is not None and _is_process_control(close_primary):
            primary = close_primary
            raise close_primary
        if runtime_close_error is not None:
            fatal = runtime_close_error
            status = "operational_failure"
            stop_reason = "runtime_close_failure"
        elapsed = budget.elapsed()
        overshoot = max(0.0, elapsed - budget.allowed_seconds)
        summary = _base_summary(
            config, runtime, status=status, stop_reason=stop_reason, run_dir=run_dir,
            elapsed=elapsed, overshoot=overshoot, records=records,
            diagnostics=latest_diagnostics, recovery=initial_recovery, fatal_error=fatal,
            policy_kind=policy_kind, action_selection=action_selection, allow_final=allow_final,
        )
        if status == "complete" and summary["case_accounting"]["complete"] < summary["case_accounting"]["scheduled"]:
            status = "incomplete"
            summary["status"] = status
            summary["stop_reason"] = "fixed_case_budget_incomplete"
        _mark_receipt_pending(summary, budget)
        writer.finalize(summary)
        report = NeuralReport(summary, 1, run_dir)
        return report
    except BaseException as error:
        primary = error
        if not _is_process_control(error):
            fatal = error
            status = "artifact_failure" if isinstance(error, OSError) else "operational_failure"
            stop_reason = "artifact_or_runner_failure"
            if not runtime_close_attempted:
                runtime_close_attempted = True
                close_primary, runtime_close_error = _close_resource(error, runtime, "runtime")
                if runtime_close_error is not None:
                    close_errors.append(runtime_close_error)
                if close_primary is not error and close_primary is not None and _is_process_control(close_primary):
                    primary = close_primary
                    raise close_primary
            if writer is not None:
                try:
                    elapsed = budget.elapsed()
                    summary = _base_summary(
                        config, runtime, status=status, stop_reason=stop_reason, run_dir=run_dir,
                        elapsed=elapsed, overshoot=max(0.0, elapsed - budget.allowed_seconds),
                        records=records, diagnostics=latest_diagnostics, recovery=initial_recovery,
                        fatal_error=fatal, policy_kind=policy_kind,
                        action_selection=action_selection, allow_final=allow_final,
                    )
                    _mark_receipt_pending(summary, budget)
                    writer.finalize(summary)
                except BaseException as summary_error:
                    _note(error, f"failure report finalization also failed: {type(summary_error).__name__}: {_error_value(summary_error)['message']}")
            if summary is None:
                summary = {
                    "status": status, "stop_reason": stop_reason, "fatal_error": _error_value(error),
                    "run_directory": str(run_dir) if run_dir else None,
                }
                try:
                    _mark_receipt_pending(summary, budget)
                except BaseException as summary_error:
                    _note(error, f"failure timing snapshot also failed: {type(summary_error).__name__}: {_error_value(summary_error)['message']}")
            report = NeuralReport(summary, 1, run_dir)
            return report
        raise
    finally:
        if not runtime_close_attempted:
            runtime_close_attempted = True
            primary, runtime_close_error = _close_resource(primary, runtime, "runtime")
            if runtime_close_error is not None:
                close_errors.append(runtime_close_error)
        if writer is not None:
            primary, writer_close_error = _close_resource(primary, writer, "artifact writer")
            if writer_close_error is not None:
                close_errors.append(writer_close_error)
        final_clock_error: BaseException | None = None
        try:
            final_elapsed = budget.elapsed()
        except BaseException as error:
            final_elapsed = float(summary.get("elapsed_seconds", 0.0)) if summary else 0.0
            final_clock_error = error
            if primary is None:
                primary = error
            elif _is_process_control(error):
                primary = _promote_finalization_control(primary, error, "final lifecycle clock")
            else:
                _note(primary, f"final lifecycle clock read failed: {type(error).__name__}: {_error_value(error)['message']}")
        overall_status = status
        if close_errors or (final_clock_error is not None and not _is_process_control(final_clock_error)):
            if overall_status == "complete":
                overall_status = "operational_failure"
        if primary is not None and _is_process_control(primary):
            overall_status = "interrupted"
        receipt: dict[str, object] = {
            "schema_id": "spire-learn-neural-lifecycle-v1",
            "status": overall_status,
            "exit_code": 0 if overall_status == "complete" else 1,
            "elapsed_seconds": final_elapsed,
            "wall_overshoot_seconds": max(0.0, final_elapsed - budget.allowed_seconds),
            "timing_boundary": "after runtime and artifact writer close; before lifecycle receipt serialization and fsync",
            "close_errors": [_error_value(item) for item in close_errors],
            "receipt_path": "lifecycle.json",
        }
        if final_clock_error is not None:
            receipt["clock_error"] = _error_value(final_clock_error)
        if fatal is not None:
            receipt["operation_error"] = _error_value(fatal)
        receipt_error: BaseException | None = None
        if writer is not None and run_dir is not None:
            try:
                _write_lifecycle_receipt(writer, run_dir, receipt)
            except BaseException as error:
                receipt_error = error
                if _is_process_control(error):
                    primary = _promote_finalization_control(primary, error, "lifecycle receipt write")
                    overall_status = "interrupted"
                    receipt["status"] = overall_status
                    receipt["exit_code"] = 1
                elif primary is not None:
                    _note(primary, f"lifecycle receipt write failed: {type(error).__name__}: {_error_value(error)['message']}")
                else:
                    overall_status = "artifact_failure"
                    receipt["status"] = overall_status
                    receipt["exit_code"] = 1
        if summary is not None:
            summary["elapsed_seconds"] = final_elapsed
            summary["wall_overshoot_seconds"] = max(0.0, final_elapsed - budget.allowed_seconds)
            summary["status"] = overall_status
            summary["exit_code"] = receipt["exit_code"]
            summary["lifecycle_receipt_required"] = True
            summary["lifecycle"] = receipt
            if receipt_error is not None:
                summary["lifecycle_receipt_error"] = _error_value(receipt_error)
            if report is not None:
                report.exit_code = int(receipt["exit_code"])
        elif primary is not None and _is_process_control(primary) and run_dir is not None and writer is not None:
            try:
                _write_lifecycle_receipt(writer, run_dir, receipt)
            except BaseException as error:
                _note(primary, f"interruption lifecycle receipt write failed: {type(error).__name__}: {_error_value(error)['message']}")
        if primary is not None and _is_process_control(primary):
            raise primary


def calibrate(
    config: NeuralConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> NeuralReport:
    """Run the one explicitly bounded Task 8 CUDA calibration cohort lazily."""
    from spire_learn.neural.calibration import calibrate as run_calibration

    return run_calibration(config, clock=clock)


class _NeuralTrainingRuntime:
    def __init__(
        self,
        *,
        config: NeuralConfig,
        collector: Collector,
        policy: Any,
        optimizer: Any | None,
        torch: Any | None,
        policy_generator: Any | None,
        shuffle_generator: Any | None,
        curriculum_rng: random.Random,
        controller_rng: random.Random,
        device_plan: RuntimeDevicePlan,
        device_report: Mapping[str, object],
        budget: Any,
        recovery: CollectorRecovery | None,
        recovered_progress: Mapping[str, object] | None = None,
        parent_checkpoint: Path | None = None,
        policy_kind: str = "checkpoint",
        action_selection: str = "seeded-sampling",
        controller_id: str = "neural-policy-v1",
        stream_transition: Mapping[str, object] | None = None,
    ):
        self.config = config
        self.collector = collector
        self.policy = policy
        self.optimizer = optimizer
        self.torch = torch
        self.policy_generator = policy_generator
        self.shuffle_generator = shuffle_generator
        self.curriculum_rng = curriculum_rng
        self.controller_rng = controller_rng
        self.device_plan = device_plan
        self.device_report = dict(device_report)
        self.budget = budget
        self.operation_limits = RuntimeOperationLimits()
        self.collector_recovery = recovery
        self.recovered_progress = dict(recovered_progress or {})
        self.parent_checkpoint = parent_checkpoint
        self.policy_kind = policy_kind
        self.action_selection = action_selection
        self.controller_id = controller_id
        self.stream_transition = stream_transition
        self.optimizer_attempted_steps = int(self.recovered_progress.get("optimizer_attempted_steps", 0))
        self.optimizer_accepted_steps = int(self.recovered_progress.get("optimizer_accepted_steps", 0))
        self.optimizer_uncertain_steps = int(self.recovered_progress.get("optimizer_uncertain_steps", 0))
        self.update_count = int(self.recovered_progress.get("update_count", 0))
        self._closed = False
        self.last_update_prefix: dict[str, object] | None = None
        self._optimizer_step_start = optimizer.attempted if isinstance(optimizer, _CountingOptimizer) else 0

    def set_budget(self, budget: Any) -> None:
        self.budget = budget
        self.collector.budget = budget

    def record_update(self, metrics: UpdateMetrics) -> None:
        self.optimizer_attempted_steps += metrics.attempted_steps
        self.optimizer_accepted_steps += metrics.accepted_steps
        self.update_count += 1

    def record_update_fault(self, error: UpdateFault) -> None:
        self.optimizer_attempted_steps += error.attempted_steps
        self.optimizer_accepted_steps += error.accepted_steps
        self.optimizer_uncertain_steps += error.uncertain_steps
        self.update_count += 1

    def update(self, rollout: Rollout) -> UpdateMetrics:
        if self.config.operation != "train" or self.config.split != "train":
            raise ValueError("runtime update is train-split only")
        if self.optimizer is None or self.shuffle_generator is None:
            raise RuntimeError("runtime has no training optimizer")
        before_attempted = self.optimizer.attempted
        before_accepted = self.optimizer.accepted
        try:
            metrics = ppo_update(
                self.policy,
                self.optimizer,
                rollout,
                self.config.ppo,
                shuffle_generator=self.shuffle_generator,
                budget=self.budget,
            )
        except UpdateFault as error:
            self.last_update_prefix = {
                "attempted_steps": error.attempted_steps,
                "accepted_steps": error.accepted_steps,
                "uncertain_steps": error.uncertain_steps,
                "phase": error.phase,
                "resulting_behavior_policy_version": error.resulting_behavior_policy_version,
                "consumed_decisions": error.consumed_decisions,
            }
            raise
        except BaseException as error:
            attempted = self.optimizer.attempted - before_attempted
            accepted = self.optimizer.accepted - before_accepted
            uncertain = max(0, attempted - accepted)
            self.optimizer_attempted_steps += attempted
            self.optimizer_accepted_steps += accepted
            self.optimizer_uncertain_steps += uncertain
            self.last_update_prefix = {
                "attempted_steps": attempted,
                "accepted_steps": accepted,
                "uncertain_steps": uncertain,
                "resulting_behavior_policy_version": getattr(self.policy, "_behavior_policy_version", None),
            }
            try:
                error.optimizer_attempted_steps = attempted
                error.optimizer_accepted_steps = accepted
                error.optimizer_uncertain_steps = uncertain
            except BaseException:
                pass
            raise
        delta_attempted = self.optimizer.attempted - before_attempted
        delta_accepted = self.optimizer.accepted - before_accepted
        if (delta_attempted != metrics.attempted_steps or delta_accepted != metrics.accepted_steps):
            raise RuntimeError("optimizer wrapper counts disagree with UpdateMetrics")
        self.last_update_prefix = None
        return metrics

    def _latest_counters(self) -> dict[str, object]:
        diagnostics = _current_collection(self)
        current = _diagnostic_projection(diagnostics)
        result: dict[str, object] = {}
        for name in _PROGRESS_FIELDS:
            if name in {"optimizer_attempted_steps", "optimizer_accepted_steps", "optimizer_uncertain_steps", "update_count"}:
                result[name] = getattr(self, name)
            elif name == "incomplete_decisions":
                result[name] = int(self.recovered_progress.get(name, 0)) + int(current.get(name, 0))
            else:
                result[name] = int(self.recovered_progress.get(name, 0)) + int(current.get(name, 0))
        result["elapsed_seconds"] = float(self.recovered_progress.get("elapsed_seconds", 0.0)) + self.budget.elapsed()
        return result

    def save(self, path: Path, purpose: str) -> None:
        if self.config.operation != "train" or self.optimizer is None or self.torch is None:
            raise RuntimeError("only a training runtime can save training checkpoints")
        if purpose not in {"final", "resume"}:
            raise ValueError("checkpoint save purpose must be final or resume")
        recovery = _recovery_state(self)
        counters = self._latest_counters()
        identity = _selected_identity(self.config)
        rng_state = {
            "policy": self.policy_generator.get_state(),
            "minibatch": self.shuffle_generator.get_state(),
            "curriculum": self.curriculum_rng.getstate(),
            "controller": self.controller_rng.getstate(),
        }
        training_state = {
            "optimizer_step": int(counters["optimizer_accepted_steps"]),
            "hyperparameter_id": identity.config_hashes["ppo"],
            "rng_domains": ["policy", "minibatch", "curriculum", "controller"],
            "counters": counters,
            "update_id": f"update-{counters['update_count']}",
            "behavior_policy_version": self.policy._behavior_policy_version,
            "next_environment_seed_ordinal": recovery.next_seed_ordinal,
            "next_decision_sequence": recovery.next_decision_sequence,
            "saved_inflight_record_count": 0,
        }
        recovery_projection = {
            "next_seed_ordinal": recovery.next_seed_ordinal,
            "next_decision_sequence": recovery.next_decision_sequence,
            "next_episode_ordinals": list(recovery.next_episode_ordinals),
        }
        run_metadata = {
            "mode": self.config.mode,
            "operation": "train",
            "split": "train",
            "partition": "train",
            "config_sha256": self.config.config_sha256,
            "manifest_sha256": self.config.manifest_sha256,
            "environment_revision": self.config.environment_revision,
            "runtime_device": self.device_plan.device,
            "policy_rng_backend": self.device_plan.policy_rng_backend,
            "runtime_gpu_uuid": self.device_plan.gpu_uuid,
            "collector_recovery": recovery_projection,
            "cumulative_prefix": counters,
            "policy_kind": "checkpoint",
            "controller_id": self.controller_id,
        }
        if self.config.scenario_stream is not None:
            from .stream_resume import stream_checkpoint_state
            if self.stream_transition is not None:
                run_metadata["scenario_stream_transition"] = self.stream_transition
            sampler_state = self.collector.sampler_state
            run_metadata["scenario_stream"] = stream_checkpoint_state(
                self.config, {} if sampler_state is None else sampler_state
            )
        metadata = {
            "identity": identity_payload(identity),
            "run_metadata": run_metadata,
            "training_state": training_state,
            "parent_checkpoint": None if self.parent_checkpoint is None else {
                "path": str(self.parent_checkpoint),
            },
        }
        save_checkpoint(
            path,
            policy=self.policy,
            optimizer=self.optimizer,
            rng_state=rng_state,
            metadata=metadata,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.collector.close()


def _checkpoint_recovery(loaded: Any, config: NeuralConfig) -> tuple[CollectorRecovery, dict[str, object]]:
    metadata = loaded.metadata
    run_metadata = metadata.get("run_metadata")
    training_state = metadata.get("training_state")
    if not isinstance(run_metadata, Mapping) or not isinstance(training_state, Mapping):
        raise ValueError("resume checkpoint is missing run metadata or training state")
    for name, expected in (("mode", config.mode), ("operation", "train"), ("split", "train"), ("partition", "train")):
        if run_metadata.get(name) != expected:
            raise ValueError(f"resume checkpoint {name} does not match the selected config")
    if config.scenario_stream is not None:
        from .stream_resume import validate_stream_resume
        validate_stream_resume(run_metadata, config)
    else:
        if run_metadata.get("config_sha256") != config.config_sha256:
            raise ValueError("resume checkpoint config hash does not match the selected config")
        if run_metadata.get("manifest_sha256") != config.manifest_sha256:
            raise ValueError("resume checkpoint manifest hash does not match the selected config")
    if run_metadata.get("environment_revision") != config.environment_revision:
        raise ValueError("resume checkpoint environment revision does not match the selected config")
    raw_recovery = run_metadata.get("collector_recovery")
    if not isinstance(raw_recovery, Mapping) or set(raw_recovery) != {
        "next_seed_ordinal", "next_decision_sequence", "next_episode_ordinals",
    }:
        raise ValueError("resume checkpoint lacks the explicit collector recovery projection")
    episode_values = raw_recovery["next_episode_ordinals"]
    if not isinstance(episode_values, (list, tuple)):
        raise ValueError("resume checkpoint episode recovery state must be a sequence")
    recovery = CollectorRecovery(
        next_seed_ordinal=raw_recovery["next_seed_ordinal"],
        next_decision_sequence=raw_recovery["next_decision_sequence"],
        next_episode_ordinals=tuple(episode_values),
    )
    if len(recovery.next_episode_ordinals) != config.rollout_rows:
        raise ValueError("resume checkpoint recovery width does not match config rollout rows")
    if (training_state.get("next_environment_seed_ordinal") != recovery.next_seed_ordinal
            or training_state.get("next_decision_sequence") != recovery.next_decision_sequence):
        raise ValueError("resume checkpoint recovery scalars disagree with training state")
    raw_progress = training_state.get("counters")
    if not isinstance(raw_progress, Mapping) or set(raw_progress) != _PROGRESS_FIELDS | {"elapsed_seconds"}:
        raise ValueError("resume checkpoint counters do not match Task 7 progress fields")
    progress: dict[str, object] = {}
    for name in _PROGRESS_FIELDS:
        value = raw_progress[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"resume checkpoint counter {name} must be a non-negative integer")
        progress[name] = value
    elapsed = raw_progress["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("resume checkpoint elapsed_seconds must be finite and non-negative")
    progress["elapsed_seconds"] = float(elapsed)
    metadata_prefix = run_metadata.get("cumulative_prefix")
    if not isinstance(metadata_prefix, Mapping) or dict(metadata_prefix) != progress:
        raise ValueError("resume checkpoint run metadata cumulative prefix disagrees with training state")
    if recovery.next_seed_ordinal != progress["delivered_reset_slots"]:
        raise ValueError("resume checkpoint seed ordinal disagrees with delivered reset slots")
    if sum(recovery.next_episode_ordinals) != progress["delivered_reset_slots"]:
        raise ValueError("resume checkpoint episode progress disagrees with delivered reset slots")
    delivered_slots = progress["delivered_environment_slots"]
    if delivered_slots % config.rollout_rows or recovery.next_decision_sequence != delivered_slots // config.rollout_rows:
        raise ValueError("resume checkpoint decision sequence disagrees with accepted environment slots")
    if progress["optimizer_uncertain_steps"] != 0:
        raise ValueError("resume is forbidden after an uncertain optimizer step")
    if progress["optimizer_accepted_steps"] != training_state.get("optimizer_step"):
        raise ValueError("resume checkpoint optimizer step disagrees with cumulative counters")
    if progress["optimizer_attempted_steps"] != progress["optimizer_accepted_steps"]:
        raise ValueError("resume checkpoint contains an unresolved optimizer attempt")
    return recovery, progress


def create_runtime(
    config: NeuralConfig,
    *,
    purpose: str,
    checkpoint: Path | None = None,
    policy_kind: str = "checkpoint",
    action_selection: str = "seeded-sampling",
) -> TrainingRuntime:
    """Lazily assemble the selected device, model and training runtime."""
    if not isinstance(config, NeuralConfig):
        raise TypeError("config must be a NeuralConfig")
    if purpose not in {"train", "resume", "warm_start", "evaluate"}:
        raise ValueError("purpose must be train, resume, warm_start, or evaluate")
    if purpose in {"train", "resume", "warm_start"} and (config.operation != "train" or config.split != "train"):
        raise ValueError("training runtimes require config.operation=train and split=train")
    if purpose == "evaluate" and (config.operation != "evaluate" or config.split not in {"selection", "final"}):
        raise ValueError("evaluation runtimes require operation=evaluate and selection/final split")
    if purpose == "evaluate" and (config.rollout_rows != 1 or config.rollout_steps != 1):
        raise ValueError("fixed-case evaluation requires rollout.rows=1 and rollout.steps=1")
    if purpose in {"resume", "warm_start"} and checkpoint is None:
        raise ValueError(f"{purpose} requires a checkpoint path")
    if purpose == "train" and checkpoint is not None:
        raise ValueError("fresh training does not accept a checkpoint")
    if purpose == "evaluate":
        if policy_kind not in {"checkpoint", *REFERENCE_POLICIES}:
            raise ValueError("evaluation policy must be checkpoint, legal-random, or end-turn-last-v1")
        if policy_kind == "checkpoint" and checkpoint is None:
            raise ValueError("checkpoint evaluation requires a checkpoint path")
        if policy_kind != "checkpoint" and checkpoint is not None:
            raise ValueError("reference policies do not accept a checkpoint")
        if action_selection not in {"deterministic", "seeded-sampling"}:
            raise ValueError("action_selection must be deterministic or seeded-sampling")
        if policy_kind != "checkpoint" and policy_kind == "legal-random" and action_selection != "seeded-sampling":
            raise ValueError("legal-random evaluation requires seeded-sampling")
        if policy_kind == "end-turn-last-v1" and action_selection != "deterministic":
            raise ValueError("end-turn-last-v1 evaluation is deterministic")

    if config.scenario_stream is not None:
        from .stream_resume import stream_identity
        stream_identity(config)  # Pure admission before optional runtime/device effects.

    device_plan = plan_runtime_device(config)
    if (purpose == "evaluate" and policy_kind in REFERENCE_POLICIES
            and device_plan.device != "cpu"):
        raise ValueError("stateless reference policies require a CPU runtime config")

    selected_identity = _selected_identity(config)
    loaded = None
    recovery = None
    recovered_progress: dict[str, object] = {}
    load_purpose = None
    if checkpoint is not None:
        load_purpose = "resume" if purpose == "resume" else "warm_start" if purpose == "warm_start" else "evaluate"
        preflight = preflight_checkpoint(
            Path(checkpoint), selected_identity, purpose=load_purpose,
        )
        if purpose == "resume":
            run_metadata = preflight.metadata.get("run_metadata")
            if not isinstance(run_metadata, Mapping):
                raise ValueError("resume checkpoint is missing run metadata")
            validate_resume_device(run_metadata, device_plan)
            _checkpoint_recovery(SimpleNamespace(metadata=preflight.metadata), config)

    needs_torch = not (purpose == "evaluate" and policy_kind in REFERENCE_POLICIES)
    torch = None
    if needs_torch:
        require_runtime_execution_gate(device_plan)
        import torch as torch_module

        torch = torch_module
        device_report = qualify_runtime_device(torch, device_plan)
        from spire_learn.neural.model import build_policy
    else:
        device_report = qualify_runtime_device(None, device_plan)

    if checkpoint is not None:
        assert load_purpose is not None
        loaded = load_checkpoint(Path(checkpoint), selected_identity, purpose=load_purpose, device="cpu")
        if purpose == "resume":
            loaded_metadata = getattr(loaded, "metadata", None)
            if not isinstance(loaded_metadata, Mapping):
                raise ValueError("resume checkpoint metadata must be a mapping")
            run_metadata = loaded_metadata.get("run_metadata")
            if not isinstance(run_metadata, Mapping):
                raise ValueError("resume checkpoint is missing run metadata")
            validate_resume_device(run_metadata, device_plan)
            recovery, recovered_progress = _checkpoint_recovery(loaded, config)

    stream_transition = None
    if purpose == "resume" and config.scenario_stream is not None:
        from .stream_resume import validate_stream_resume
        assert loaded is not None
        stream_state = validate_stream_resume(loaded.metadata["run_metadata"], config)
        stream_transition = {"identity": stream_state["identity"],
                             "from": stream_state["current"],
                             "to": stream_state["destination"]}

    case_sampler = None
    if config.scenario_stream is not None and config.scenario_stream["arm"] == "mixed":
        from .curriculum import CurriculumSampler
        from .scenario_stream import generate_scenario_shard, load_scenario_recipe
        table = config.scenario_stream
        recipe = load_scenario_recipe(Path(table["resolved_recipe_path"]))
        shard = generate_scenario_shard(recipe, split="train",
            shard_index=table["shard_index"], arm=table["arm"])
        sampler_state = None
        if purpose == "resume":
            assert loaded is not None
            sampler_state = loaded.metadata["run_metadata"]["scenario_stream"]["sampler_state"]
        case_sampler = CurriculumSampler(shard.annotations,
            distribution_sha256=shard.distribution_sha256,
            categories=recipe.category_keys, state=sampler_state)
        case_sampler.validate_cases(config.cases.for_mode(config.mode))

    from spire_learn.neural.envs import make_driver
    driver = make_driver(config, config.cases)

    torch_generator = None
    shuffle_generator = None
    optimizer = None
    curriculum_rng = random.Random(config.seed_streams.curriculum)
    controller_rng = random.Random(config.seed_streams.noncombat)
    learned_combat_policy = True
    try:
        if purpose == "evaluate" and policy_kind in REFERENCE_POLICIES:
            learned_combat_policy = False
            policy = _ReferencePolicy(selected_identity, driver.schema, policy_kind)
            policy_generator = random.Random(config.seed_streams.policy)
        else:
            assert torch is not None
            policy = build_policy(
                driver.schema, config.model, seed=config.seed_streams.policy,
                device=device_plan.device,
            )
            if loaded is not None:
                policy.load_state_dict(dict(loaded.model_tensors), strict=True)
                training_state = loaded.metadata.get("training_state")
                if (purpose in {"resume", "evaluate"} and isinstance(training_state, Mapping)
                        and training_state.get("behavior_policy_version") is not None):
                    policy._behavior_policy_version = training_state["behavior_policy_version"]
            torch_generator = torch.Generator(device=device_plan.policy_generator_device)
            torch_generator.manual_seed(config.seed_streams.policy)
            shuffle_generator = torch.Generator(device=device_plan.shuffle_generator_device)
            shuffle_generator.manual_seed(config.seed_streams.optimizer)
            if purpose == "resume":
                assert loaded is not None and loaded.rng_state is not None
                torch_generator.set_state(loaded.rng_state["policy"])
                shuffle_generator.set_state(loaded.rng_state["minibatch"])
                curriculum_rng.setstate(loaded.rng_state["curriculum"])
                controller_rng.setstate(loaded.rng_state["controller"])
            policy_generator = torch_generator
            if purpose in {"train", "resume", "warm_start"}:
                optimizer = _CountingOptimizer(torch.optim.Adam(
                    policy.parameters(), lr=config.ppo.learning_rate, eps=config.ppo.adam_eps
                ))
                if purpose == "resume":
                    assert loaded is not None and loaded.optimizer_state is not None
                    optimizer.load_state_dict(loaded.optimizer_state)
            elif action_selection == "deterministic":
                policy = _DeterministicPolicy(policy)
    except BaseException as error:
        primary = _close_preserving(error, driver, "environment driver")
        if primary is not error:
            raise primary
        raise

    try:
        budget = _RunnerBudget(config.budgets.max_wall_seconds, time.monotonic, None)
        collector = Collector(
            driver,
            policy,
            config,
            controller_rng=controller_rng,
            curriculum_rng=curriculum_rng,
            budget=budget,
            policy_generator=policy_generator,
            recovery=recovery,
            learned_combat_policy=learned_combat_policy,
            case_sampler=case_sampler,
        )
    except BaseException as error:
        primary = _close_preserving(error, driver, "environment driver")
        if primary is not error:
            raise primary
        raise
    if purpose == "resume":
        controller_id = "neural-policy-resume-v1"
    elif purpose == "evaluate" and policy_kind in REFERENCE_POLICIES:
        controller_id = policy_kind
    elif purpose == "evaluate":
        controller_id = "neural-policy-checkpoint-v1"
    else:
        controller_id = "neural-policy-v1"
    try:
        return _NeuralTrainingRuntime(
            config=config,
            collector=collector,
            policy=policy if not isinstance(policy, _DeterministicPolicy) else policy._policy,
            optimizer=optimizer,
            torch=torch,
            policy_generator=policy_generator,
            shuffle_generator=shuffle_generator,
            curriculum_rng=curriculum_rng,
            controller_rng=controller_rng,
            device_plan=device_plan,
            device_report=device_report,
            budget=budget,
            recovery=recovery,
            recovered_progress=recovered_progress,
            parent_checkpoint=Path(checkpoint) if checkpoint is not None else None,
            policy_kind=policy_kind if purpose == "evaluate" else "checkpoint",
            action_selection=action_selection,
            controller_id=controller_id,
            stream_transition=stream_transition,
        )
    except BaseException as error:
        primary = _close_preserving(error, collector, "collector")
        if primary is not error:
            raise primary
        raise


__all__ = [
    "NeuralReport", "RuntimeOperationLimits", "TrainingRuntime", "calibrate",
    "create_runtime", "evaluate_policy", "train",
]
