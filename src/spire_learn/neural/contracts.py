"""Immutable, dependency-free records for neural combat learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from types import MappingProxyType

from spire_learn.contracts import ContractError
from spire_learn.neural.views import COMMON_V1, VIEWS, view_for_fingerprint


_FINGERPRINT = re.compile(r"[0-9a-fA-F]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*\Z")
_COMMON_VIEW_FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"
_ALL_VIEW_OUTCOMES = frozenset(outcome for view in VIEWS for outcome in view.outcomes)


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    return float(value)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{name} must be a non-empty identifier")
    return value


def _freeze_json(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError("metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError("metadata mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    raise ContractError("metadata must contain JSON-compatible values")


def _frozen_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) or not key for key in value):
        raise ContractError(f"{name} must map non-empty strings to values")
    frozen = _freeze_json(value)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True)
class BufferArray:
    """Owned native-endian int8/int64 tensor bytes with a checked shape."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str) or self.dtype not in {"int8", "int64"}:
            raise ContractError("BufferArray dtype must be native-endian int8 or int64")
        if not isinstance(self.shape, tuple) or any(type(dim) is not int or dim < 0 for dim in self.shape):
            raise ContractError("BufferArray shape must contain non-negative integer dimensions")
        if type(self.data) is not bytes:
            raise ContractError("BufferArray data must be immutable bytes")
        item_size = 1 if self.dtype == "int8" else 8
        expected = math.prod(self.shape) * item_size
        if len(self.data) != expected:
            raise ContractError(f"BufferArray shape requires {expected} bytes, got {len(self.data)}")


@dataclass(frozen=True)
class CombatFrame:
    """Actor-facing feature tensors only; lifecycle and seed metadata stay outside."""

    tensors: Mapping[str, BufferArray]
    schema_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, Mapping):
            raise ContractError("CombatFrame tensors must be a mapping")
        copied = dict(self.tensors)
        if not isinstance(self.schema_fingerprint, str) or _FINGERPRINT.fullmatch(self.schema_fingerprint) is None:
            raise ContractError("schema_fingerprint must be 64 hexadecimal characters")
        # Synthetic fixtures may patch the module-level common fingerprint.
        if self.schema_fingerprint == _COMMON_VIEW_FINGERPRINT:
            view = COMMON_V1
        else:
            try:
                view = view_for_fingerprint(self.schema_fingerprint)
            except ContractError:
                raise ContractError("CombatFrame schema_fingerprint is not an accepted combat view") from None
        allowed = view.tensor_names
        if any(not isinstance(name, str) or name not in allowed for name in copied):
            raise ContractError("CombatFrame contains a tensor outside the public actor-input descriptor")
        if any(not isinstance(value, BufferArray) for value in copied.values()):
            raise ContractError("CombatFrame tensors must be BufferArray records")
        object.__setattr__(self, "tensors", MappingProxyType(copied))


@dataclass(frozen=True)
class Lifecycle:
    """Run/combat routing metadata, kept separate from actor-visible frames."""

    active: bool
    routing: str
    selection_owner: str | None
    active_visit: int | None
    combats_entered: int
    combats_settled: int
    last_exit: Mapping[str, object] | None
    phase: str | None = None
    selection_resume_phase: str | None = None

    def __post_init__(self) -> None:
        if type(self.active) is not bool:
            raise ContractError("Lifecycle active must be a boolean")
        if not isinstance(self.routing, str) or self.routing not in {
            "noncombat", "combat", "combat_selection",
        }:
            raise ContractError("Lifecycle routing must be noncombat, combat, or combat_selection")
        if self.routing == "combat_selection":
            if (not isinstance(self.selection_owner, str)
                    or self.selection_owner not in {"card", "potion", "end_turn", "relic", "power"}):
                raise ContractError("combat_selection requires a recognized selection_owner")
        elif self.selection_owner is not None:
            raise ContractError("selection_owner must be None when routing has no selector")
        phases = {"decision", "selection", "player_action", "enemy_turn", "round_end",
                  "player_start", "terminal"}
        for name in ("phase", "selection_resume_phase"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or value not in phases):
                raise ContractError(f"Lifecycle {name} must be a public combat phase or None")
        if self.active_visit is not None and (type(self.active_visit) is not int or self.active_visit < 0):
            raise ContractError("Lifecycle active_visit must be a non-negative integer or None")
        for name, value in (("combats_entered", self.combats_entered),
                            ("combats_settled", self.combats_settled)):
            if type(value) is not int or value < 0:
                raise ContractError(f"Lifecycle {name} must be a non-negative integer")
        if self.combats_settled > self.combats_entered:
            raise ContractError("Lifecycle settled combats cannot exceed entered combats")
        if self.active:
            if self.routing not in {"combat", "combat_selection"}:
                raise ContractError("active lifecycle must route to combat or combat_selection")
            if self.combats_entered < 1 or self.active_visit != self.combats_entered - 1:
                raise ContractError("active_visit must identify the latest entered combat")
        elif self.routing != "noncombat" or self.active_visit is not None:
            raise ContractError("inactive lifecycle must use noncombat routing with no active_visit")
        if self.combats_settled > 0 and self.last_exit is None:
            raise ContractError("settled combats require a public last_exit record")
        if self.last_exit is not None:
            frozen = _frozen_mapping(self.last_exit, "Lifecycle last_exit")
            required_exit_fields = {
                "visit", "encounter", "outcome", "hp_before_healing", "hp_after_cleanup",
            }
            if not required_exit_fields <= frozen.keys():
                raise ContractError("Lifecycle last_exit is missing required public exit fields")
            visit = frozen["visit"]
            if type(visit) is not int or visit < 0:
                raise ContractError("Lifecycle last_exit.visit must be a non-negative integer")
            if visit >= self.combats_entered or visit >= self.combats_settled:
                raise ContractError("Lifecycle last_exit.visit is outside entered/settled counters")
            if self.combats_settled and visit != self.combats_settled - 1:
                raise ContractError("Lifecycle last_exit.visit must be the latest settled combat")
            encounter = frozen["encounter"]
            if not isinstance(encounter, str) or not encounter:
                raise ContractError("Lifecycle last_exit.encounter must be a non-empty string")
            outcome = frozen.get("outcome")
            if not isinstance(outcome, str) or outcome not in _ALL_VIEW_OUTCOMES:
                raise ContractError("Lifecycle last_exit.outcome is not accepted by a combat view")
            for name in ("hp_before_healing", "hp_after_cleanup"):
                hp = frozen[name]
                if type(hp) is not int or hp < 0:
                    raise ContractError(f"Lifecycle last_exit.{name} must be a non-negative integer")
            object.__setattr__(self, "last_exit", frozen)


@dataclass(frozen=True)
class TaskSignal:
    """Combat-objective reward and independent bootstrap/GAE boundary flags."""

    reward: float
    task_terminal: bool
    bootstrap: bool
    continuation: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward", _finite_number(self.reward, "TaskSignal reward"))
        for name in ("task_terminal", "bootstrap", "continuation"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"TaskSignal {name} must be a boolean")
        if self.task_terminal and (self.bootstrap or self.continuation):
            raise ContractError("terminal tasks cannot bootstrap or continue GAE")
        if not self.task_terminal and not self.bootstrap:
            raise ContractError("nonterminal tasks must bootstrap from their actual successor")


@dataclass(frozen=True)
class PolicyChoice:
    """Original decision-local action row and optional learned-policy statistics."""

    original_action_index: int
    log_probability: float | None = None
    value: float | None = None
    behavior_policy_version: str | int | None = None

    def __post_init__(self) -> None:
        if type(self.original_action_index) is not int or self.original_action_index < 0:
            raise ContractError("original_action_index must be a non-negative integer")
        present = (self.log_probability is not None, self.value is not None,
                   self.behavior_policy_version is not None)
        if any(present) and not all(present):
            raise ContractError("learned choice statistics must be present together")
        if self.log_probability is not None:
            object.__setattr__(self, "log_probability",
                               _finite_number(self.log_probability, "log_probability"))
            object.__setattr__(self, "value", _finite_number(self.value, "value"))
            version = self.behavior_policy_version
            if type(version) is int:
                if version < 0:
                    raise ContractError("behavior_policy_version must be non-negative")
            else:
                _identifier(version, "behavior_policy_version")


@dataclass(frozen=True)
class StepRecord:
    """One accepted environment effect, attributed to a stream and learner task."""

    stream_id: str
    task_id: str
    decision_id: str
    source: CombatFrame
    successor: CombatFrame
    native_reward: float
    terminated: bool
    truncated: bool
    task_signal: TaskSignal
    choice: PolicyChoice
    actor_training_mask: bool
    next_value: float | None

    def __post_init__(self) -> None:
        for name in ("stream_id", "task_id", "decision_id"):
            _identifier(getattr(self, name), f"StepRecord {name}")
        if not isinstance(self.source, CombatFrame) or not isinstance(self.successor, CombatFrame):
            raise ContractError("StepRecord source and successor must be CombatFrame records")
        if self.source.schema_fingerprint != self.successor.schema_fingerprint:
            raise ContractError("StepRecord frames must use the same schema fingerprint")
        object.__setattr__(self, "native_reward", _finite_number(self.native_reward, "native_reward"))
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ContractError("StepRecord native end flags must be booleans")
        if self.terminated and self.truncated:
            raise ContractError("a StepRecord cannot be both terminated and truncated")
        if not isinstance(self.task_signal, TaskSignal) or not isinstance(self.choice, PolicyChoice):
            raise ContractError("StepRecord requires TaskSignal and PolicyChoice records")
        if self.terminated and (
            not self.task_signal.task_terminal
            or self.task_signal.bootstrap
            or self.task_signal.continuation
        ):
            raise ContractError("native termination requires a terminal signal with no bootstrap or continuation")
        if self.truncated:
            if self.task_signal.continuation:
                raise ContractError("native truncation cannot continue GAE recursion")
            if self.task_signal.bootstrap and self.task_signal.task_terminal:
                raise ContractError("only a nonterminal supported truncation may bootstrap")
        if type(self.actor_training_mask) is not bool:
            raise ContractError("actor_training_mask must be a boolean")
        has_policy_statistics = self.choice.value is not None
        if self.actor_training_mask:
            if not has_policy_statistics or self.next_value is None:
                raise ContractError("learned rows require policy statistics and actual-successor value")
            object.__setattr__(self, "next_value", _finite_number(self.next_value, "next_value"))
        elif has_policy_statistics or self.next_value is not None:
            raise ContractError("fixed-controller rows cannot carry fabricated policy/value statistics")


@dataclass(frozen=True)
class ReturnBatch:
    """Aligned numeric targets; false-mask entries are zero placeholders to ignore."""

    advantages: tuple[float, ...]
    targets: tuple[float, ...]
    train_mask: tuple[bool, ...]
    decision_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("advantages", "targets", "train_mask", "decision_ids"):
            if not isinstance(getattr(self, name), tuple):
                raise ContractError(f"ReturnBatch {name} must be an immutable tuple")
        sizes = {len(self.advantages), len(self.targets), len(self.train_mask), len(self.decision_ids)}
        if len(sizes) != 1:
            raise ContractError("ReturnBatch arrays must have equal lengths")
        for name in ("advantages", "targets"):
            values = tuple(_finite_number(value, f"ReturnBatch {name} entry")
                           for value in getattr(self, name))
            object.__setattr__(self, name, values)
        if any(type(mask) is not bool for mask in self.train_mask):
            raise ContractError("ReturnBatch train_mask entries must be booleans")
        if any(not isinstance(value, str) or not value for value in self.decision_ids):
            raise ContractError("ReturnBatch decision IDs must be non-empty strings")


@dataclass(frozen=True)
class ModelConfig:
    """Small, serializable policy/value network settings; no framework import."""

    embedding_width: int
    hidden_width: int
    max_parameters: int

    def __post_init__(self) -> None:
        for name in ("embedding_width", "hidden_width", "max_parameters"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ContractError(f"ModelConfig {name} must be a positive integer")
        if self.max_parameters > 10_000_000:
            raise ContractError("ModelConfig max_parameters cannot exceed 10,000,000")


@dataclass(frozen=True)
class PPOConfig:
    """Validated scalar PPO settings shared by the later learner tasks."""

    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    epochs: int
    minibatch_rows: int
    adam_eps: float

    def __post_init__(self) -> None:
        scalar_names = (
            "learning_rate", "gamma", "gae_lambda", "clip_epsilon",
            "value_coefficient", "entropy_coefficient", "max_grad_norm", "adam_eps",
        )
        for name in scalar_names:
            value = _finite_number(getattr(self, name), f"PPOConfig {name}")
            object.__setattr__(self, name, value)
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.adam_eps <= 0:
            raise ContractError("PPO learning rate, max_grad_norm, and adam_eps must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ContractError("PPO gamma and gae_lambda must be in [0, 1]")
        if not 0 < self.clip_epsilon < 1:
            raise ContractError("PPO clip_epsilon must be in (0, 1)")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ContractError("PPO loss coefficients must be non-negative")
        for name in ("epochs", "minibatch_rows"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ContractError(f"PPO {name} must be a positive integer")


@dataclass(frozen=True)
class NeuralIdentity:
    """Checkpoint lineage and explicit compatibility with public raw-mode views."""

    model_id: str
    features_id: str
    action_id: str
    objective_id: str
    accepted_raw_mode_mappings: Mapping[str, str]
    config_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("model_id", "features_id", "action_id", "objective_id"):
            _identifier(getattr(self, name), f"NeuralIdentity {name}")
        if not isinstance(self.accepted_raw_mode_mappings, Mapping):
            raise ContractError("NeuralIdentity accepted_raw_mode_mappings must be a mapping")
        mappings = dict(self.accepted_raw_mode_mappings)
        if not mappings or any(mode not in {"standalone", "run"} for mode in mappings):
            raise ContractError("NeuralIdentity must declare accepted standalone/run mappings")
        for mode, mapping_id in mappings.items():
            _identifier(mapping_id, f"NeuralIdentity {mode} mapping")
        if not isinstance(self.config_hashes, Mapping):
            raise ContractError("NeuralIdentity config_hashes must be a mapping")
        hashes = dict(self.config_hashes)
        if not hashes or any(not isinstance(name, str) or not name for name in hashes):
            raise ContractError("NeuralIdentity config_hashes must be a non-empty named mapping")
        if any(not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None
               for value in hashes.values()):
            raise ContractError("NeuralIdentity config hashes must be 64 hexadecimal characters")
        object.__setattr__(self, "accepted_raw_mode_mappings", MappingProxyType(mappings))
        object.__setattr__(self, "config_hashes", MappingProxyType(hashes))


@dataclass(frozen=True)
class CumulativeCounters:
    """Monotonic collector counters attached to one immutable rollout snapshot."""

    rows: int
    actor_rows: int
    fixed_rows: int
    task_terminals: int
    environment_terminations: int
    environment_truncations: int

    def __post_init__(self) -> None:
        values = (self.rows, self.actor_rows, self.fixed_rows, self.task_terminals,
                  self.environment_terminations, self.environment_truncations)
        if any(type(value) is not int or value < 0 for value in values):
            raise ContractError("cumulative counters must be non-negative integers")
        if self.actor_rows + self.fixed_rows != self.rows:
            raise ContractError("cumulative actor/fixed rows must sum to rows")
        if self.task_terminals > self.rows or self.environment_terminations > self.rows:
            raise ContractError("cumulative event counters cannot exceed rows")
        if self.environment_truncations > self.rows:
            raise ContractError("cumulative truncation counter cannot exceed rows")


# Kept as a concise public spelling for callers constructing rollout snapshots.
RolloutCounters = CumulativeCounters


@dataclass(frozen=True)
class SeedAllocation:
    """One reset's authored case identity and actual environment seed."""

    stream_id: str
    slot: int
    case_id: str
    entry_id: str | None
    authored_reference_seed: int
    actual_seed: int
    ordinal: int
    allocation_policy_id: str
    delivery: str = "delivered"

    def __post_init__(self) -> None:
        _identifier(self.stream_id, "SeedAllocation stream_id")
        _identifier(self.case_id, "SeedAllocation case_id")
        _identifier(self.allocation_policy_id, "SeedAllocation allocation_policy_id")
        if self.entry_id is not None:
            _identifier(self.entry_id, "SeedAllocation entry_id")
        if type(self.slot) is not int or self.slot < 0:
            raise ContractError("SeedAllocation slot must be a non-negative integer")
        for name in ("authored_reference_seed", "actual_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**64:
                raise ContractError(f"SeedAllocation {name} must be an unsigned 64-bit integer")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ContractError("SeedAllocation ordinal must be a non-negative integer")
        if self.delivery not in {
            "attempted", "delivered", "uncertain", "accepted-undelivered", "rejected",
        }:
            raise ContractError("SeedAllocation delivery status is unsupported")


@dataclass(frozen=True)
class ResourceCheckpoint:
    """Public owned combat frame captured at one newly settled visit."""

    stream_id: str
    task_id: str
    visit: int
    outcome: str
    frame: CombatFrame
    encounter: str
    hp_before_healing: int
    hp_after_cleanup: int

    def __post_init__(self) -> None:
        _identifier(self.stream_id, "ResourceCheckpoint stream_id")
        _identifier(self.task_id, "ResourceCheckpoint task_id")
        if type(self.visit) is not int or self.visit < 0:
            raise ContractError("ResourceCheckpoint visit must be a non-negative integer")
        if not isinstance(self.frame, CombatFrame):
            raise ContractError("ResourceCheckpoint frame must be a public CombatFrame")
        if not isinstance(self.outcome, str) or self.outcome not in _ALL_VIEW_OUTCOMES:
            raise ContractError("ResourceCheckpoint outcome is not accepted by a combat view")
        if not isinstance(self.encounter, str) or not self.encounter:
            raise ContractError("ResourceCheckpoint encounter must be a non-empty string")
        for name in ("hp_before_healing", "hp_after_cleanup"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"ResourceCheckpoint {name} must be a non-negative integer")


@dataclass(frozen=True)
class IncompleteTask:
    """An open task left at a collection horizon or cumulative ceiling."""

    stream_id: str
    task_id: str
    decisions: int

    def __post_init__(self) -> None:
        _identifier(self.stream_id, "IncompleteTask stream_id")
        _identifier(self.task_id, "IncompleteTask task_id")
        if type(self.decisions) is not int or self.decisions <= 0:
            raise ContractError("IncompleteTask decisions must be positive")


@dataclass(frozen=True)
class CollectionFault:
    """Bounded primitive fault evidence retained without exception/native objects."""

    phase: str
    exception_type: str
    message: str
    operation_slots: int = 0
    known_accepted_slots: int = 0
    uncertain_slots: int = 0

    def __post_init__(self) -> None:
        for name in ("phase", "exception_type"):
            _identifier(getattr(self, name), f"CollectionFault {name}")
        if not isinstance(self.message, str) or len(self.message) > 512:
            raise ContractError("CollectionFault message must be at most 512 characters")
        for name in ("operation_slots", "known_accepted_slots", "uncertain_slots"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"CollectionFault {name} must be non-negative")
        if self.known_accepted_slots > self.operation_slots:
            raise ContractError("CollectionFault known accepted slots exceed operation width")
        if self.uncertain_slots > self.operation_slots:
            raise ContractError("CollectionFault uncertain slots exceed operation width")


@dataclass(frozen=True)
class CollectionDiagnostics:
    """Immutable cumulative acceptance, reset, seed, task and time evidence."""

    attempted_environment_slots: int = 0
    forwarded_environment_slots: int = 0
    known_accepted_environment_slots: int = 0
    delivered_environment_slots: int = 0
    uncertain_environment_slots: int = 0
    reset_operations: int = 0
    reset_slots: int = 0
    attempted_reset_slots: int = 0
    forwarded_reset_slots: int = 0
    known_accepted_reset_slots: int = 0
    delivered_reset_slots: int = 0
    uncertain_reset_slots: int = 0
    learned_decisions: int = 0
    fixed_decisions: int = 0
    tasks_started: int = 0
    task_terminals: int = 0
    tasks_completed: int = 0
    victories: int = 0
    deaths: int = 0
    escaped: int = 0
    faults: int = 0
    elapsed_seconds: float = 0.0
    wall_overshoot_seconds: float = 0.0
    incomplete_decisions: int = 0
    incomplete_environment_slots: int = 0
    next_seed_ordinal: int = 0
    seed_allocations: tuple[SeedAllocation, ...] = ()
    resource_checkpoints: tuple[ResourceCheckpoint, ...] = ()
    incomplete_tasks: tuple[IncompleteTask, ...] = ()
    stop_reason: str | None = None
    fault: CollectionFault | None = None
    undispatched_actor_decisions: int = 0
    undispatched_fixed_decisions: int = 0
    elapsed_seconds_is_fallback: bool = False

    def __post_init__(self) -> None:
        integer_names = (
            "attempted_environment_slots", "forwarded_environment_slots",
            "known_accepted_environment_slots", "delivered_environment_slots",
            "uncertain_environment_slots", "reset_operations", "reset_slots",
            "attempted_reset_slots", "forwarded_reset_slots",
            "known_accepted_reset_slots", "delivered_reset_slots",
            "uncertain_reset_slots", "learned_decisions", "fixed_decisions",
            "tasks_started", "task_terminals", "tasks_completed", "victories",
            "deaths", "escaped", "faults", "incomplete_decisions",
            "incomplete_environment_slots", "next_seed_ordinal",
            "undispatched_actor_decisions", "undispatched_fixed_decisions",
        )
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"CollectionDiagnostics {name} must be non-negative")
        for name in ("elapsed_seconds", "wall_overshoot_seconds"):
            value = _finite_number(getattr(self, name), f"CollectionDiagnostics {name}")
            if value < 0:
                raise ContractError(f"CollectionDiagnostics {name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.forwarded_environment_slots > self.attempted_environment_slots:
            raise ContractError("forwarded environment slots exceed attempted slots")
        if self.known_accepted_environment_slots > self.forwarded_environment_slots:
            raise ContractError("known accepted environment slots exceed forwarded slots")
        if self.delivered_environment_slots > self.known_accepted_environment_slots:
            raise ContractError("delivered environment slots exceed known accepted slots")
        if self.uncertain_environment_slots > self.forwarded_environment_slots:
            raise ContractError("uncertain environment slots exceed forwarded slots")
        if self.forwarded_reset_slots > self.attempted_reset_slots:
            raise ContractError("forwarded reset slots exceed attempted reset slots")
        if self.known_accepted_reset_slots > self.forwarded_reset_slots:
            raise ContractError("known accepted reset slots exceed forwarded slots")
        if self.delivered_reset_slots > self.known_accepted_reset_slots:
            raise ContractError("delivered reset slots exceed known accepted slots")
        if self.uncertain_reset_slots > self.forwarded_reset_slots:
            raise ContractError("uncertain reset slots exceed forwarded slots")
        if self.tasks_completed > self.task_terminals:
            raise ContractError("completed tasks cannot exceed task terminals")
        if self.victories + self.deaths + self.escaped > self.tasks_completed:
            raise ContractError("natural outcomes cannot exceed completed tasks")
        if any(not isinstance(values, tuple) for values in (
            self.seed_allocations, self.resource_checkpoints, self.incomplete_tasks,
        )):
            raise ContractError("collection diagnostic histories must be immutable tuples")
        if any(not isinstance(value, SeedAllocation) for value in self.seed_allocations):
            raise ContractError("seed_allocations must contain SeedAllocation records")
        if any(not isinstance(value, ResourceCheckpoint) for value in self.resource_checkpoints):
            raise ContractError("resource_checkpoints must contain ResourceCheckpoint records")
        if any(not isinstance(value, IncompleteTask) for value in self.incomplete_tasks):
            raise ContractError("incomplete_tasks must contain IncompleteTask records")
        if self.stop_reason is not None:
            _identifier(self.stop_reason, "CollectionDiagnostics stop_reason")
        if self.fault is not None and not isinstance(self.fault, CollectionFault):
            raise ContractError("CollectionDiagnostics fault must be CollectionFault or None")
        if type(self.elapsed_seconds_is_fallback) is not bool:
            raise ContractError("CollectionDiagnostics elapsed_seconds_is_fallback must be a boolean")


@dataclass(frozen=True)
class Rollout:
    """One frozen set of task-grouped transitions and its producing identity."""

    mode: str
    split: str
    identity: NeuralIdentity
    records: tuple[StepRecord, ...]
    counters: CumulativeCounters
    collection: CollectionDiagnostics | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.mode, str) or self.mode not in {"standalone", "run"}
                or not isinstance(self.split, str) or self.split not in {"train", "selection", "final"}):
            raise ContractError("Rollout mode or split is unsupported")
        if not isinstance(self.identity, NeuralIdentity):
            raise ContractError("Rollout identity must be NeuralIdentity")
        if not isinstance(self.records, tuple) or any(not isinstance(row, StepRecord) for row in self.records):
            raise ContractError("Rollout records must be an immutable tuple of StepRecord values")
        if not isinstance(self.counters, CumulativeCounters):
            raise ContractError("Rollout counters must be CumulativeCounters")
        if self.collection is not None and not isinstance(self.collection, CollectionDiagnostics):
            raise ContractError("Rollout collection must be CollectionDiagnostics or None")
        if self.counters.rows < len(self.records):
            raise ContractError("cumulative row count cannot be smaller than this rollout")
        current_actor_rows = sum(row.actor_training_mask for row in self.records)
        current_fixed_rows = len(self.records) - current_actor_rows
        if self.counters.actor_rows < current_actor_rows or self.counters.fixed_rows < current_fixed_rows:
            raise ContractError("cumulative actor/fixed counters cannot be smaller than this rollout")
        if self.counters.task_terminals < sum(row.task_signal.task_terminal for row in self.records):
            raise ContractError("cumulative task terminal count cannot be smaller than this rollout")
        if self.counters.environment_terminations < sum(row.terminated for row in self.records):
            raise ContractError("cumulative termination count cannot be smaller than this rollout")
        if self.counters.environment_truncations < sum(row.truncated for row in self.records):
            raise ContractError("cumulative truncation count cannot be smaller than this rollout")
        if self.mode not in self.identity.accepted_raw_mode_mappings:
            raise ContractError("rollout identity does not accept its raw mode")
        seen: set[tuple[str, str]] = set()
        for row in self.records:
            key = (row.stream_id, row.decision_id)
            if key in seen:
                raise ContractError("Rollout contains duplicate decision IDs within a stream")
            seen.add(key)
