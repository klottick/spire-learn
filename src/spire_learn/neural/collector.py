"""Persistent, dependency-light collection over the accepted combat driver."""

from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.neural.cases import NeuralCase, environment_seed_for
from spire_learn.neural.config import NeuralConfig
from spire_learn.neural.contracts import (
    CollectionDiagnostics,
    CollectionFault,
    CumulativeCounters,
    IncompleteTask,
    NeuralIdentity,
    PolicyChoice,
    ResourceCheckpoint,
    Rollout,
    SeedAllocation,
    StepRecord,
)
from spire_learn.neural.envs import DriverFault, DriverRow, DriverTransition
from spire_learn.neural.returns import task_signal


_TRAIN_SEED_POLICY = "environment-seed-for-replica0-v1"
_REFERENCE_SEED_POLICY = "authored-reference-seed-v1"
_MAX_FAULT_TEXT = 512


def _clean_error(error: BaseException) -> str:
    text = str(error).replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    return text if len(text) <= _MAX_FAULT_TEXT else text[:_MAX_FAULT_TEXT - 1] + "…"


def _valid_policy_version(value: object) -> bool:
    if type(value) is int:
        return value >= 0
    if not isinstance(value, str) or not value:
        return False
    return all(character.isalnum() or character in "_.:/@+-" for character in value)


class Collector:
    """Collect full native batches while retaining stream and RNG state.

    The two injected Python RNGs have disjoint jobs: ``controller_rng`` chooses
    legal Run-mode noncombat actions, while ``curriculum_rng`` chooses training
    cases. The optional policy generator is retained by reference and is created
    lazily only when a learned action is first requested.
    """

    def __init__(
        self,
        driver: Any,
        policy: Any,
        config: NeuralConfig,
        *,
        controller_rng: random.Random,
        curriculum_rng: random.Random,
        budget: Any,
        policy_generator: Any | None = None,
    ) -> None:
        if not isinstance(config, NeuralConfig):
            raise TypeError("Collector requires a NeuralConfig")
        mode = getattr(driver, "mode", None)
        batch_size = getattr(driver, "batch_size", None)
        if mode != config.mode or type(batch_size) is not int or batch_size != config.rollout_rows:
            raise ContractError("driver mode and batch width must match the neural config")
        rows = getattr(driver, "rows", None)
        if not isinstance(rows, tuple) or len(rows) != batch_size:
            raise ContractError("driver must expose one current row boundary per native slot")
        if any(item is not None and not isinstance(item, DriverRow) for item in rows):
            raise ContractError("driver rows must be owned DriverRow values or None")
        if not callable(getattr(driver, "reset", None)) or not callable(getattr(driver, "step", None)):
            raise ContractError("driver must expose reset and step operations")
        if not callable(getattr(driver, "close", None)):
            raise ContractError("driver must expose close")
        if not callable(getattr(budget, "remaining", None)) or not callable(
            getattr(budget, "elapsed", None)
        ):
            raise TypeError("budget must expose elapsed and remaining wall time")
        if not callable(getattr(controller_rng, "choice", None)):
            raise TypeError("controller_rng must expose choice")
        if not callable(getattr(curriculum_rng, "randrange", None)):
            raise TypeError("curriculum_rng must expose randrange")
        for name in ("act", "values"):
            if not callable(getattr(policy, name, None)):
                raise ContractError(f"policy must expose {name}()")

        cases = config.cases.for_mode(mode)
        if not cases:
            raise ContractError(f"neural case manifest has no {mode} cases")
        identity_fields = ("model_id", "features_id", "action_id", "objective_id")
        policy_identity = tuple(getattr(policy, name, None) for name in identity_fields)
        identities = tuple(
            identity for identity in config.accepted_identities
            if tuple(getattr(identity, name) for name in identity_fields) == policy_identity
            and mode in identity.accepted_raw_mode_mappings
        )
        if len(identities) != 1:
            raise ContractError("policy must match one configured identity for the selected mode")

        self.driver = driver
        self.policy = policy
        self.config = config
        self.controller_rng = controller_rng
        self.curriculum_rng = curriculum_rng
        self.budget = budget
        self.identity: NeuralIdentity = identities[0]
        self._cases: tuple[NeuralCase, ...] = tuple(cases)
        self._policy_generator = policy_generator
        self._mode = mode
        self._batch_size = batch_size
        self._rows: tuple[DriverRow | None, ...] = rows
        self._needs_reset = [row is None for row in rows]
        self._episodes = [0 if row is not None else -1 for row in rows]
        self._decision_ordinals = [0] * batch_size
        self._slot_cases: list[NeuralCase | None] = [None] * batch_size
        self._slot_seeds: list[int | None] = [None] * batch_size
        self._seed_ordinal = 0
        self._seed_allocations: list[SeedAllocation] = []
        self._resource_checkpoints: list[ResourceCheckpoint] = []
        self._open_task_decisions: dict[tuple[str, str], int] = {}
        self._run_started_visits: set[tuple[int, int, int]] = set()
        self._inactive_reset_boundary = False
        self._closed = False
        self._failed = False
        self._phase = "idle"
        self._pending_decisions = 0
        self._sampled_actor_this_batch = 0
        self._sampled_fixed_this_batch = 0
        self._elapsed_seconds_is_fallback = False
        self._clock_failure_pending = False
        self._current_operation_slots = 0
        self._current_operation_kind = "none"
        self._fault: CollectionFault | None = None
        self._stop_reason: str | None = None
        self._last_rollout: Rollout | None = None
        self._fault_incomplete_decisions = 0
        self._incomplete_environment_slots = 0
        self._counts = {
            "rows": 0,
            "actor_rows": 0,
            "fixed_rows": 0,
            "task_terminals": 0,
            "environment_terminations": 0,
            "environment_truncations": 0,
            "attempted_environment_slots": 0,
            "forwarded_environment_slots": 0,
            "known_accepted_environment_slots": 0,
            "delivered_environment_slots": 0,
            "uncertain_environment_slots": 0,
            "reset_operations": 0,
            "reset_slots": 0,
            "attempted_reset_slots": 0,
            "forwarded_reset_slots": 0,
            "known_accepted_reset_slots": 0,
            "delivered_reset_slots": 0,
            "uncertain_reset_slots": 0,
            "learned_decisions": 0,
            "fixed_decisions": 0,
            "tasks_started": 0,
            "tasks_completed": 0,
            "victories": 0,
            "deaths": 0,
            "faults": 0,
            "undispatched_actor_decisions": 0,
            "undispatched_fixed_decisions": 0,
        }
        initial_elapsed = float(budget.elapsed())
        initial_remaining = float(budget.remaining())
        if (not math.isfinite(initial_elapsed) or initial_elapsed < 0
                or not math.isfinite(initial_remaining) or initial_remaining < 0):
            raise ContractError("wall budget elapsed and remaining values must be finite and non-negative")
        self._last_elapsed_seconds = initial_elapsed
        self._budget_total = initial_elapsed + initial_remaining
        if not math.isfinite(self._budget_total) or self._budget_total <= 0:
            raise ContractError("wall budget must have a finite positive total duration")
        for slot, row in enumerate(rows):
            if row is not None and row.lifecycle.active:
                self._mark_task_started(slot, row.lifecycle.active_visit)

    @property
    def policy_generator(self) -> Any | None:
        """The retained sampling generator for explicit later state handling."""
        return self._policy_generator

    @property
    def partial_rollout(self) -> Rollout | None:
        """Most recent immutable output, including a known prefix after failure."""
        return self._last_rollout

    def _mark_task_started(self, slot: int, visit: int | None) -> None:
        if self._mode == "standalone":
            self._counts["tasks_started"] += 1
            return
        if visit is None:
            raise ContractError("active Run row has no public visit index")
        episode = self._episodes[slot]
        if episode < 0:
            raise ContractError("active Run row has no initialized Run episode")
        key = (slot, episode, visit)
        if key not in self._run_started_visits:
            self._run_started_visits.add(key)
            self._counts["tasks_started"] += 1

    def _stream_id(self, slot: int) -> str:
        return f"stream-{slot}"

    def _task_id(self, slot: int, source: DriverRow) -> str:
        if self._mode == "standalone":
            if self._episodes[slot] < 0:
                self._episodes[slot] = 0
            return f"task-{slot}-{self._episodes[slot]}"
        episode = self._episodes[slot]
        if episode < 0:
            raise ContractError("Run task cannot be identified before its reset episode")
        visit = source.lifecycle.active_visit
        if visit is None:
            visit = source.lifecycle.combats_entered
        return f"task-{slot}-run-{episode}-visit-{visit}"

    def _append_resource_checkpoint(
        self,
        stream_id: str,
        task_id: str,
        visit: int,
        exit_record,
        frame,
    ) -> None:
        self._resource_checkpoints.append(ResourceCheckpoint(
            stream_id=stream_id,
            task_id=task_id,
            visit=visit,
            outcome=exit_record["outcome"],
            frame=frame,
            encounter=exit_record["encounter"],
            hp_before_healing=exit_record["hp_before_healing"],
            hp_after_cleanup=exit_record["hp_after_cleanup"],
        ))

    def _wall_remaining(self) -> float:
        previous_phase = self._phase
        self._phase = "budget-check"
        try:
            remaining = float(self.budget.remaining())
            elapsed = float(self.budget.elapsed())
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ContractError("wall budget elapsed time must be finite and non-negative")
            self._last_elapsed_seconds = elapsed
            self._elapsed_seconds_is_fallback = False
            if not math.isfinite(remaining) or remaining < 0:
                raise ContractError("wall budget remaining time must be finite and non-negative")
            remaining = min(remaining, max(0.0, self._budget_total - elapsed))
            self._clock_failure_pending = False
        except BaseException:
            self._elapsed_seconds_is_fallback = True
            self._clock_failure_pending = True
            raise
        self._phase = previous_phase
        return remaining

    def _refresh_elapsed_best_effort(self, primary: BaseException) -> None:
        if self._clock_failure_pending:
            return
        try:
            elapsed = float(self.budget.elapsed())
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ContractError("wall budget elapsed time must be finite and non-negative")
        except BaseException as clock_error:
            self._elapsed_seconds_is_fallback = True
            self._clock_failure_pending = True
            add_note = getattr(primary, "add_note", None)
            if callable(add_note):
                add_note(
                    "elapsed-time snapshot used the last successfully observed value after "
                    f"{type(clock_error).__name__}: {_clean_error(clock_error)}"
                )
        else:
            self._last_elapsed_seconds = elapsed
            self._elapsed_seconds_is_fallback = False

    def _actual_reset_seed(self, case: NeuralCase, ordinal: int) -> tuple[int, str]:
        if self.config.split == "train":
            return environment_seed_for(self.config.split, replica=0, ordinal=ordinal), _TRAIN_SEED_POLICY
        return case.environment_seed, _REFERENCE_SEED_POLICY

    def _seed_domain_available(self, mask: tuple[bool, ...]) -> bool:
        if self.config.split != "train":
            return True
        ordinal = self._seed_ordinal
        for selected in mask:
            if not selected:
                continue
            try:
                environment_seed_for(self.config.split, replica=0, ordinal=ordinal)
            except ValueError:
                return False
            ordinal += 1
        return True

    def _choose_case(self, allocation_index: int) -> NeuralCase:
        if self.config.split == "train":
            return self._cases[self.curriculum_rng.randrange(len(self._cases))]
        return self._cases[allocation_index % len(self._cases)]

    def _prepare_reset(
        self,
        mask: tuple[bool, ...],
    ) -> tuple[tuple[int, ...], tuple[str | None, ...], tuple[SeedAllocation, ...]]:
        seeds: list[int] = []
        entries: list[str | None] = []
        allocations: list[SeedAllocation] = []
        next_ordinal = self._seed_ordinal
        for slot, selected in enumerate(mask):
            if selected:
                case = self._choose_case(next_ordinal)
                actual_seed, policy_id = self._actual_reset_seed(case, next_ordinal)
                allocation = SeedAllocation(
                    stream_id=self._stream_id(slot),
                    slot=slot,
                    case_id=case.case_id,
                    entry_id=case.entry_id,
                    authored_reference_seed=case.environment_seed,
                    actual_seed=actual_seed,
                    ordinal=next_ordinal,
                    allocation_policy_id=policy_id,
                    delivery="attempted",
                )
                allocations.append(allocation)
                seeds.append(actual_seed)
                entries.append(case.entry_id)
                next_ordinal += 1
            else:
                case = self._slot_cases[slot]
                seed = self._slot_seeds[slot]
                if case is None or seed is None:
                    raise ContractError("masked reset cannot preserve an uninitialized row")
                seeds.append(seed)
                entries.append(case.entry_id)
        self._seed_ordinal = next_ordinal
        self._seed_allocations.extend(allocations)
        return tuple(seeds), tuple(entries), tuple(allocations)

    @staticmethod
    def _reset_evidence(error: BaseException, slots: int) -> tuple[int, int]:
        evidence = getattr(error, "evidence", None)
        if isinstance(error, DriverFault) and evidence is not None:
            return (
                min(slots, max(0, int(evidence.known_current_delivery_slots))),
                min(slots, max(0, int(evidence.uncertain_current_slots))),
            )
        return 0, slots

    def _reset_selected(self, mask: tuple[bool, ...]) -> tuple[DriverRow, ...]:
        selected_slots = sum(mask)
        seeds, entries, allocations = self._prepare_reset(mask)
        self._counts["attempted_reset_slots"] += selected_slots
        self._phase = "reset"
        self._current_operation_kind = "reset"
        self._current_operation_slots = selected_slots
        try:
            self._counts["forwarded_reset_slots"] += selected_slots
            result = self.driver.reset(seeds=seeds, entry_ids=entries, reset_mask=mask)
        except BaseException as error:
            known, uncertain = self._reset_evidence(error, selected_slots)
            self._counts["known_accepted_reset_slots"] += known
            self._counts["uncertain_reset_slots"] += uncertain
            self._seed_allocations[-len(allocations):] = [
                replace(item, delivery=("uncertain" if uncertain else
                                        "accepted-undelivered" if known else "rejected"))
                for item in allocations
            ] if allocations else []
            raise
        self._phase = "post-reset"
        self._counts["known_accepted_reset_slots"] += selected_slots
        if not isinstance(result, tuple) or len(result) != self._batch_size:
            raise ContractError("driver reset must return a full immutable row tuple")
        if any(not isinstance(item, DriverRow) for item in result):
            raise ContractError("driver reset returned a non-DriverRow boundary")
        self._counts["reset_operations"] += 1
        self._counts["reset_slots"] += selected_slots
        self._counts["delivered_reset_slots"] += selected_slots
        self._seed_allocations[-len(allocations):] = [
            replace(item, delivery="delivered") for item in allocations
        ] if allocations else []
        for slot, selected in enumerate(mask):
            if not selected:
                if result[slot] != self._rows[slot]:
                    raise ContractError("masked reset changed an unselected public boundary")
                continue
            self._rows = tuple(result)
            self._needs_reset[slot] = False
            allocation = next(item for item in allocations if item.slot == slot)
            case = next(item for item in self._cases if item.case_id == allocation.case_id)
            self._slot_cases[slot] = case
            self._slot_seeds[slot] = allocation.actual_seed
            self._episodes[slot] += 1
            boundary = result[slot]
            if boundary.lifecycle.active:
                self._mark_task_started(slot, boundary.lifecycle.active_visit)
            elif self._mode == "standalone":
                exit_record = boundary.lifecycle.last_exit
                if exit_record is not None:
                    outcome = exit_record.get("outcome")
                    visit = exit_record.get("visit")
                    if outcome in {"victory", "death"} and type(visit) is int:
                        self._append_resource_checkpoint(
                            self._stream_id(slot),
                            self._task_id(slot, boundary),
                            visit,
                            exit_record,
                            boundary.frame,
                        )
                self._inactive_reset_boundary = True
        self._rows = tuple(result)
        self._phase = "ready"
        self._current_operation_kind = "none"
        self._current_operation_slots = 0
        return self._rows  # type: ignore[return-value]

    def _reserved_learned_rows(self, rows: tuple[DriverRow | None, ...]) -> int:
        total = 0
        for slot, row in enumerate(rows):
            if self._needs_reset[slot] or row is None:
                total += 1
            elif row.lifecycle.active:
                total += 1
        return total

    def _stop_for_capacity(self, action_limit: int, decision_limit: int) -> bool:
        if self._counts["attempted_environment_slots"] + self._batch_size > action_limit:
            self._stop_reason = "environment_action_budget"
            return True
        required = self._reserved_learned_rows(self._rows)
        if self._counts["learned_decisions"] + required > decision_limit:
            self._stop_reason = "learned_decision_budget"
            return True
        return False

    def _policy_generator_for_actor(self) -> Any:
        if self._policy_generator is None:
            import torch

            generator = torch.Generator(device=self.config.device)
            generator.manual_seed(self.config.seed_streams.policy)
            self._policy_generator = generator
        return self._policy_generator

    def _actions(self, rows: tuple[DriverRow, ...], policy_version: str | int):
        learned_slots = tuple(index for index, row in enumerate(rows) if row.lifecycle.active)
        self._pending_decisions = len(learned_slots)
        self._sampled_actor_this_batch = 0
        self._sampled_fixed_this_batch = 0
        actions: list[int | None] = [None] * self._batch_size
        choices: list[PolicyChoice | None] = [None] * self._batch_size
        if learned_slots:
            generator = self._policy_generator_for_actor()
            frames = tuple(rows[index].frame for index in learned_slots)
            policy_choices = self.policy.act(frames, generator=generator, deterministic=False)
            if not isinstance(policy_choices, tuple) or len(policy_choices) != len(learned_slots):
                raise ContractError("policy.act must return one immutable choice per learned row")
            self._sampled_actor_this_batch = len(learned_slots)
            for slot, choice in zip(learned_slots, policy_choices, strict=True):
                if not isinstance(choice, PolicyChoice):
                    raise ContractError("policy.act returned a non-PolicyChoice value")
                if (choice.value is None or choice.log_probability is None
                        or choice.behavior_policy_version != policy_version):
                    raise ContractError("learned choice statistics/version disagree with the requested policy")
                legal = tuple(candidate.index for candidate in rows[slot].candidates)
                if choice.original_action_index not in legal:
                    raise ContractError("policy selected an action outside the current legal candidates")
                actions[slot] = choice.original_action_index
                choices[slot] = choice
        for slot, current in enumerate(rows):
            if current.lifecycle.active:
                continue
            if self._mode != "run":
                raise ContractError("inactive standalone rows cannot be dispatched")
            if not current.noncombat_indices:
                raise ContractError("inactive Run row has no public legal noncombat actions")
            action = self.controller_rng.choice(current.noncombat_indices)
            if type(action) is not int or action not in current.noncombat_indices:
                raise ContractError("controller_rng selected an action outside the public legal set")
            actions[slot] = action
            choices[slot] = PolicyChoice(action)
            self._sampled_fixed_this_batch += 1
        if any(action is None for action in actions) or any(choice is None for choice in choices):
            raise ContractError("collector did not construct a complete native batch")
        return tuple(actions), tuple(choices), learned_slots

    @staticmethod
    def _validate_lifecycle_jump(previous, current, transition: DriverTransition) -> None:
        entered_delta = current.combats_entered - previous.combats_entered
        settled_delta = current.combats_settled - previous.combats_settled
        if entered_delta not in (0, 1) or settled_delta not in (0, 1):
            raise ContractError("one action cannot skip or settle multiple public visits")
        if previous.active and current.active:
            if entered_delta or settled_delta or previous.active_visit != current.active_visit:
                raise ContractError("active combat lifecycle jumped between visits")
        elif not previous.active and current.active:
            if entered_delta != 1 or current.active_visit != previous.combats_entered:
                raise ContractError("Run combat entry does not follow the public visit sequence")
            if settled_delta:
                raise ContractError("one action cannot both enter and settle a combat")
        elif previous.active and not current.active:
            if settled_delta != 1 and not (transition.terminated or transition.truncated):
                raise ContractError("active combat exited without one public settled visit")
            if entered_delta:
                raise ContractError("combat exit cannot also enter another visit")
        elif entered_delta or settled_delta:
            raise ContractError("inactive lifecycle changed visit counters without a combat boundary")

    def _step_batch(self, rows: tuple[DriverRow, ...], policy_version: str | int):
        actions, choices, learned_slots = self._actions(rows, policy_version)
        if self._wall_remaining() <= 0:
            self._counts["undispatched_actor_decisions"] += self._sampled_actor_this_batch
            self._counts["undispatched_fixed_decisions"] += self._sampled_fixed_this_batch
            self._sampled_actor_this_batch = 0
            self._sampled_fixed_this_batch = 0
            self._pending_decisions = 0
            self._phase = "ready"
            self._stop_reason = "wall_budget"
            return None
        self._counts["attempted_environment_slots"] += self._batch_size
        self._phase = "step"
        self._current_operation_kind = "step"
        self._current_operation_slots = self._batch_size
        self._sampled_actor_this_batch = 0
        self._sampled_fixed_this_batch = 0
        try:
            self._counts["forwarded_environment_slots"] += self._batch_size
            transitions = self.driver.step(actions)
        except BaseException as error:
            evidence = getattr(error, "evidence", None)
            if isinstance(error, DriverFault) and evidence is not None:
                known = min(self._batch_size, max(0, evidence.known_current_delivery_slots))
                uncertain = min(self._batch_size, max(0, evidence.uncertain_current_slots))
            else:
                known, uncertain = 0, self._batch_size
            self._counts["known_accepted_environment_slots"] += known
            self._counts["uncertain_environment_slots"] += uncertain
            self._incomplete_environment_slots += known + uncertain
            raise
        self._phase = "post-step"
        self._counts["known_accepted_environment_slots"] += self._batch_size
        self._counts["delivered_environment_slots"] += self._batch_size
        if not isinstance(transitions, tuple) or len(transitions) != self._batch_size:
            raise ContractError("driver.step must return one immutable transition per native slot")
        if any(not isinstance(item, DriverTransition) for item in transitions):
            raise ContractError("driver.step returned a non-DriverTransition value")

        prepared = []
        bootstrap_frames = []
        for slot, (source, result, choice) in enumerate(zip(rows, transitions, choices, strict=True)):
            if result.source != source:
                raise ContractError("driver transition source differs from its accepted input snapshot")
            self._validate_lifecycle_jump(source.lifecycle, result.successor.lifecycle, result)
            signal = task_signal(
                source.lifecycle,
                result.successor.lifecycle,
                terminated=result.terminated,
                truncated=result.truncated,
            )
            task_id = self._task_id(slot, source)
            decision_id = f"decision-{self._decision_ordinals[slot]}"
            learned = slot in learned_slots
            if learned and not signal.task_terminal:
                bootstrap_frames.append((slot, result.successor.frame))
            prepared.append((slot, result, choice, signal, task_id, decision_id, learned))

        next_values: dict[int, float] = {}
        if bootstrap_frames:
            value_frames = tuple(value for _, value in bootstrap_frames)
            values = self.policy.values(value_frames)
            if not isinstance(values, tuple) or len(values) != len(value_frames):
                raise ContractError("policy.values must return one value per actual successor frame")
            for (slot, _), value in zip(bootstrap_frames, values, strict=True):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ContractError("policy.values returned a non-finite successor value")
                next_values[slot] = float(value)

        records: list[StepRecord] = []
        for slot, result, choice, signal, task_id, decision_id, learned in prepared:
            next_value = 0.0 if learned and signal.task_terminal else next_values.get(slot)
            records.append(StepRecord(
                stream_id=self._stream_id(slot),
                task_id=task_id,
                decision_id=decision_id,
                source=result.source.frame,
                successor=result.successor.frame,
                native_reward=result.native_reward,
                terminated=result.terminated,
                truncated=result.truncated,
                task_signal=signal,
                choice=choice,
                actor_training_mask=learned,
                next_value=next_value,
            ))

        for record, (slot, result, _choice, signal, task_id, _decision_id, learned) in zip(
            records, prepared, strict=True
        ):
            self._decision_ordinals[slot] += 1
            self._counts["rows"] += 1
            self._counts["actor_rows"] += int(learned)
            self._counts["fixed_rows"] += int(not learned)
            self._counts["learned_decisions"] += int(learned)
            self._counts["fixed_decisions"] += int(not learned)
            self._counts["task_terminals"] += int(signal.task_terminal)
            self._counts["environment_terminations"] += int(result.terminated)
            self._counts["environment_truncations"] += int(result.truncated)
            task_key = (record.stream_id, task_id)
            if signal.task_terminal:
                self._open_task_decisions.pop(task_key, None)
            else:
                self._open_task_decisions[task_key] = self._open_task_decisions.get(task_key, 0) + 1

            previous = result.source.lifecycle
            current = result.successor.lifecycle
            entered_delta = current.combats_entered - previous.combats_entered
            settled_delta = current.combats_settled - previous.combats_settled
            if self._mode == "run" and entered_delta == 1:
                self._mark_task_started(slot, current.active_visit)
            if settled_delta == 1:
                exit_record = current.last_exit
                if exit_record is None:
                    raise ContractError("settled Run visit lacks its public exit checkpoint")
                outcome = exit_record.get("outcome")
                visit = exit_record.get("visit")
                if outcome not in {"victory", "death"} or type(visit) is not int:
                    raise ContractError("settled Run visit has an unsupported outcome")
                self._counts["tasks_completed"] += 1
                self._counts["victories"] += int(outcome == "victory")
                self._counts["deaths"] += int(outcome == "death")
                self._append_resource_checkpoint(
                    record.stream_id, task_id, visit, exit_record, result.successor.frame
                )
            if self._mode == "standalone" and (signal.task_terminal or result.terminated):
                self._needs_reset[slot] = True
            elif self._mode == "run" and result.terminated:
                self._needs_reset[slot] = True

        self._rows = tuple(item.successor for item in transitions)
        self._phase = "ready"
        self._pending_decisions = 0
        self._current_operation_kind = "none"
        self._current_operation_slots = 0
        return tuple(records)

    def _fault_from_exception(self, error: BaseException) -> CollectionFault:
        evidence = getattr(error, "evidence", None)
        operation_slots = 0
        known = 0
        uncertain = 0
        phase = self._phase
        if isinstance(error, DriverFault) and evidence is not None:
            phase = evidence.phase
            operation_slots = min(self._batch_size, max(0, evidence.operation_slots))
            known = min(operation_slots, max(0, evidence.known_current_delivery_slots))
            uncertain = min(operation_slots, max(0, evidence.uncertain_current_slots))
        elif self._current_operation_slots:
            operation_slots = self._current_operation_slots
            if phase == "post-step":
                known = operation_slots
            elif phase == "post-reset":
                known = operation_slots
            elif self._current_operation_kind in {"step", "reset"}:
                uncertain = operation_slots
        return CollectionFault(
            phase=phase if phase and phase != "idle" else "collector",
            exception_type=type(error).__name__,
            message=_clean_error(error),
            operation_slots=operation_slots,
            known_accepted_slots=known,
            uncertain_slots=uncertain,
        )

    def _diagnostics(self, *, stop_reason: str | None = None) -> CollectionDiagnostics:
        elapsed = self._last_elapsed_seconds
        overshoot = max(0.0, elapsed - self._budget_total)
        incomplete = tuple(
            IncompleteTask(stream, task, decisions)
            for (stream, task), decisions in sorted(self._open_task_decisions.items())
            if decisions > 0
        )
        incomplete_decisions = sum(item.decisions for item in incomplete) + self._fault_incomplete_decisions
        return CollectionDiagnostics(
            attempted_environment_slots=self._counts["attempted_environment_slots"],
            forwarded_environment_slots=self._counts["forwarded_environment_slots"],
            known_accepted_environment_slots=self._counts["known_accepted_environment_slots"],
            delivered_environment_slots=self._counts["delivered_environment_slots"],
            uncertain_environment_slots=self._counts["uncertain_environment_slots"],
            reset_operations=self._counts["reset_operations"],
            reset_slots=self._counts["reset_slots"],
            attempted_reset_slots=self._counts["attempted_reset_slots"],
            forwarded_reset_slots=self._counts["forwarded_reset_slots"],
            known_accepted_reset_slots=self._counts["known_accepted_reset_slots"],
            delivered_reset_slots=self._counts["delivered_reset_slots"],
            uncertain_reset_slots=self._counts["uncertain_reset_slots"],
            learned_decisions=self._counts["learned_decisions"],
            fixed_decisions=self._counts["fixed_decisions"],
            tasks_started=self._counts["tasks_started"],
            task_terminals=self._counts["task_terminals"],
            tasks_completed=self._counts["tasks_completed"],
            victories=self._counts["victories"],
            deaths=self._counts["deaths"],
            faults=self._counts["faults"],
            elapsed_seconds=elapsed,
            wall_overshoot_seconds=overshoot,
            incomplete_decisions=incomplete_decisions,
            incomplete_environment_slots=self._incomplete_environment_slots,
            undispatched_actor_decisions=self._counts["undispatched_actor_decisions"],
            undispatched_fixed_decisions=self._counts["undispatched_fixed_decisions"],
            elapsed_seconds_is_fallback=self._elapsed_seconds_is_fallback,
            next_seed_ordinal=self._seed_ordinal,
            seed_allocations=tuple(self._seed_allocations),
            resource_checkpoints=tuple(self._resource_checkpoints),
            incomplete_tasks=incomplete,
            stop_reason=stop_reason if stop_reason is not None else self._stop_reason,
            fault=self._fault,
        )

    def _make_rollout(self, records: tuple[StepRecord, ...], *, stop_reason=None) -> Rollout:
        counters = CumulativeCounters(
            self._counts["rows"],
            self._counts["actor_rows"],
            self._counts["fixed_rows"],
            self._counts["task_terminals"],
            self._counts["environment_terminations"],
            self._counts["environment_truncations"],
        )
        return Rollout(
            mode=self._mode,
            split=self.config.split,
            identity=self.identity,
            records=records,
            counters=counters,
            collection=self._diagnostics(stop_reason=stop_reason),
        )

    def _close_after_fault(self, primary: BaseException) -> None:
        try:
            self.close()
        except BaseException as close_error:
            add_note = getattr(primary, "add_note", None)
            if callable(add_note):
                add_note(
                    "collector close after primary failure also raised "
                    f"{type(close_error).__name__}: {_clean_error(close_error)}"
                )

    def collect(
        self,
        *,
        max_learned_decisions: int,
        max_env_actions: int,
        policy_version: str | int,
    ) -> Rollout:
        """Collect at most one configured horizon under cumulative ceilings."""
        for name, value in (("max_learned_decisions", max_learned_decisions),
                            ("max_env_actions", max_env_actions)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative built-in integer")
        if not _valid_policy_version(policy_version):
            raise ValueError("policy_version must be a non-negative integer or identifier")
        if self._failed:
            raise RuntimeError("collector is faulted; collection cannot be retried")
        if self._closed:
            raise RuntimeError("collector is closed")

        action_limit = min(max_env_actions, self.config.budgets.max_environment_actions)
        decision_limit = max_learned_decisions
        if self.config.operation == "train":
            decision_limit = min(decision_limit, self.config.budgets.max_training_decisions)
        call_records: list[StepRecord] = []
        self._stop_reason = None
        try:
            for _ in range(self.config.rollout_steps):
                if self._wall_remaining() <= 0:
                    self._stop_reason = "wall_budget"
                    break
                if self._inactive_reset_boundary:
                    self._stop_reason = "inactive_reset_boundary"
                    break
                if self._stop_for_capacity(action_limit, decision_limit):
                    break

                reset_mask = tuple(
                    self._needs_reset[slot] or self._rows[slot] is None
                    for slot in range(self._batch_size)
                )
                if any(reset_mask):
                    if not self._seed_domain_available(reset_mask):
                        self._stop_reason = "seed_domain_exhausted"
                        break
                    self._reset_selected(reset_mask)
                    if self._inactive_reset_boundary:
                        self._stop_reason = "inactive_reset_boundary"
                        break
                    if self._wall_remaining() <= 0:
                        self._stop_reason = "wall_budget"
                        break

                current_rows = self._rows
                if any(row is None for row in current_rows):
                    raise ContractError("collector reset left an uninitialized native row")
                rows = tuple(row for row in current_rows if row is not None)
                if self._mode == "standalone" and any(not row.lifecycle.active for row in rows):
                    self._stop_reason = "inactive_reset_boundary"
                    self._inactive_reset_boundary = True
                    for slot, current in enumerate(rows):
                        if not current.lifecycle.active and current.lifecycle.last_exit is not None:
                            exit_record = current.lifecycle.last_exit
                            outcome, visit = exit_record.get("outcome"), exit_record.get("visit")
                            if outcome in {"victory", "death"} and type(visit) is int:
                                task_id = self._task_id(slot, current)
                                self._append_resource_checkpoint(
                                    self._stream_id(slot), task_id, visit,
                                    exit_record, current.frame,
                                )
                    break
                if self._wall_remaining() <= 0:
                    self._stop_reason = "wall_budget"
                    break
                batch_records = self._step_batch(rows, policy_version)
                if batch_records is None:
                    break
                call_records.extend(batch_records)
                if self._wall_remaining() <= 0:
                    self._stop_reason = "wall_budget"
                    break
                if self._mode == "run" and any(record.terminated for record in batch_records):
                    self._stop_reason = "run_episode_terminal"
                    break
            if self._stop_reason is None:
                self._stop_reason = "horizon"
            self._last_rollout = self._make_rollout(tuple(call_records))
            return self._last_rollout
        except BaseException as error:
            self._refresh_elapsed_best_effort(error)
            self._counts["faults"] += 1
            self._fault = self._fault_from_exception(error)
            self._failed = True
            if self._current_operation_kind != "step":
                self._counts["undispatched_actor_decisions"] += self._sampled_actor_this_batch
                self._counts["undispatched_fixed_decisions"] += self._sampled_fixed_this_batch
            self._sampled_actor_this_batch = 0
            self._sampled_fixed_this_batch = 0
            self._fault_incomplete_decisions += self._pending_decisions
            if self._phase == "post-step":
                self._incomplete_environment_slots += self._batch_size
            self._pending_decisions = 0
            self._stop_reason = "fault"
            self._close_after_fault(error)
            try:
                self._last_rollout = self._make_rollout(tuple(call_records))
            except BaseException as snapshot_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "partial rollout snapshot also failed: "
                        f"{type(snapshot_error).__name__}: {_clean_error(snapshot_error)}"
                    )
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    f"collector retained {len(call_records)} complete records; "
                    f"phase={self._fault.phase}; incomplete_decisions="
                    f"{self._fault_incomplete_decisions}; collection cannot be retried"
                )
            raise

    def close(self) -> None:
        """Close the owned driver exactly once."""
        if self._closed:
            return
        self._closed = True
        self.driver.close()


__all__ = ["Collector"]
