"""Pure-core collection for one fixed case in a single environment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import re

from ..agents.base import Policy, resolve_decision
from ..collection.boundaries import route_single_step
from ..contracts import (
    CaseExecution,
    CaseResult,
    Candidate,
    ContractError,
    DecisionRequest,
    Failure,
    PublicDecision,
    Route,
    Transition,
)
from ..envs.native import (
    NATIVE_CONTRACT,
    native_failure,
    native_snapshot,
    normalize_public_info,
    public_build_info,
)
from ..experiments.budget import Budget
from ..schema import EnvironmentContract, ObservationSchema, candidates_for, parse_descriptor


_ACTION_MAPPING_ID = "spire-learn-native-current-index-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_SPLITS = frozenset({"train", "selection", "final"})


def run_case(
    case: CaseExecution,
    policy: Policy,
    *,
    env_factory: Callable[[CaseExecution], object],
    budget: Budget,
    max_decisions: int,
    on_transition: Callable[[Transition], None] | None = None,
    classify_failure: Callable[[BaseException], Failure | None] = native_failure,
    contract: EnvironmentContract = NATIVE_CONTRACT,
) -> CaseResult:
    """Run one declared case and return its delivered public result."""
    if not isinstance(case, CaseExecution):
        raise TypeError("case must be a CaseExecution")

    status = "setup_error"
    started = False
    outcome = None
    steps = 0
    reward_sum = 0.0
    stop_reason = "setup_error"
    resources: dict[str, int | None] = {}
    primary_failure: Failure | None = None
    secondary_errors: list[Failure] = []
    observed_descriptor: Mapping[str, object] | None = None
    build_info: Mapping[str, object] | None = None
    env: object | None = None
    close_env: Callable[[], object] | None = None
    process_control: BaseException | None = None
    phase = "preflight"

    try:
        _validate_inputs(case, budget, max_decisions, env_factory, on_transition, classify_failure, contract)
        if budget.exhausted():
            status, stop_reason = "not_started", "wall_time_budget"
        else:
            phase = "factory"
            env = env_factory(case)
            if env is None:
                raise ContractError("environment factory returned None")
            close_candidate = getattr(env, "close", None)
            if callable(close_candidate):
                close_env = close_candidate

            for method in ("observation_spec", "reset", "step"):
                if not callable(getattr(env, method, None)):
                    raise ContractError(f"environment is missing callable {method}()")
            if close_env is None:
                raise ContractError("environment is missing callable close()")

            phase = "descriptor"
            descriptor = env.observation_spec()
            schema = parse_descriptor(descriptor, allowed_schema_ids=contract.schema_ids)
            if schema.content_id not in contract.content_ids:
                raise ContractError(f"unsupported content_id: {schema.content_id}")
            if case.expected_schema_id is not None and schema.schema_id != case.expected_schema_id:
                raise ContractError("observed schema_id does not match the case expectation")
            if case.expected_fingerprint is not None and schema.fingerprint != case.expected_fingerprint:
                raise ContractError("observed fingerprint does not match the case expectation")
            observed_descriptor = _observed_descriptor(schema)

            phase = "build_info"
            build_info = public_build_info()

            if budget.exhausted():
                status, stop_reason = "not_started", "wall_time_budget"
            else:
                phase = "reset"
                reset_result = env.reset(
                    seed=case.environment_seed,
                    options={"scenario_id": case.scenario_id},
                )
                started = True
                if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                    raise ContractError("reset() must return an observation and info pair")
                reset_observation, reset_info = reset_result
                current = native_snapshot(_public_mapping(reset_observation, "reset observation"), schema)
                public_info = normalize_public_info(_public_mapping(reset_info, "reset info"), schema)
                resources = _resources(public_info)
                status, stop_reason = "running", ""

                phase = "policy_init"
                policy_state = policy.init_state()
                decision_index = 0

                while status == "running":
                    phase = "budget"
                    if budget.exhausted():
                        status, stop_reason = "budget_stopped", "wall_time_budget"
                        break
                    if steps >= max_decisions:
                        status, stop_reason = "budget_stopped", "decision_cap"
                        break

                    phase = "decision_input"
                    mask, candidates = _decision_parts(current, schema)
                    route = Route(
                        case.execution_id,
                        0,
                        case.execution_id,
                        f"{case.execution_id}:{decision_index}",
                        decision_index == 0,
                    )
                    request = DecisionRequest(
                        PublicDecision(current, candidates, mask, schema.fingerprint, _ACTION_MAPPING_ID),
                        route,
                    )

                    phase = "policy"
                    decision = resolve_decision(request, policy, policy_state)
                    action = decision.choice.action
                    next_policy_state = decision.choice.next_state

                    phase = "budget"
                    if budget.exhausted():
                        status, stop_reason = "budget_stopped", "wall_time_budget"
                        break

                    phase = "step"
                    step_result = env.step(action)
                    steps += 1
                    if not isinstance(step_result, tuple) or len(step_result) != 5:
                        raise ContractError("step() must return observation, reward, flags, and info")
                    next_raw, reward, terminated, truncated, info = step_result
                    if type(reward) not in (int, float):
                        raise ContractError("step reward must be a finite primitive number")
                    try:
                        reward_finite = math.isfinite(reward)
                    except OverflowError as error:
                        raise ContractError("step reward is outside the finite numeric range") from error
                    if not reward_finite:
                        raise ContractError("step reward must be a finite primitive number")
                    try:
                        reward = float(reward)
                    except OverflowError as error:
                        raise ContractError("step reward is outside the finite numeric range") from error
                    if type(terminated) is not bool or type(truncated) is not bool:
                        raise ContractError("step ending flags must be bool values")
                    next_reward_sum = reward_sum + reward
                    reward_sum_overflow = not math.isfinite(next_reward_sum)
                    if not reward_sum_overflow:
                        reward_sum = next_reward_sum

                    phase = "successor"
                    successor = native_snapshot(_public_mapping(next_raw, "step observation"), schema)
                    public_info = normalize_public_info(_public_mapping(info, "step info"), schema)
                    resources = _resources(public_info)
                    outcome = _terminal_outcome(public_info, terminated)
                    ended = terminated or truncated
                    if terminated:
                        status, stop_reason = "terminal", "terminated"
                    elif truncated:
                        status, stop_reason = "env_truncated", "truncated"

                    transition = route_single_step(
                        route=route,
                        source=current,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        successor=successor,
                        state_after_source=next_policy_state,
                    ).transition
                    policy_state = next_policy_state
                    decision_index += 1

                    if on_transition is not None:
                        phase = "trace"
                        try:
                            on_transition(transition)
                        except Exception as error:
                            if ended:
                                secondary_errors.append(_plain_failure(error, "trace"))
                            else:
                                if reward_sum_overflow:
                                    secondary_errors.append(
                                        Failure(
                                            "contract_error",
                                            "reward sum is outside the finite numeric range",
                                            "reward_sum",
                                            "ContractError",
                                        )
                                    )
                                raise

                    if reward_sum_overflow:
                        phase = "reward_sum"
                        raise ContractError("reward sum is outside the finite numeric range")

                    if ended:
                        break
                    current = successor
    except BaseException as error:
        if not isinstance(error, Exception):
            process_control = error
            error.add_note(
                f"process-control exception during {phase}; status={status}, outcome={outcome}, steps={steps}"
            )
        else:
            primary_failure = _failure_for(error, phase, classify_failure, secondary_errors)
            if not started:
                status = "setup_error"
            elif phase in ("policy", "policy_init"):
                status = "agent_error"
            elif primary_failure.kind == "simulator_fault":
                status = "simulator_fault"
            elif isinstance(error, ContractError):
                status = "contract_error"
            else:
                status = "exec_error"
            stop_reason = status
    finally:
        if close_env is not None:
            try:
                close_env()
            except BaseException as close_error:
                if not isinstance(close_error, Exception):
                    if process_control is None:
                        process_control = close_error
                    else:
                        process_control.add_note(
                            f"environment close also raised {type(close_error).__name__}: {close_error}"
                        )
                else:
                    failure = _failure_for(close_error, "close", classify_failure, secondary_errors)
                    if process_control is not None:
                        process_control.add_note(
                            f"environment close also failed: {type(close_error).__name__}: {close_error}"
                        )
                        secondary_errors.append(failure)
                    elif primary_failure is not None or status in ("terminal", "env_truncated"):
                        secondary_errors.append(failure)
                    else:
                        primary_failure = failure
                        status = "setup_error" if not started else "exec_error"
                        stop_reason = status

    if process_control is not None:
        _add_process_control_context(
            process_control, primary_failure, secondary_errors, status, outcome, steps
        )
        raise process_control

    return CaseResult(
        case,
        status,
        started,
        outcome,
        steps,
        reward_sum,
        stop_reason,
        resources,
        primary_failure=primary_failure,
        secondary_errors=tuple(secondary_errors),
        observed_descriptor=observed_descriptor,
        native_build_info=build_info,
    )


def _validate_inputs(
    case: CaseExecution,
    budget: Budget,
    max_decisions: int,
    env_factory: Callable[[CaseExecution], object],
    on_transition: Callable[[Transition], None] | None,
    classify_failure: Callable[[BaseException], Failure | None],
    contract: EnvironmentContract,
) -> None:
    if not isinstance(contract, EnvironmentContract):
        raise ContractError("contract must be an EnvironmentContract")
    if (
        type(contract.schema_ids) is not frozenset
        or type(contract.content_ids) is not frozenset
        or not contract.schema_ids
        or not contract.content_ids
    ):
        raise ContractError("environment contract must allow schema and content IDs")
    if any(not isinstance(value, str) or not value for value in (*contract.schema_ids, *contract.content_ids)):
        raise ContractError("environment contract IDs must be non-empty strings")
    if not isinstance(contract.reward_contract_id, str) or not contract.reward_contract_id:
        raise ContractError("environment contract must name its reward contract")
    if not isinstance(contract.reward_evidence, str) or not contract.reward_evidence:
        raise ContractError("environment contract must describe its reward evidence")
    for label, value in (
        ("execution_id", case.execution_id),
        ("case_id", case.case_id),
        ("partition", case.partition),
        ("scenario_id", case.scenario_id),
    ):
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise ContractError(f"{label} must be a non-empty identifier")
    if not isinstance(case.split, str) or case.split not in _SPLITS:
        raise ContractError("split must be train, selection, or final")
    if any(type(seed) is not int or seed < 0 for seed in (case.environment_seed, case.policy_seed)):
        raise ContractError("case seeds must be non-negative integers")
    if case.expected_schema_id is not None:
        if not isinstance(case.expected_schema_id, str) or case.expected_schema_id not in contract.schema_ids:
            raise ContractError("expected_schema_id is outside the environment contract")
    if case.expected_fingerprint is not None and (
        not isinstance(case.expected_fingerprint, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", case.expected_fingerprint) is None
    ):
        raise ContractError("expected_fingerprint must be a 64-character hexadecimal value")
    if not isinstance(budget, Budget):
        raise ContractError("budget must be a Budget")
    if type(max_decisions) is not int or max_decisions <= 0:
        raise ContractError("max_decisions must be a positive integer")
    if not callable(env_factory):
        raise ContractError("env_factory must be callable")
    if on_transition is not None and not callable(on_transition):
        raise ContractError("on_transition must be callable or None")
    if not callable(classify_failure):
        raise ContractError("classify_failure must be callable")


def _public_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return value


def _decision_parts(
    observation, schema: ObservationSchema
) -> tuple[tuple[bool, ...], tuple[Candidate, ...]]:
    raw_mask = observation["action_mask"].values
    if any(type(value) is not int or value not in (0, 1) for value in raw_mask):
        raise ContractError("action mask values must be integer zero or one")
    mask = tuple(value == 1 for value in raw_mask)
    candidates = candidates_for(observation, schema)
    if not any(mask):
        raise ContractError("nonterminal observation has no legal actions")
    return mask, candidates


def _failure_for(
    error: Exception,
    phase: str,
    classify_failure: Callable[[BaseException], Failure | None],
    secondary_errors: list[Failure],
) -> Failure:
    if phase in ("policy", "policy_init"):
        return _plain_failure(error, phase, "agent_error")
    if phase == "trace":
        return _plain_failure(error, phase)
    if isinstance(error, ContractError):
        return _plain_failure(error, phase, "contract_error")
    try:
        classified = classify_failure(error)
    except Exception as classification_error:
        secondary_errors.append(_plain_failure(classification_error, "failure_classification"))
        classified = None
    if isinstance(classified, Failure):
        return Failure(
            classified.kind,
            classified.reason,
            phase,
            classified.exception_type or type(error).__name__,
        )
    if phase in ("preflight", "factory", "descriptor", "build_info", "reset"):
        kind = "setup_error"
    else:
        kind = "exec_error"
    return _plain_failure(error, phase, kind)


def _add_process_control_context(
    error: BaseException,
    primary: Failure | None,
    secondary: list[Failure],
    status: str,
    outcome: str | None,
    steps: int,
) -> None:
    if primary is not None:
        error.add_note(f"prior primary failure: {primary.kind} during {primary.phase}: {primary.reason}")
    for failure in secondary:
        error.add_note(f"prior secondary failure: {failure.kind} during {failure.phase}: {failure.reason}")
    error.add_note(f"case context before propagation: status={status}, outcome={outcome}, steps={steps}")


def _plain_failure(error: Exception, phase: str, kind: str = "exec_error") -> Failure:
    try:
        reason = str(error)
    except Exception:
        reason = "exception message unavailable"
    return Failure(kind, reason, phase, type(error).__name__)


def _terminal_outcome(info: Mapping[str, object], terminated: bool) -> str | None:
    if not terminated:
        return None
    outcome = info.get("outcome")
    return outcome if outcome in ("victory", "death") else None


def _resources(info: Mapping[str, object]) -> dict[str, int | None]:
    resources = dict(info.get("final_resources", {}))
    for key in ("hp_before_healing", "hp_after_cleanup"):
        if key in info:
            resources[key] = info[key]
    return resources


def _observed_descriptor(schema: ObservationSchema) -> Mapping[str, object]:
    return {
        "schema_id": schema.schema_id,
        "content_id": schema.content_id,
        "fingerprint": schema.fingerprint,
        "action_space": {"type": "Discrete", "n": schema.action_count, "start": 0},
        "tensors": tuple(
            {
                "key": tensor.key,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "fields": tensor.fields,
                "axes": tensor.axes,
                "category": tensor.category,
                "mask": tensor.mask,
            }
            for tensor in schema.tensors
        ),
        "vocabularies": schema.vocabularies,
    }
