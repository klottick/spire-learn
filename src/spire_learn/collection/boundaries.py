"""Route delivered step data across episode boundaries without touching an environment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..contracts import ContractError, Observation, Route, Transition


@dataclass(frozen=True)
class BoundaryResult:
    transition: Transition
    active_observation: Observation
    active_state: object | None
    episode_started: bool


def route_single_step(
    *, route: Route, source: Observation, action: int, reward: float,
    terminated: bool, truncated: bool, successor: Observation,
    state_after_source: object | None,
) -> BoundaryResult:
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ContractError("terminated and truncated flags must be bool values")
    ended = terminated or truncated
    transition = Transition(
        route=route, source=source, action=action, reward=reward,
        terminated=terminated, truncated=truncated, successor=successor,
        replacement=None, state_after_source=state_after_source,
    )
    return BoundaryResult(transition, transition.successor, None if ended else state_after_source, False)


def route_vector_step(
    *, routes: Sequence[Route], sources: Sequence[Observation],
    actions: Sequence[int], rewards: Sequence[float],
    terminated: Sequence[bool], truncated: Sequence[bool],
    main_observations: Sequence[Observation],
    final_observations: Sequence[Observation | None],
    final_presence: Sequence[bool], final_info_presence: Sequence[bool],
    states_after_source: Sequence[object | None],
    reset_state: Callable[[int], object | None],
) -> tuple[BoundaryResult, ...]:
    rows = (
        routes, sources, actions, rewards, terminated, truncated,
        main_observations, final_observations, final_presence,
        final_info_presence, states_after_source,
    )
    try:
        width = len(routes)
        if any(len(row) != width for row in rows):
            raise ContractError("all vector boundary rows must have equal width")
    except TypeError as error:
        raise ContractError("vector boundary inputs must be sized sequences") from error
    if not callable(reset_state):
        raise ContractError("reset_state must be callable")

    prepared: list[tuple[Transition, Observation, bool, int]] = []
    for index in range(width):
        term, trunc = terminated[index], truncated[index]
        present, info_present = final_presence[index], final_info_presence[index]
        final = final_observations[index]
        if type(term) is not bool or type(trunc) is not bool:
            raise ContractError("terminated and truncated flags must be bool values")
        if type(present) is not bool or type(info_present) is not bool:
            raise ContractError("final presence masks must contain bool values")
        ended = term or trunc
        if ended:
            if not present or not info_present or final is None:
                raise ContractError("ended rows require final observation and final info")
            successor, replacement = final, main_observations[index]
        else:
            if present or info_present or final is not None:
                raise ContractError("live rows cannot contain final boundary data")
            successor, replacement = main_observations[index], None
        if type(actions[index]) is not int:
            raise ContractError("actions must be integer values")
        transition = Transition(
            route=routes[index], source=sources[index], action=actions[index],
            reward=rewards[index], terminated=term, truncated=trunc,
            successor=successor, replacement=replacement,
            state_after_source=states_after_source[index],
        )
        active_observation = replacement if ended else successor
        prepared.append((transition, active_observation, ended, index))

    results = []
    for transition, active_observation, ended, index in prepared:
        state = reset_state(index) if ended else states_after_source[index]
        results.append(BoundaryResult(transition, active_observation, state, ended))
    return tuple(results)
