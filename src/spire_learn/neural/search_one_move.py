"""Batched one-move public search with common random numbers."""

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
import random

from . import search_view
from .search_view import View


@dataclass(frozen=True)
class ActionStat:
    mean: float
    visits: int
    refused: int


@dataclass(frozen=True)
class StepOutcome:
    observation: Mapping | None
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping
    refused: bool = False


def step_branches(search, branches, actions, *, view: str) -> list[StepOutcome]:
    """Step public branches in request order with the arena's configured worker count.

    Gym v1.2.0 sets workers when opening the arena. Branch and arena
    ownership stay with the caller.
    """
    unsupported, _, fault_error = search_view._native_errors()
    try:
        observations, rewards, terminated, truncated, info = search.step_many(branches, actions, view=view)
    except unsupported:
        # A whole-call preflight refusal leaves all requests untouched and
        # gives no index. Scalar stepping attributes the refusal to its branch.
        outcomes = []
        for branch, action in zip(branches, actions, strict=True):
            try:
                row, reward, done, cut, detail = branch.step(action, view=view)
                if view == "mask":
                    row = {"action_mask": row}
                outcomes.append(StepOutcome(row, float(reward), bool(done), bool(cut), detail))
            except unsupported:
                outcomes.append(StepOutcome(None, -1.0, False, False, {}, refused=True))
        return outcomes

    # Gym returns zero-filled observations on faulted rows.
    for index, detail in enumerate(info["fault"]):
        if detail is not None:
            error = fault_error(f"public search branch {index}: {detail}")
            error.detail = detail
            raise error

    fields = [(key, values, info["_" + key]) for key, values in info.items()
              if key != "fault" and not key.startswith("_")]
    return [
        StepOutcome(
            {key: values[i] for key, values in observations.items()},
            float(rewards[i]), bool(terminated[i]), bool(truncated[i]),
            {key: values[i] for key, values, present in fields if present[i]},
        )
        for i in range(len(branches))
    ]


@dataclass
class _Playout:
    branch: object
    action: int
    rng: random.Random
    value: float = 0.0
    refused: bool = False


def _close_branch(branch) -> None:
    # Preserve the search result or the original fault if cleanup fails.
    with suppress(Exception):
        branch.close()


def one_move_values(search, legal: Sequence[int], view: View, *, samples: int,
                    playout_steps: int, research_seed: int,
                    branch_view: str = "full") -> dict[int, ActionStat]:
    """Evaluate each action over paired worlds. The caller owns the search root."""
    with ExitStack() as handles:
        by_action = {}
        active = []
        for action in legal:
            group = []
            for sample in range(samples):
                branch = search.branch(sample)
                handles.callback(_close_branch, branch)
                playout = _Playout(branch, action, random.Random(research_seed * 1_000_003 + sample))
                group.append(playout)
                active.append(playout)
            by_action[action] = group

        for depth in range(playout_steps + 1):
            if not active:
                break
            branches = [item.branch for item in active]
            actions = [item.action for item in active]
            continuing = []
            for item, result in zip(active, step_branches(search, branches, actions, view=branch_view), strict=True):
                if result.refused or result.terminated:
                    item.value = result.reward
                    item.refused = result.refused
                elif result.truncated or depth == playout_steps:
                    item.value = 0.5 * view.balance(result.observation)
                else:
                    item.action = view.heuristic_choice(result.observation, item.rng)
                    continuing.append(item)
            active = continuing
            # Row views retain the stacked arrays. Release them before the next dispatch.
            result = None

        values = {}
        for action, group in by_action.items():
            # Accumulate in sample order with the original float additions.
            total = 0.0
            for item in group:
                total += item.value
            values[action] = ActionStat(total / samples, samples, sum(item.refused for item in group))
        return values
