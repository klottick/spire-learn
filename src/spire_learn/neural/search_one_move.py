"""Batched one-move public search with common random numbers."""

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from hashlib import sha256
import random

from . import search_view
from .search_value import RootContext, ValueRule, cutoff_value, terminal_value
from .search_view import PLAYOUT_POLICIES, View, summary_balance

PLAYOUT_BACKENDS = ("python", "native")
# Gym's declared public rule: Learn's View.heuristic_choice excludes these Standard verbs.
PLAYOUT_RULE_ID = "spire-gym-public-playout-v1"
PLAYOUT_EXCLUDE_VERBS = ("end_turn", "discard_potion")
_FLAT = ValueRule("flat-v1")


def check_value_options(value_rule: ValueRule, playout_policy: str, playout_backend: str,
                        branch_view: str = "full") -> None:
    """Guards shared by both teachers for the value rule and playout policy."""
    if not isinstance(value_rule, ValueRule):
        raise ValueError("value_rule must be a ValueRule")
    if playout_policy not in PLAYOUT_POLICIES:
        raise ValueError(f"playout_policy must be one of {PLAYOUT_POLICIES}")
    if playout_backend != "python" and (value_rule.name != "flat-v1" or playout_policy != "uniform-v1"):
        raise ValueError(f"The native backend plays out under Gym's {PLAYOUT_RULE_ID} rule, which cannot "
                         "keep potions out; use playout_backend 'python' for "
                         f"{value_rule.name} with {playout_policy}")
    if playout_policy != "uniform-v1" and branch_view != "full":
        raise ValueError(f"playout_policy {playout_policy} requires the full branch view")
    if value_rule.name == "resource-v2" and playout_policy != "no-potions-v2":
        raise ValueError("value rule resource-v2 requires playout_policy no-potions-v2")


def terminal_score(rule: ValueRule, root: RootContext, view: View, observation, reward, info) -> float:
    """resource-v2 value of a terminal step; the reason comes from the step's info."""
    reason = info.get("terminal_reason")
    if reason is None:
        raise ValueError("A terminal step under resource-v2 has no terminal_reason")
    return terminal_value(rule, root, view, observation, reward, reason)


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


def playout_seed(research_seed: int, world: int, n: int) -> int:
    """Native playout seed for the n-th playout in a world.

    The arena's sample key already binds research_seed and world, so the first
    visit uses the reference stream (seed 0), as the Python path's first visit
    uses its base stream. Later visits get an independent nonzero u64.
    """
    if n == 0:
        return 0
    digest = sha256(f"spire-learn-playout-v1:{research_seed}:{world}:{n}".encode()).digest()
    return int.from_bytes(digest[:8], "big") or 1


@dataclass(frozen=True)
class PlayoutOutcome:
    value: float
    refused: bool = False


def playout_branches(search, branches, *, max_steps: int, stream: int = 0) -> list[PlayoutOutcome]:
    """Native playouts of distinct live branches, scored as terminal reward, else half the HP balance.

    A refused playout scores -1 and is flagged. A faulted branch raises after
    the caller's handles close. Branch and arena ownership stay with the caller.
    """
    unsupported, _, _ = search_view._native_errors()
    options = {"rule": "uniform_exclude_verbs", "exclude_verbs": PLAYOUT_EXCLUDE_VERBS,
               "max_steps": max_steps, "playout_seed": stream}
    try:
        return _playout_call(search, list(branches), options)
    except unsupported:
        # A whole-call preflight refusal leaves all branches untouched without an
        # index. Single-branch calls attribute the refusal.
        outcomes = []
        for branch in branches:
            try:
                outcomes.extend(_playout_call(search, [branch], options))
            except unsupported:
                outcomes.append(PlayoutOutcome(-1.0, True))
        return outcomes


def _playout_call(search, branches, options) -> list[PlayoutOutcome]:
    _, _, fault_error = search_view._native_errors()
    result = search.playout_many(branches, **options)
    for index, detail in enumerate(result["fault"]):
        if detail is not None:
            error = fault_error(f"public search playout branch {index}: {detail}")
            error.detail = detail
            raise error
    return [_score_playout(result, index) for index in range(len(branches))]


def _score_playout(result, index: int) -> PlayoutOutcome:
    if result["end_reason"][index] == "refused":
        return PlayoutOutcome(-1.0, True)
    if result["terminated"][index]:
        return PlayoutOutcome(float(result["rewards"][index]))
    # step_limit and max_steps leave the branch live: score the summary.
    return PlayoutOutcome(0.5 * summary_balance(result, index))


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
                    branch_view: str = "full", playout_backend: str = "python",
                    value_rule: ValueRule = _FLAT, playout_policy: str = "uniform-v1",
                    root: RootContext | None = None) -> dict[int, ActionStat]:
    """Evaluate each action over paired worlds. The caller owns the search root.

    The native backend takes the root action with ``step_many`` and plays every
    live branch out in one ``playout_many`` call under the Gym's public rule.
    ``resource-v2`` needs ``root``, the context of the decision being searched.
    """
    if playout_backend not in PLAYOUT_BACKENDS:
        raise ValueError(f"playout_backend must be one of {PLAYOUT_BACKENDS}")
    check_value_options(value_rule, playout_policy, playout_backend, branch_view)
    v2 = value_rule.name == "resource-v2"
    if v2 and root is None:
        raise ValueError("resource-v2 needs the root context of the decision")
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

        if playout_backend == "native":
            active = _native_playouts(search, active, view, playout_steps, research_seed, branch_view)
        for depth in range(playout_steps + 1):
            if not active:
                break
            branches = [item.branch for item in active]
            actions = [item.action for item in active]
            continuing = []
            for item, result in zip(active, step_branches(search, branches, actions, view=branch_view), strict=True):
                if result.refused or result.terminated:
                    if v2 and not result.refused:
                        item.value = terminal_score(value_rule, root, view, result.observation,
                                                    result.reward, result.info)
                    else:
                        item.value = result.reward
                    item.refused = result.refused
                elif result.truncated or depth == playout_steps:
                    if v2:
                        item.value = cutoff_value(value_rule, root, view, result.observation)
                    else:
                        item.value = 0.5 * view.balance(result.observation)
                else:
                    if playout_policy == "uniform-v1":
                        item.action = view.heuristic_choice(result.observation, item.rng)
                    else:
                        item.action = view.playout_choice(result.observation, item.rng, playout_policy)
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


def _native_playouts(search, active, view, playout_steps, research_seed, branch_view):
    """Take the root action, then score live branches natively. Returns no branches to continue.

    Gym's playout rule and summary serve only flat-v1 with uniform-v1; ``check_value_options`` refuses the rest.
    """
    results = step_branches(search, [item.branch for item in active],
                            [item.action for item in active], view=branch_view)
    live = []
    for item, result in zip(active, results, strict=True):
        if result.refused or result.terminated:
            item.value = result.reward
            item.refused = result.refused
        elif result.truncated or not playout_steps:
            item.value = 0.5 * view.balance(result.observation)
        else:
            live.append(item)
    results = None
    if live:
        outcomes = playout_branches(search, [item.branch for item in live], max_steps=playout_steps,
                                    stream=playout_seed(research_seed, 0, 0))
        for item, outcome in zip(live, outcomes, strict=True):
            item.value, item.refused = outcome.value, outcome.refused
    return []
