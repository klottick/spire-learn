"""Public-view tree search with world-specific fork points."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import ceil, isfinite, log, sqrt
import random
from time import process_time

from . import search_view
from .gym_contracts import STANDARD_COMPACT_PRESET
from .search_one_move import (
    PLAYOUT_BACKENDS, PLAYOUT_RULE_ID, ActionStat, _close_branch, playout_branches, playout_seed,
)
from .search_view import View

# Bound retained native branches and their Python observations per fight.
_MAX_LIVE_BRANCHES = 128


@dataclass(frozen=True, kw_only=True)
class TeacherConfig:
    simulations: int
    min_root_visits: int = 8
    worlds: int
    ucb_c: float
    max_depth: int
    widening: tuple[float, float] | None
    playout_steps: int = 80
    branch_layout: str = "compact"
    workers: int = 1
    # "python" is the reference rollout; "native" uses Gym's playout_many.
    playout_backend: str = "python"

    def __post_init__(self) -> None:
        for name in ("simulations", "worlds", "max_depth", "workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_root_visits", "playout_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.worlds > 2**64:
            raise ValueError("worlds cannot exceed the u64 sample space")
        if self.branch_layout not in ("compact", "dense") or self.workers > 256:
            raise ValueError("Invalid public search layout or worker count")
        if self.playout_backend not in PLAYOUT_BACKENDS:
            raise ValueError(f"playout_backend must be one of {PLAYOUT_BACKENDS}")
        if not isfinite(self.ucb_c) or self.ucb_c < 0:
            raise ValueError("ucb_c must be finite and nonnegative")
        if self.widening is not None:
            scale, exponent = self.widening
            if not isfinite(scale) or scale <= 0 or not 0 <= exponent <= 1:
                raise ValueError("widening requires a positive finite scale and an exponent in [0, 1]")


@dataclass(frozen=True)
class TeacherResult:
    choice: int
    stats: dict[int, ActionStat]
    simulations: int
    refused: int
    cpu_seconds: float


def config_identity(config: TeacherConfig) -> dict:
    """Identity fields of a config. Python teachers keep the pre-native fields, so their hashes hold."""
    fields = asdict(config)
    if config.playout_backend == "python":
        del fields["playout_backend"]
    else:
        fields["playout_rule_id"] = PLAYOUT_RULE_ID
    return fields


def config_sha256(config: TeacherConfig) -> str:
    payload = json.dumps(config_identity(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def search_options(config: TeacherConfig) -> dict:
    """Build the arena keyword arguments for the caller's env.public_search call."""
    def margin(count):
        return count + (count + 9) // 10

    path_steps = config.max_depth + config.playout_steps
    points = config.simulations * (config.max_depth - 1)
    options = {
        "observation": config.branch_layout,
        "workers": config.workers,
        "limits": {
            "max_live_branches": min(_MAX_LIVE_BRANCHES, margin(1 + points)),
            "max_branches": margin(config.simulations + points),
            "max_steps_per_branch": margin(path_steps),
            "max_total_steps": margin(config.simulations * path_steps),
        },
    }
    if config.branch_layout == "compact":
        options["capacity_preset"] = STANDARD_COMPACT_PRESET
    return options


@dataclass
class _Action:
    total: float = 0.0
    visits: int = 0
    refused: int = 0


@dataclass
class _Node:
    legal: Sequence[int]
    actions: dict[int, _Action] = field(init=False)
    children: dict[tuple[int, bytes], "_Node"] = field(default_factory=dict)
    visits: int = 0
    playouts_by_world: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.actions = {action: _Action() for action in self.legal}


@dataclass
class _ForkPoint:
    branch: object
    observation: Mapping
    node: _Node


def _public_key(observation: Mapping) -> bytes:
    digest = sha256()
    # Includes action_mask alongside the other public arrays. Frame names,
    # shapes and bytes so different tensor boundaries cannot alias.
    for name, tensor in sorted(observation.items()):
        buffer = memoryview(tensor)
        header = repr((name, buffer.format, buffer.shape, buffer.nbytes)).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(buffer if buffer.c_contiguous else buffer.tobytes())
    return digest.digest()


def _select(node, config, *, root=False):
    actions = node.legal
    if root:
        for action in actions:
            if node.actions[action].visits < max(1, config.min_root_visits):
                return action
    elif config.widening is not None:
        scale, exponent = config.widening
        count = max(1, ceil(scale * max(1, node.visits) ** exponent))
        actions = actions[:count]

    exploration = log(max(1, node.visits))
    best_action, best_score = actions[0], -float("inf")
    for action in actions:
        stats = node.actions[action]
        if not stats.visits:
            return action
        score = stats.total / stats.visits + config.ucb_c * sqrt(exploration / stats.visits)
        if score > best_score:
            best_action, best_score = action, score
    return best_action


def _playout_rng(node, research_seed, world, *, uniform):
    # Depth-one uniform evaluates each action/world pair with the reference
    # stream, even when two root actions reach the same public leaf.
    n = 0 if uniform else node.playouts_by_world.get(world, 0)
    if not uniform:
        node.playouts_by_world[world] = n + 1
    if n == 0:
        return random.Random(research_seed * 1_000_003 + world)
    # Pack (n, research_seed, world) with two 64-bit fields. For n >= 1
    # the result is at least 2**128, above every first-visit value for u64 inputs.
    return random.Random((n << 128) | (research_seed << 64) | world)


class _PlayoutRefused(Exception):
    """A native playout stopped at a refusal boundary. Counted like a refused branch."""


def _native_playout(search, branch, observation, view, config, research_seed, world, n):
    outcome, = playout_branches(search, [branch], max_steps=config.playout_steps,
                                stream=playout_seed(research_seed, world, n))
    if outcome.refused:
        raise _PlayoutRefused
    return outcome.value


def _playout(branch, observation, view, rng, steps):
    for _ in range(steps):
        action = view.heuristic_choice(observation, rng)
        observation, reward, terminated, truncated, _ = branch.step(action, view="full")
        if terminated:
            return reward
        if truncated:
            break
    return 0.5 * view.balance(observation)


def _simulate(search, world, root, first_action, view, config, nodes, points, capacity, path, *,
              research_seed, uniform):
    node, action = root, first_action
    prefix = ()
    branch = point = None
    try:
        for depth in range(1, config.max_depth + 1):
            path.append((node, action))
            prefix += (action,)
            # A public node can share statistics across worlds or paths. A
            # native fork point must preserve the exact world and action history.
            key = (world, prefix)
            cached = points.get(key)
            if cached is not None:
                points.move_to_end(key)
                if branch is not None:
                    _close_branch(branch)
                    branch = None
                point = cached
                observation, node = point.observation, point.node
                expanded = False
            else:
                if branch is None:
                    branch = search.branch(world) if point is None else point.branch.fork()
                if not nodes:
                    root_observation, _ = branch.decision()
                    nodes[_public_key(root_observation)] = root
                observation, reward, terminated, truncated, _ = branch.step(action, view="full")
                if terminated:
                    return reward
                if truncated:
                    return 0.5 * view.balance(observation)
                public_key = _public_key(observation)
                child = nodes.get(public_key)
                expanded = child is None
                if expanded:
                    child_legal = tuple(sorted(view.legal(observation)))
                    if not child_legal:
                        raise RuntimeError("Public search returned a nonterminal node with no legal actions")
                    child = nodes[public_key] = _Node(child_legal)
                node.children[action, public_key] = child
                node = child
                if capacity and depth < config.max_depth:
                    if len(points) == capacity:
                        _, evicted = points.popitem(last=False)
                        _close_branch(evicted.branch)
                    point = points[key] = _ForkPoint(branch, observation, node)
                    branch = None

            if expanded or depth == config.max_depth:
                path.append((node, None))
                if not config.playout_steps:
                    return 0.5 * view.balance(observation)
                if branch is None:
                    branch = point.branch.fork()
                if config.playout_backend == "native":
                    n = 0 if uniform else node.playouts_by_world.get(world, 0)
                    if not uniform:
                        node.playouts_by_world[world] = n + 1
                    return _native_playout(search, branch, observation, view, config, research_seed, world, n)
                rng = _playout_rng(node, research_seed, world, uniform=uniform)
                return _playout(branch, observation, view, rng, config.playout_steps)
            action = _select(node, config)
    finally:
        if branch is not None:
            _close_branch(branch)


def teach(search, legal: Sequence[int], view: View, config: TeacherConfig, *,
          research_seed: int) -> TeacherResult:
    """Run a tree search on the caller's arena with its matching view.

    A min_root_visits value of 0 is treated as 1.
    """
    start = process_time()
    if type(research_seed) is not int or not 0 <= research_seed < 2**64:
        raise ValueError("research_seed must be a nonnegative built-in u64 integer")
    legal = tuple(sorted(legal))
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("Expected a nonempty sequence of distinct legal actions")
    if config.simulations < len(legal) * max(1, config.min_root_visits):
        raise ValueError("Simulation budget cannot cover required root visits")

    unsupported, _, _ = search_view._native_errors()
    root = _Node(legal)
    nodes = {}
    points = OrderedDict()
    available = search.limits()["max_live_branches"] - search.counters()["live_branches"]
    capacity = max(0, min(available, _MAX_LIVE_BRANCHES) - 1)
    uniform = config.max_depth == 1 and config.simulations == config.worlds * len(legal)
    refused = 0
    try:
        for index in range(config.simulations):
            world = index % config.worlds
            action = legal[index // config.worlds] if uniform else _select(root, config, root=True)
            path = []
            rejected = False
            try:
                value = _simulate(
                    search, world, root, action, view, config, nodes, points, capacity, path,
                    research_seed=research_seed, uniform=uniform,
                )
            except (unsupported, _PlayoutRefused):
                rejected = True
                value = -1.0
                refused += 1
            for depth, (node, chosen) in enumerate(path):
                # A later return to the same public decision shares its node,
                # while root quotas and results count the first choice only.
                if depth and node is root:
                    continue
                node.visits += 1
                if chosen is not None:
                    stats = node.actions[chosen]
                    stats.total += value
                    stats.visits += 1
                    stats.refused += rejected
    finally:
        for point in points.values():
            _close_branch(point.branch)

    stats = {action: ActionStat(data.total / data.visits, data.visits, data.refused)
             for action, data in root.actions.items()}
    choice = max(legal, key=lambda action: stats[action].mean)
    return TeacherResult(choice, stats, config.simulations, refused, process_time() - start)
