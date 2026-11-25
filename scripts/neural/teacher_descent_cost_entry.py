"""Measure paired public-search fork and replay costs on level-0 training fights."""

import argparse
from collections import Counter
from contextlib import closing
from itertools import batched
import json
from pathlib import Path
import random
from statistics import mean, median
import sys
from time import perf_counter, process_time

if __package__ in (None, ""):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / "src"), str(_repo)]

from scripts.neural import boss_winnability as bw
from spire_learn.neural import search_view
from spire_learn.neural.gym_contracts import GYM_V1_2_0_CONTRACT, STANDARD_COMPACT_PRESET
from spire_learn.neural.search_one_move import _close_branch

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/neural/baalorlord-standard-gym-1.2.0/recipe.json"
REVISION = GYM_V1_2_0_CONTRACT.source_revision
DEPTHS = (1, 2, 4, 8)
ROOTS = 50
SEED = 20261117
WORLD = 0
LIMITS = {
    "max_live_branches": 2,
    "max_branches": 1 + 2 * len(DEPTHS),
    "max_steps_per_branch": max(DEPTHS) + 2,
    "max_total_steps": max(DEPTHS) + sum(depth + 2 for depth in DEPTHS),
}


def training_roots(states):
    by_character = {}
    for state in states:
        if state["entry"]["ascension"] == 20 and bw._is_train(state["run_id"]):
            by_character.setdefault(state["character"], []).append(state)
    rng = random.Random(SEED)
    selected = []
    while len(selected) < ROOTS and by_character:
        for character in sorted(by_character):
            pool = by_character[character]
            selected.append(pool.pop(rng.randrange(len(pool))))
            if not pool:
                del by_character[character]
            if len(selected) == ROOTS:
                break
    if len(selected) != ROOTS:
        raise ValueError(f"Need {ROOTS} A20 training states, found {len(selected)}")
    return selected


def extend_prefix(branch, view, rng, prefix, observation, depth):
    """Advance the stored branch to depth, returning its public decision or stop reason."""
    while len(prefix) < depth:
        action = view.heuristic_choice(observation, rng)
        observation, _, terminated, truncated, _ = branch.step(action, view="full")
        prefix.append(action)
        if terminated or truncated:
            return None, "prefix_terminated" if terminated else "prefix_truncated"
    return observation, None


def timing_summary(samples):
    """Summarize (CPU seconds, wall seconds) for one method."""
    result = {"samples": len(samples)}
    for index, clock in enumerate(("cpu", "wall")):
        seconds = [sample[index] for sample in samples]
        result[f"mean_{clock}_seconds"] = mean(seconds) if seconds else None
        result[f"median_{clock}_seconds"] = median(seconds) if seconds else None
    return result


def timed_descent(open_branch, replay, view, rng, node_observation=None):
    """Time branch creation, optional replay and one heuristic step, excluding cleanup."""
    cpu_start = process_time()
    wall_start = perf_counter()
    branch = open_branch()
    try:
        for action in replay:
            node_observation, _, terminated, truncated, _ = branch.step(action, view="full")
            if terminated or truncated:
                raise RuntimeError("Replay reached a boundary before the measured step")
        action = view.heuristic_choice(node_observation, rng)
        successor = branch.step(action, view="full")
        elapsed = process_time() - cpu_start, perf_counter() - wall_start
        return elapsed, node_observation, action, successor
    finally:
        _close_branch(branch)


def check_pair(fork_result, replay_result):
    from numpy import array_equal

    _, fork_node, fork_action, fork_step = fork_result
    _, replay_node, replay_action, replay_step = replay_result
    if fork_action != replay_action or fork_step[1:4] != replay_step[1:4]:
        raise RuntimeError("Fork and replay produced different actions or outcomes")
    for left, right in ((fork_node, replay_node), (fork_step[0], replay_step[0])):
        if left.keys() != right.keys() or any(not array_equal(left[key], right[key]) for key in left):
            raise RuntimeError("Fork and replay produced different public observations")


def measure_root(env, entry, live_view, seed, timings, skips, root_refusals, fingerprints):
    observation, info = env.reset(seed=seed, options={"entry_id": entry["id"]})
    unsupported, state_error, _ = search_view._native_errors()
    if not live_view.legal(observation):
        if info.get("phase") != "terminal":
            raise RuntimeError("Live entry has no legal actions")
        for depth in DEPTHS:
            skips[depth]["terminal_at_reset"] += 1
        return
    try:
        search = env.public_search(
            seed * 1009, observation="compact", capacity_preset=STANDARD_COMPACT_PRESET,
            workers=1, limits=LIMITS,
        )
    except (unsupported, state_error) as error:
        root_refusals[search_view.refusal_class(error)] += 1
        for depth in DEPTHS:
            skips[depth]["root_refused"] += 1
        return

    with closing(search):
        descriptor = search.observation_spec()
        fingerprints.add(descriptor["fingerprint"])
        view = bw.View(descriptor)
        stored = None
        try:
            reason = None
            try:
                stored = search.branch(WORLD)
            except unsupported:
                reason = "prefix_refused"
            if stored is not None:
                observation, _ = stored.decision()
            prefix = []
            rng = random.Random(seed)
            for depth in DEPTHS:
                if reason is None:
                    try:
                        observation, reason = extend_prefix(stored, view, rng, prefix, observation, depth)
                    except unsupported:
                        reason = "prefix_refused"
                if reason is not None:
                    skips[depth][reason] += 1
                    continue

                # Alternate order across roots while pairing the heuristic RNG.
                fork_rng = random.Random()
                fork_rng.setstate(rng.getstate())
                replay_rng = random.Random()
                replay_rng.setstate(rng.getstate())
                methods = {
                    "fork": (stored.fork, (), view, fork_rng, observation),
                    "replay": (lambda: search.branch(WORLD), prefix, view, replay_rng),
                }
                order = ("fork", "replay") if seed % 2 else ("replay", "fork")
                results = {}
                refused = []
                for method in order:
                    try:
                        results[method] = timed_descent(*methods[method])
                    except unsupported:
                        refused.append(method)
                if refused:
                    skips[depth]["_and_".join(sorted(refused)) + "_refused"] += 1
                else:
                    check_pair(results["fork"], results["replay"])
                    for method, result in results.items():
                        timings[depth][method].append(result[0])
        finally:
            if stored is not None:
                _close_branch(stored)


def run(output):
    revision, states = bw._table_states(RECIPE)
    if revision != REVISION:
        raise ValueError(f"Expected Gym v1.2.0 table revision {REVISION}, got {revision}")
    bw._check_live_revision(revision)
    selected = training_roots(states)
    timings = {depth: {"fork": [], "replay": []} for depth in DEPTHS}
    skips = {depth: Counter() for depth in DEPTHS}
    root_refusals = Counter()
    fingerprints = set()
    for batch in batched(selected, 128):
        catalog = bw._catalog(batch, native=True)
        with closing(bw._combat_env(catalog)) as env:
            descriptor = env.observation_spec()
            view = bw.View(descriptor)
            for index, state in enumerate(batch):
                measure_root(env, state["entry"], view, SEED + index, timings, skips,
                             root_refusals, fingerprints)

    summary = {
        "schema": "spire-learn-teacher-descent-cost-v1",
        "provenance": bw._search_measurement_provenance(RECIPE, descriptor, "compact"),
        "gym_revision": revision,
        "seed": SEED,
        "world": WORLD,
        "split": "train",
        "level": 0,
        "ascension": 20,
        "roots_attempted": ROOTS,
        "roots_by_character": dict(Counter(state["character"] for state in selected)),
        "root_refusals": dict(root_refusals),
        "workers": 1,
        "capacity_preset": STANDARD_COMPACT_PRESET,
        "branch_observation_fingerprints": sorted(fingerprints),
        "limits": LIMITS,
        "depths": {
            depth: {
                **{method: timing_summary(samples) for method, samples in timings[depth].items()},
                "skips": dict(skips[depth]),
                "roots_skipped": ROOTS - len(timings[depth]["fork"]),
            }
            for depth in DEPTHS
        },
    }
    with (output / "teacher_descent_cost.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
