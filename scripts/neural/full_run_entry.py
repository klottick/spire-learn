#!/usr/bin/env python3
"""Play whole Ironclad A20 Standard runs with a combat player and noncombat-heuristic-v1 (gate R1 input).

The run panel is ``RUN_PANEL_SIZE`` selection-partition runs with environment seeds
``environment_seed_for("selection", replica=0, ordinal=RUN_PANEL_SEED_OFFSET + i)``, experienced
creation and a full Neow; its identity is the sha256 of the seed list, the partition and the
settings. One call plays ``--runs FIRST:COUNT`` (at most ``CHUNK_RUNS`` runs) into a fresh
``--output`` under ``artifacts/development/neural-combat/full-run/``. ``--smoke`` plays the same
indices on ordinals past the panel (never a panel run) for harness checks.

Every boundary is routed by ``parse_lifecycle(combat_view()["info"]).active``: combat boundaries
(events' combats too) go to ``--combat-player``, the rest to ``noncombat-heuristic-v1``.

- ``teacher:<name>`` loads ``<name>`` from ``--teachers`` with ``allow_uncalibrated=True`` (recorded)
  and plays each move through ``search_teacher_entry._decision``, the search being
  ``RunEnv.public_search(..., observation="compact", capacity_preset=...)`` on the bridged compact
  frame. A root refused with a ``RunUnsupportedBoundaryError`` whose class is in
  ``ROOT_REFUSAL_CLASSES`` plays the teacher's ``playout_choice`` and is counted; any other refusal
  class, and every ``RunStateError`` at a live root, is a fault and raises.
- ``checkpoint:<path>`` plays ``CheckpointPlayer(deterministic=True)`` on the bridged ``CombatFrame``.

Caps: ``TURN_CAP`` actions in one public turn and ``FIGHT_CAP`` actions in one fight, ``RUN_CAP``
decisions in one run. A cap censors the run (its floors and the reason are kept). Gym faults,
bridge overflows and heuristic errors are errors, never losses: a failed call leaves no
``summary.json``; rerun the chunk into a fresh directory.

Decision ``d`` of run ``i`` searches with research seed ``H(seed, i, d)`` and the run's teacher
Random is seeded ``H(seed, "run", i)``, where ``H(*parts)`` is the first 16 hex digits of
``content_sha256(list(parts))``: results do not depend on chunking or scheduling.

Outputs: ``runs.jsonl`` (one deterministic row per run), ``fights.jsonl`` (the teacher entry's fight
rows plus the run, act, floor and root counts), ``report.json`` (CPU and wall time per run, host) and
``summary.json`` (aggregates and provenance; written last).
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
from time import perf_counter, process_time

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from scripts.neural import distill_label_entry as label
from scripts.neural import search_teacher_entry as entry
from spire_learn.neural.cases import environment_seed_for
from spire_learn.neural.config import model_config_hash_payload
from spire_learn.neural.noncombat_heuristic import HEURISTIC_ID, CardPriority, NoncombatHeuristic, RoomFlags
from spire_learn.neural.scenario_support import content_sha256

# Pinned plan constants.
GYM_REVISION = "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1"
RUN_PANEL_SIZE = 64
RUN_PANEL_SEED_OFFSET = 2 ** 34
TURN_CAP = 500
FIGHT_CAP = 1024
RUN_CAP = 10_000
CHUNK_RUNS = 8

SCHEMA = "spire-learn-full-run-v1"
PARTITION = "selection"
CHARACTER = "ironclad"
ASCENSION = 20
OUTPUT_ROOT = Path("artifacts/development/neural-combat/full-run")
PRIORITY_PATH = Path("configs/neural/full-run/ironclad-priority-v1.json")
# The pre-registered priority list for gate R1; any other file is refused unless --smoke.
PRIORITY_SHA256 = "9dd28b208c16e6fb05daced54ca264d287c9437c95d8aad4500e861448f36548"
VICTORY_TERMINALS = ("act3_no_keys", "heart_victory")
CENSOR_REASONS = ("turn_cap", "fight_cap", "run_cap")
PANEL_SETTINGS = {
    "character": CHARACTER, "ascension": ASCENSION, "profile": "standard", "creation": "experienced",
    "neow": "full", "keys": "skipped", "turn_cap": TURN_CAP, "fight_cap": FIGHT_CAP, "run_cap": RUN_CAP,
    "panel_size": RUN_PANEL_SIZE, "seed_offset": RUN_PANEL_SEED_OFFSET,
    "seed_rule": "environment_seed_for('selection', replica=0, ordinal=seed_offset + index)",
}
_COMPACT_PROBE_ENTRY = {
    "id": "full-run-compact-descriptor", "character": CHARACTER, "ascension": ASCENSION, "encounter": "cultist",
    "hp": 75, "max_hp": 75, "gold": 99, "relics": ["burning_blood"], "relic_counters": {}, "potions": [None, None],
    "deck": [{"card": card, "upgrades": 0, "misc": 0} for card in ["bash"] + ["defend"] * 4 + ["strike"] * 5],
}


class RunFault(RuntimeError):
    """A combat boundary the harness cannot play faithfully (a state error at a live root, an unknown refusal)."""


def run_ordinal(index: int, *, smoke: bool = False) -> int:
    return RUN_PANEL_SEED_OFFSET + (RUN_PANEL_SIZE if smoke else 0) + index


def run_seed(index: int, *, smoke: bool = False) -> int:
    return environment_seed_for(PARTITION, replica=0, ordinal=run_ordinal(index, smoke=smoke))


def panel_seeds() -> list[int]:
    return [run_seed(index) for index in range(RUN_PANEL_SIZE)]


def panel_identity() -> str:
    """sha256 of the run panel's seed list, partition and settings."""
    return content_sha256({"seeds": panel_seeds(), "partition": PARTITION, "settings": PANEL_SETTINGS})


def run_id(index: int, *, smoke: bool = False) -> str:
    return f"{'smoke' if smoke else 'run'}-{index:02d}"


def parse_player(text: str) -> tuple[str, str]:
    kind, sep, value = text.partition(":")
    if not sep or not value or kind not in ("teacher", "checkpoint"):
        raise argparse.ArgumentTypeError("must be teacher:<name> or checkpoint:<path>")
    return kind, value


def _runs(text: str) -> tuple[int, int]:
    first, sep, count = text.partition(":")
    if not sep or not first.isdigit() or not count.isdigit() or not 1 <= int(count) <= CHUNK_RUNS:
        raise argparse.ArgumentTypeError(f"must be FIRST:COUNT with 1 <= COUNT <= {CHUNK_RUNS}")
    return int(first), int(count)


@dataclass(frozen=True)
class RunJob:
    index: int
    seed: int
    run_id: str
    research_seed: int  # the call's --seed


@dataclass(frozen=True)
class PlayContext:
    """What every worker needs (plain, picklable values)."""

    player_kind: str
    teacher: object
    checkpoint: str | None
    mapping: str | None
    priority_path: str
    gym_revision: str


# --- per-process resources ------------------------------------------------------------------

_CACHE: dict = {}


def compact_descriptor() -> dict:
    """The live compact Standard combat descriptor at the pinned preset (from a throwaway standalone env)."""
    if "compact" not in _CACHE:
        from spire_lab.gymnasium import StandardCombatEnv
        from spire_learn.neural.gym_contracts import STANDARD_COMPACT_PRESET

        catalog = {"catalog_schema_id": "spire-gym-standard-combat-catalog-v1", "profile_id": "standard-combat-v1",
                   "entries": [_COMPACT_PROBE_ENTRY]}
        probe = StandardCombatEnv(catalog, partition=PARTITION, observation="compact",
                                  capacity_preset=STANDARD_COMPACT_PRESET)
        try:
            _CACHE["compact"] = probe.observation_spec()
        finally:
            probe.close()
    return _CACHE["compact"]


def _resources(context: PlayContext, env):
    """(bridge, compact View, heuristic, player, turn column), built once per process."""
    key = ("resources", context.player_kind, context.checkpoint, context.priority_path, context.gym_revision)
    if key not in _CACHE:
        from spire_learn.neural.compact_bridge import CompactBridge
        from spire_learn.neural.schema import parse_combat_descriptor
        from spire_learn.neural.search_view import View

        compact = compact_descriptor()
        schema = parse_combat_descriptor(compact, revision=context.gym_revision)
        bridge = CompactBridge(env.combat_observation_spec(), compact, schema)
        heuristic = NoncombatHeuristic(env.observation_spec(), CardPriority.load(context.priority_path))
        player = None
        if context.player_kind == "checkpoint":
            from spire_learn.neural.policy_player import CheckpointPlayer, standard_ff_identity

            player = CheckpointPlayer(context.checkpoint, schema, seed=0, model_config=label.MIXED_MODEL_CONFIG,
                                      expected_identity=standard_ff_identity(label.MIXED_MODEL_CONFIG, context.mapping),
                                      deterministic=True)
        _CACHE[key] = (bridge, View(compact), heuristic, player, label._turn_column(compact))
    return _CACHE[key]


class RunSearchRoot:
    """``public_search`` on the live Run, with Ruling J: only documented root refusals stay refusals.

    ``_decision`` treats a ``RunUnsupportedBoundaryError`` or ``RunStateError`` as a refused root. Here a
    ``RunStateError`` (terminal, not reset, closed, busy, stale, budget) is a caller or state bug and an
    unsupported boundary outside ``ROOT_REFUSAL_CLASSES`` is unknown; both raise ``RunFault`` instead.
    """

    def __init__(self, env):
        self.env = env

    def public_search(self, research_seed, **options):
        from spire_learn.neural import search_view

        unsupported, state_error, _ = search_view._native_errors()
        try:
            return self.env.public_search(research_seed, **options)
        except state_error as error:
            raise RunFault(f"RunStateError at a live combat root: {type(error).__name__}: {str(error)[:120]}") from error
        except unsupported as error:
            refusal = root_refusal(error)
            if refusal not in search_view.ROOT_REFUSAL_CLASSES:
                raise RunFault(f"undocumented root refusal class: {refusal!r}") from error
            raise


def root_refusal(error: BaseException) -> str:
    """The class of a Run root refusal: the error text exactly, the one place it is normalised.

    Gym v1.2.5 raises ``RunUnsupportedBoundaryError(u.reason())`` (crates/spire-python/src/run_env.rs:39) with
    ``u.reason()`` a static class string (crates/spire-core/src/combat/belief.rs:61-75; mapped at
    crates/spire-env/src/public_search.rs:32 and :34): no prefix, no detail and under 80 characters, so the text
    is the class and ``search_view.refusal_class`` (which keeps the first 80 characters) returns the same string.
    """
    return str(error)


# --- one run --------------------------------------------------------------------------------

def _context_value(heuristic, observation, field_name):
    return heuristic._scalar(observation, "context", field_name)


def _boss_node(heuristic, observation) -> bool:
    """Whether the Run's current map node is a boss node (read from the public map)."""
    row = heuristic._scalar(observation, "current_node", "row")
    if row is None or row < 0:
        return False
    return heuristic._name("map_kind", heuristic._cell(observation, "map_nodes", row, "kind")) == "boss"


def fight_resource_fields(view, start, end) -> dict:
    """The teacher entry's resource fields of one fight, and ``end_state``.

    ``end`` is the settled state read at the first boundary after the fight (``entry._resources``); None when
    the next boundary is already another combat, where no boundary shows the settled state: then the end fields
    are None and ``end_state`` is ``"unavailable"``, never read from the next combat.
    """
    if end is not None:
        return {**entry._resource_fields(view, start, end), "end_state": "settled"}
    hp_start, max_hp_start, held = start
    return {"hp_start": int(hp_start), "max_hp_start": int(max_hp_start), "max_hp_end": None,
            "potions_start": sum(held.values()), "potions_end": None, "potion_hp_spent": None,
            "end_state": "unavailable"}


def noncombat_move(heuristic, observation, legal, flags: RoomFlags, rules: Counter, fallbacks: Counter) -> int:
    """The heuristic's move at a non-combat boundary; every rule is tallied and every fallback counted."""
    choice = heuristic.choose(observation, legal, flags)
    flags.record(heuristic.room_key(observation), choice)
    rules[choice.rule] += 1
    if choice.fallback:
        fallbacks[choice.rule] += 1
    return choice.action


def play_run(env, job: RunJob, context: PlayContext) -> dict:
    """Play one whole run; returns ``{"run", "fights", "timing"}``. Faults raise."""
    from spire_learn.neural.schema import parse_lifecycle
    from spire_learn.neural.views import STANDARD_COMBAT_V1

    wall_start, cpu_start = perf_counter(), process_time()
    bridge, view, heuristic, player, turn_column = _resources(context, env)
    search = RunSearchRoot(env)
    rng = random.Random(label.derive_seed(job.research_seed, "run", job.index))
    flags = RoomFlags()
    observation, info = env.reset(seed=job.seed)
    creation = info.get("creation") or {}
    if creation.get("neow") != "full" or any(len(act) != 3 for act in creation.get("seen_bosses", ())):
        raise RunFault(f"run {job.run_id} did not start with experienced creation and a full Neow")

    roots, refusals, rules, fallbacks = Counter(), Counter(), Counter(), Counter()
    fights, boss_entries = [], []
    potions_used = potions_discarded = branch_refusals = 0
    decisions = combat_decisions = 0
    floor_max = act_max = 0
    fight = None
    censor = None
    terminal_reason = None

    def start_fight(lifecycle, compact):
        nonlocal fight
        act, floor = _context_value(heuristic, observation, "act"), _context_value(heuristic, observation, "floor")
        hp, max_hp = view.player_hp_now(compact), view.player_max_hp(compact)
        boss = _boss_node(heuristic, observation)
        if boss:
            boss_entries.append({"act": act, "floor": floor, "hp": hp, "max_hp": max_hp})
        fight = {"visit": lifecycle.active_visit, "act": act, "floor": floor, "boss": boss, "steps": 0,
                 "turn": None, "turn_actions": 0, "start": entry._resources(view, compact),
                 "roots": Counter(), "cpu": process_time(), "wall": perf_counter()}

    def finish_fight(lifecycle, end_view, censor_reason=None):
        """Close the fight; ``end_view`` None when no boundary shows its settled state (a combat follows directly)."""
        nonlocal fight
        last_exit = lifecycle.last_exit
        if censor_reason is None:
            if not isinstance(last_exit, Mapping) or last_exit.get("visit") != fight["visit"]:
                raise RunFault(f"fight {fight['visit']} of {job.run_id} ended without its settled exit")
            outcome, hp = last_exit["outcome"], last_exit.get("hp_after_cleanup")
        else:
            outcome, hp = "censored", None
        end = None if end_view is None else entry._resources(view, bridge.arrays(end_view))
        fights.append({
            "case_id": f"{job.run_id}/fight-{len(fights):03d}", "run": job.index, "run_id": job.run_id,
            "act": fight["act"], "floor": fight["floor"], "boss": fight["boss"],
            "encounter": None if censor_reason else last_exit.get("encounter"),
            "outcome": outcome, "censor_reason": censor_reason, "steps": fight["steps"],
            "hp_after_cleanup": None if hp is None else int(hp),
            **fight_resource_fields(view, fight["start"], end),
            "roots": dict(fight["roots"]),
            "cpu_seconds": process_time() - fight["cpu"], "wall_seconds": perf_counter() - fight["wall"],
        })
        fight = None

    while True:
        act = _context_value(heuristic, observation, "act") or 0
        floor = _context_value(heuristic, observation, "floor") or 0
        act_max, floor_max = max(act_max, act), max(floor_max, floor)
        combat = env.combat_view()
        lifecycle = parse_lifecycle(combat["info"], view=STANDARD_COMBAT_V1)
        legal = [index for index, flag in enumerate(observation["action_mask"]) if flag]
        if not legal:
            raise RunFault(f"{job.run_id} has a live boundary without a legal action at decision {decisions}")
        if decisions == RUN_CAP:
            censor = "run_cap"
            if fight is not None:
                finish_fight(lifecycle, combat["obs"], "run_cap")
            break
        if lifecycle.active:
            compact, frame = bridge(combat["obs"]) if player is not None else (bridge.arrays(combat["obs"]), None)
            if view.legal(compact) != legal:
                raise RunFault(f"{job.run_id}: the bridged combat legal set differs from the Run's legal indices")
            if fight is not None and fight["visit"] != lifecycle.active_visit:
                # A combat that follows another with no boundary between: the next combat's first view is not
                # the previous fight's settled state, so its end resources are recorded as unavailable.
                finish_fight(lifecycle, None)
            if fight is None:
                start_fight(lifecycle, compact)
            turn = entry._public_turn(compact, turn_column)
            if fight["turn"] is not None:
                fight["turn_actions"] = fight["turn_actions"] + 1 if turn == fight["turn"] else 0
            fight["turn"] = turn
            if fight["turn_actions"] == TURN_CAP or fight["steps"] == FIGHT_CAP:
                censor = "turn_cap" if fight["turn_actions"] == TURN_CAP else "fight_cap"
                finish_fight(lifecycle, combat["obs"], censor)
                break
            if player is not None:
                action, root = player.act(frame, legal), "policy"
            else:
                seed = label.derive_seed(job.research_seed, job.index, decisions)
                root, action, _, _, refused, _ = entry._decision(search, compact, legal, view, context.teacher,
                                                                 rng, seed)
                branch_refusals += refused
                if root not in ("searched", "single_legal"):
                    refusals[root] += 1
                    root = "refused"
            if action not in legal:
                raise RunFault(f"{job.run_id}: the combat player chose illegal action {action}")
            roots[root] += 1
            fight["roots"][root] += 1
            fight["steps"] += 1
            verb = int(compact["actions"][action][view.verb])
            potions_used += verb == view.use_potion
            potions_discarded += verb == view.discard
            combat_decisions += 1
        else:
            if fight is not None:
                finish_fight(lifecycle, combat["obs"])
            action = noncombat_move(heuristic, observation, legal, flags, rules, fallbacks)
        observation, _, terminated, truncated, info = env.step(action)
        decisions += 1
        if truncated:
            raise RunFault(f"{job.run_id}: a Standard Run step reported truncation")
        if terminated:
            terminal_reason = info.get("terminal_reason")
            act_max = max(act_max, _context_value(heuristic, observation, "act") or 0)
            floor_max = max(floor_max, _context_value(heuristic, observation, "floor") or 0)
            if fight is not None:
                combat = env.combat_view()
                finish_fight(parse_lifecycle(combat["info"], view=STANDARD_COMBAT_V1), combat["obs"])
            break
    if terminal_reason is not None and terminal_reason not in ("death", "empty_deck", *VICTORY_TERMINALS):
        raise RunFault(f"{job.run_id} ended with an unknown terminal reason {terminal_reason!r}")
    died = terminal_reason == "death"
    run = {
        "index": job.index, "run_id": job.run_id, "seed": job.seed,
        "floors_reached": floor_max, "act_reached": act_max,
        "terminal_reason": terminal_reason, "victory": terminal_reason in VICTORY_TERMINALS,
        "death": {"act": _context_value(heuristic, observation, "act"),
                  "floor": _context_value(heuristic, observation, "floor")} if died else None,
        "censored": censor is not None, "censor_reason": censor,
        "censor_at": {"act": _context_value(heuristic, observation, "act"),
                      "floor": _context_value(heuristic, observation, "floor")} if censor else None,
        "boss_entries": boss_entries, "potions_used": potions_used, "potions_discarded": potions_discarded,
        "decisions": decisions, "combat_decisions": combat_decisions, "noncombat_decisions": decisions - combat_decisions,
        "fights": len(fights), "fight_outcomes": dict(Counter(row["outcome"] for row in fights)),
        "roots": dict(roots), "refused_roots": roots["refused"], "refusal_classes": dict(refusals),
        "branch_refusals": branch_refusals, "fallbacks": dict(fallbacks), "fallback_total": sum(fallbacks.values()),
        "heuristic_rules": dict(rules),
    }
    timing = {"index": job.index, "run_id": job.run_id, "cpu_seconds": process_time() - cpu_start,
              "wall_seconds": perf_counter() - wall_start, "decisions": decisions}
    return {"run": run, "fights": fights, "timing": timing}


def _play_job(item):
    job, context = item
    from spire_lab.gymnasium import RunEnv

    env = RunEnv(CHARACTER, ASCENSION, profile="standard", partition=PARTITION)
    try:
        return play_run(env, job, context)
    finally:
        env.close()


def _results(items, parallel):
    if parallel == 1:
        yield from map(_play_job, items)
        return
    pool = ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("spawn"))
    try:
        yield from pool.map(_play_job, items)
    finally:
        pool.shutdown(cancel_futures=True)


# --- chunk ----------------------------------------------------------------------------------

def _check_gym(teacher):
    """The live Gym build, revision, contract and compact view (the labeller's checks, recipe-free)."""
    from spire_lab import _native
    from spire_learn.neural.gym_contracts import STANDARD_COMPACT_PRESET, accepts_recorded_revision, contract_for_revision
    from spire_learn.neural.search_one_move import require_tiered_playouts

    gym_build = dict(_native.build_info())
    live_revision = gym_build.get("revision")
    if not accepts_recorded_revision(live_revision, GYM_REVISION):
        raise ValueError(f"the full-run harness requires Gym revision {GYM_REVISION} or one in its rules epoch")
    live = contract_for_revision(live_revision)
    if STANDARD_COMPACT_PRESET not in live.standard_compact_fingerprints:
        raise ValueError("the full-run harness requires a Gym with compact Standard views")
    if teacher is not None:
        _, playout_policy = entry._value_options(teacher)
        if entry._native_backend(teacher) and playout_policy != "uniform-v1":
            require_tiered_playouts(live)
    compact = compact_descriptor()
    if compact["fingerprint"] != live.standard_compact_fingerprints[STANDARD_COMPACT_PRESET]:
        raise ValueError("the live compact combat descriptor is not the contract's preset view")
    mapping = live.raw_mode_mappings_for_view(compact["fingerprint"])["standard-standalone"]
    return gym_build, live_revision, live, mapping


def _aggregate(runs: list[dict]) -> dict:
    floors = sorted(run["floors_reached"] for run in runs)
    count = len(runs)
    middle = count // 2
    total = Counter()
    for run in runs:
        total.update({"decisions": run["decisions"], "combat_decisions": run["combat_decisions"],
                      "fights": run["fights"], "refused_roots": run["refused_roots"],
                      "branch_refusals": run["branch_refusals"], "fallbacks": run["fallback_total"],
                      "potions_used": run["potions_used"]})
    return {
        "runs": count, "victories": sum(run["victory"] for run in runs),
        "deaths": sum(run["terminal_reason"] == "death" for run in runs),
        "censored": dict(Counter(run["censor_reason"] for run in runs if run["censored"])),
        "terminal_reasons": dict(Counter(str(run["terminal_reason"]) for run in runs)),
        "floors_mean": sum(floors) / count if count else None,
        "floors_median": (None if not count else floors[middle] if count % 2 else (floors[middle - 1] + floors[middle]) / 2),
        "floors_min": floors[0] if floors else None, "floors_max": floors[-1] if floors else None,
        "death_acts": dict(Counter(str(run["death"]["act"]) for run in runs if run["death"])),
        "totals": dict(total),
        "refusal_classes": dict(sum((Counter(run["refusal_classes"]) for run in runs), Counter())),
        "fallback_rules": dict(sum((Counter(run["fallbacks"]) for run in runs), Counter())),
    }


def run_chunk(args, output: Path) -> dict:
    wall_start = perf_counter()
    kind, value = args.combat_player
    teacher = teacher_sha = calibrated = None
    checkpoint = None
    if kind == "teacher":
        if args.teachers is None:
            raise ValueError("teacher:<name> needs --teachers")
        teacher, teacher_sha, calibrated = entry._teacher(args.teachers, value, None, allow_uncalibrated=True)
    else:
        checkpoint = label.checkpoint_sha256(Path(value))
    gym_build, live_revision, live, mapping = _check_gym(teacher)
    priority = CardPriority.load(args.priority)
    if not args.smoke and priority.sha256 != PRIORITY_SHA256:
        raise ValueError("--priority is not the pre-registered priority list (only --smoke may use another)")

    first, count = args.runs
    if not args.smoke and first + count > RUN_PANEL_SIZE:
        raise ValueError(f"--runs must lie inside the {RUN_PANEL_SIZE}-run panel")
    jobs = [RunJob(index=index, seed=run_seed(index, smoke=args.smoke), run_id=run_id(index, smoke=args.smoke),
                   research_seed=args.seed) for index in range(first, first + count)]
    context = PlayContext(player_kind=kind, teacher=teacher, checkpoint=None if checkpoint is None else str(value),
                          mapping=mapping if kind == "checkpoint" else None,
                          priority_path=str(args.priority), gym_revision=live_revision)
    compact = compact_descriptor()
    provenance = {
        "schema": SCHEMA, "gym_revision": live_revision, "gym_build": gym_build, "rules_epoch": live.rules_epoch,
        "learner": label._learner_state(), "combat_player": f"{kind}:{value}" if kind == "teacher" else "checkpoint",
        "teacher": value if kind == "teacher" else None, "config_sha256": teacher_sha,
        "teacher_config": None if teacher is None else entry._teacher_config(teacher),
        "teachers_sha256": None if args.teachers is None or kind != "teacher" else label._file_sha256(args.teachers),
        "calibrated": calibrated, "uncalibrated_allowed": True if kind == "teacher" else None,
        "checkpoint": checkpoint, "checkpoint_model_config": model_config_hash_payload(label.MIXED_MODEL_CONFIG) if checkpoint else None,
        "checkpoint_deterministic": True if checkpoint else None,
        "noncombat_player": HEURISTIC_ID,
        "priority": {"id": priority.id, "path": str(args.priority.relative_to(_ROOT)) if args.priority.is_relative_to(_ROOT)
                     else str(args.priority), "sha256": priority.sha256},
        "panel": None if args.smoke else {"identity": panel_identity(), "size": RUN_PANEL_SIZE,
                                          "seed_offset": RUN_PANEL_SEED_OFFSET, "settings": PANEL_SETTINGS},
        "smoke": args.smoke, "partition": PARTITION, "runs": [first, count],
        "run_seeds": [job.seed for job in jobs], "seed": args.seed,
        "research_seed_rule": "decision d of run i: H(seed, i, d); teacher Random: H(seed, 'run', i)",
        "seed_hash": "first 16 hex digits of content_sha256",
        "turn_cap": TURN_CAP, "fight_cap": FIGHT_CAP, "run_cap": RUN_CAP,
        "character": CHARACTER, "ascension": ASCENSION, "creation": "experienced", "neow": "full",
        "capacity_preset": compact["semantics"]["capacity_preset"], "compact_fingerprint": compact["fingerprint"],
        "combat_view_fingerprint": live.standard_combat_fingerprint,
        "victory_terminals": list(VICTORY_TERMINALS), "gpu_seconds": 0, "parallel": args.parallel,
        "priority_preregistered": priority.sha256 == PRIORITY_SHA256,
        "thread_pin": {**{name: os.environ.get(name) for name in label._THREAD_VARIABLES},
                       "torch_threads": 1 if checkpoint else None},
    }

    output.mkdir(parents=True)
    runs, timings = [], []
    with (output / "runs.jsonl").open("x", encoding="utf-8") as run_stream, \
            (output / "fights.jsonl").open("x", encoding="utf-8") as fight_stream:
        for result in _results([(job, context) for job in jobs], args.parallel):
            runs.append(result["run"])
            timings.append(result["timing"])
            run_stream.write(json.dumps(result["run"], sort_keys=True, allow_nan=False) + "\n")
            for row in result["fights"]:
                fight_stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    label._write_json_atomic(output / "report.json", {
        "schema": "spire-learn-full-run-report-v1", "parallel": args.parallel, "host": platform.node(),
        "cpu_count": os.cpu_count(), "run_wall_seconds": perf_counter() - wall_start,
        "run_cpu_seconds": sum(row["cpu_seconds"] for row in timings), "gpu_seconds": 0, "timings": timings})
    summary = {"schema": SCHEMA, "aggregates": _aggregate(runs), "provenance": provenance}
    label._write_json_atomic(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combat-player", type=parse_player, required=True,
                        help="teacher:<name> (from --teachers) or checkpoint:<directory>")
    parser.add_argument("--teachers", type=Path, help="teacher configs for teacher:<name>")
    parser.add_argument("--runs", type=_runs, required=True, metavar="FIRST:COUNT",
                        help=f"panel indices to play (at most {CHUNK_RUNS} per call)")
    parser.add_argument("--smoke", action="store_true", help="play the indices on ordinals past the panel")
    parser.add_argument("--seed", type=int, required=True, help="research seed base")
    parser.add_argument("--priority", type=Path, default=PRIORITY_PATH)
    parser.add_argument("--output", type=Path, required=True, help=f"a fresh directory under {OUTPUT_ROOT}/")
    parser.add_argument("--parallel", type=entry._positive, default=1)
    args = parser.parse_args(argv)

    given = _ROOT / args.output
    if given.exists() or given.is_symlink():
        parser.error("--output already exists; each chunk is written to a fresh directory")
    output = given.resolve()
    if not output.is_relative_to((_ROOT / OUTPUT_ROOT).resolve()):
        parser.error(f"--output must resolve under {OUTPUT_ROOT}/")
    expected = os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT")
    if expected and not output.is_relative_to(Path(expected).resolve()):
        parser.error("--output must be inside SPIRE_LEARN_RUNNER_OUTPUT")
    if output.exists():
        parser.error("--output already exists; each chunk is written to a fresh directory")
    if not 0 <= args.seed < 1 << 64:
        parser.error("--seed must be an unsigned 64-bit integer")
    kind, value = args.combat_player
    if kind == "teacher" and args.teachers is None:
        parser.error("teacher:<name> needs --teachers")
    if kind == "checkpoint" and args.teachers is not None:
        parser.error("--teachers is only for teacher:<name>")
    if args.parallel > CHUNK_RUNS:
        parser.error(f"--parallel must be at most {CHUNK_RUNS}")
    first, count = args.runs
    if not args.smoke and first + count > RUN_PANEL_SIZE:
        parser.error(f"--runs must lie inside the {RUN_PANEL_SIZE}-run panel (indices 0..{RUN_PANEL_SIZE - 1}) "
                     "unless --smoke")
    args.priority = (_ROOT / args.priority).resolve()
    try:
        priority_sha256 = label._file_sha256(args.priority)
    except OSError as error:
        parser.error(f"--priority cannot be read: {error}")
    if not args.smoke and priority_sha256 != PRIORITY_SHA256:
        parser.error(f"--priority sha256 {priority_sha256} is not the pre-registered {PRIORITY_SHA256}; "
                     "only --smoke may play another list")
    if args.teachers is not None:
        args.teachers = (_ROOT / args.teachers).resolve()
    if kind == "checkpoint":
        args.combat_player = (kind, str((_ROOT / value).resolve()))
    # Spawned workers inherit these before any numeric library starts its thread pools.
    os.environ.update(dict.fromkeys(label._THREAD_VARIABLES, "1"))
    summary = run_chunk(args, output)
    print(json.dumps({"output": str(output), "aggregates": summary["aggregates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
