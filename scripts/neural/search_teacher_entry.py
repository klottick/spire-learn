#!/usr/bin/env python3
"""Score a public-search teacher on selection or training-split Standard fights.

A failure during fight execution leaves partial fights.jsonl and decisions.jsonl
without a summary.json. Rerun in a fresh output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from itertools import batched
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
from time import perf_counter, process_time

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from scripts.neural.boss_winnability import _catalog, _is_train
from spire_learn.neural import search_view
from spire_learn.neural.cases import environment_seed_for
from spire_learn.neural.gym_contracts import STANDARD_COMPACT_PRESET, accepts_recorded_revision, contract_for_revision
from spire_learn.neural.recipes import build_panel, load_recipe
from spire_learn.neural.scenario_support import content_sha256, json_value
from spire_learn.neural.search_one_move import (
    check_value_options, one_move_values, playout_rule_id, require_tiered_playouts,
)
from spire_learn.neural.search_teacher import TeacherConfig, config_identity, config_sha256, search_options, teach
from spire_learn.neural.search_value import RootContext, ValueRule
from spire_learn.neural.search_view import View, refusal_class
from spire_learn.neural.standard_stream import StandardRecipe

TERMINAL_REWARDS = {"victory": 1, "death": -1, "escaped": 0}
# Gym 1.2.0 admits at most 4,096 live public branches per arena.
_CURRENT_ACTION_BATCH = 127
_FLAT = ValueRule("flat-v1")
# Fight rows price potions on this fixed yardstick, whatever the teacher's own prices.
_REFERENCE_PRICES = ValueRule("resource-v2", potion_hp=12.0,
                              potion_overrides=(("fairy_potion", 30.0), ("fruit_juice", 7.5)))


def sample_train_cases(states, count, seed):
    """Sample distinct, unmodified training entries in a reproducible order."""
    pool = sorted((state for state in states if _is_train(state["run_id"])),
                  key=lambda state: (state["run_id"], state["floor"]))
    selected = random.Random(seed).sample(pool, count)
    return [
        {"case_id": state["entry"]["id"], "entry": json_value(state["entry"]),
         "environment_seed": environment_seed_for("train", replica=0, ordinal=index)}
        for index, state in enumerate(selected)
    ]


def _vector_env(case, split):
    from spire_lab.gymnasium import StandardCombatVectorEnv

    return StandardCombatVectorEnv(
        _catalog([case], native=True), 1, partition=split, workers=1,
        autoreset_mode="disabled", fault_policy="freeze",
        observation="compact", capacity_preset=STANDARD_COMPACT_PRESET,
    )


def _reset_vector(env, case):
    import numpy as np

    return env.reset(seed=[case["environment_seed"]], options={
        "entry_ids": [case["entry"]["id"]], "reset_mask": np.ones(1, dtype=np.bool_),
    })


def preflight(cases, split):
    """Check every compact reset before executing any fights."""
    descriptor = None
    for case in cases:
        env = _vector_env(case, split)
        try:
            _reset_vector(env, case)
            if descriptor is None:
                descriptor = env.observation_spec()
        finally:
            env.close()
    return descriptor


def _vector_info(info):
    # Whitelist public terminal fields and honor Gymnasium presence masks.
    return {key: info[key][0] for key in ("phase", "terminal_reason", "hp_after_cleanup")
            if key in info and info["_" + key][0]}


class _SearchFight:
    """Play on the vector driver and obtain roots from Gym's standalone API.

    Gym 1.2.0 exposes no vector public_search. Keep the two public views in
    lockstep before using decision-local indices or opening another root.
    """

    def __init__(self, vector, standalone):
        self.vector = vector
        self.standalone = standalone

    @staticmethod
    def _match(batch, single):
        import numpy as np

        row = {key: values[0] for key, values in batch.items()}
        if row.keys() != single.keys() or any(
            not np.array_equal(values, single[key]) for key, values in row.items()
        ):
            raise RuntimeError("Vector and search session public observations diverged")
        return row

    def reset(self, case):
        batch, info = _reset_vector(self.vector, case)
        single, single_info = self.standalone.reset(
            seed=case["environment_seed"], options={"entry_id": case["entry"]["id"]})
        public = _vector_info(info)
        self._match_info(public, single_info)
        return self._match(batch, single), public

    @staticmethod
    def _match_info(public, single):
        if any(public.get(key) != single.get(key)
               for key in ("phase", "terminal_reason", "hp_after_cleanup")):
            raise RuntimeError("Vector and search session public boundary info diverged")

    def step(self, action):
        import numpy as np

        batch, rewards, terminated, truncated, info = self.vector.step(np.asarray([action], dtype="=i8"))
        single, reward, done, cut, single_info = self.standalone.step(action)
        if (float(rewards[0]), bool(terminated[0]), bool(truncated[0])) != (reward, done, cut):
            raise RuntimeError("Vector and search session transition outcomes diverged")
        public = _vector_info(info)
        self._match_info(public, single_info)
        return self._match(batch, single), reward, done, cut, public

    def public_search(self, research_seed, **options):
        return self.standalone.public_search(research_seed, **options)


@contextmanager
def _fight_env(case, split):
    from spire_lab.gymnasium import StandardCombatEnv

    with ExitStack() as resources:
        vector = _vector_env(case, split)
        resources.callback(vector.close)
        standalone = StandardCombatEnv(
            _catalog([case], native=True), partition=split,
            observation="compact", capacity_preset=STANDARD_COMPACT_PRESET,
        )
        resources.callback(standalone.close)
        yield _SearchFight(vector, standalone), vector.observation_spec()


def _value_options(teacher):
    """The teacher's value rule and playout policy, at their v1 defaults when unset."""
    if isinstance(teacher, TeacherConfig):
        return teacher.value_rule, teacher.playout_policy
    return teacher.get("value_rule", _FLAT), teacher.get("playout_policy", "uniform-v1")


def _decision(env, observation, legal, view, teacher, rng, research_seed):
    if len(legal) == 1:
        return "single_legal", legal[0], None, None, 0, None
    value_rule, playout_policy = _value_options(teacher)

    if isinstance(teacher, TeacherConfig):
        options = search_options(teacher, len(legal))
    else:
        width = teacher["samples"] * len(legal)
        depth = teacher["playout_steps"] + 2
        options = {
            "observation": "compact", "capacity_preset": STANDARD_COMPACT_PRESET, "workers": 1,
            "limits": {"max_live_branches": min(width, teacher["samples"] * _CURRENT_ACTION_BATCH) + 8,
                       "max_branches": width + 8,
                       "max_steps_per_branch": depth, "max_total_steps": width * depth},
        }

    unsupported, state_error, _ = search_view._native_errors()
    start = process_time()
    try:
        arena = env.public_search(research_seed, **options)
    except (unsupported, state_error) as error:
        if playout_policy == "uniform-v1":
            fallback = view.heuristic_choice(observation, rng)
        else:
            fallback = view.playout_choice(observation, rng, playout_policy)
        return refusal_class(error), fallback, None, None, 0, None

    try:
        # resource-v2 scores branches against this decision's own public resources; v1 passes no root.
        root = ({"root": RootContext.from_observation(view, observation, value_rule)}
                if value_rule.name == "resource-v2" else {})
        branch_view = View(arena.observation_spec())
        if isinstance(teacher, TeacherConfig):
            result = teach(arena, legal, branch_view, teacher, research_seed=research_seed, **root)
            choice, stats, simulations = result.choice, result.stats, result.simulations
        else:
            simulations = teacher["samples"] * len(legal)
            stats = {}
            for group in batched(legal, _CURRENT_ACTION_BATCH):
                stats.update(one_move_values(arena, group, branch_view, **teacher, research_seed=research_seed,
                                             **root))
            best = max(stat.mean for stat in stats.values())
            choice = rng.choice([action for action, stat in stats.items() if stat.mean == best])
    finally:
        arena.close()
    return ("searched", choice, {str(action): [stat.mean, stat.visits] for action, stat in stats.items()},
            process_time() - start, sum(stat.refused for stat in stats.values()), simulations)


def _public_turn(observation, turn_column):
    if observation["player_known"][turn_column] != 1:
        raise RuntimeError("Fight turn cap requires a known public player turn")
    return int(observation["player"][turn_column])


def _resources(view, observation):
    """Public HP, max HP and held potion kinds of one observation."""
    return view.player_hp_now(observation), view.player_max_hp(observation), Counter(view.potion_kinds(observation))


def _resource_fields(view, start, end):
    hp_start, max_hp_start, held = start
    _, max_hp_end, kept = end
    prices = view.potion_price_table(_REFERENCE_PRICES)
    try:
        spent = (sum(prices[kind] * n for kind, n in (held - kept).items())
                 - sum(prices[kind] * n for kind, n in (kept - held).items()))
    except KeyError as error:
        raise ValueError(f"potion kind {error.args[0]} has no price") from None
    return {"hp_start": int(hp_start), "max_hp_start": int(max_hp_start), "max_hp_end": int(max_hp_end),
            "potions_start": sum(held.values()), "potions_end": sum(kept.values()),
            "potion_hp_spent": float(spent)}


def play_fight(env, case, view, turn_column, teacher, *, research_seed,
               max_steps=1024, max_turn_actions=500, chooser=None, observer=None, finisher=None):
    """Play one fight with the teacher; the caps, censors and resource fields live only here.

    Optional hooks (the defaults play the teacher's move and change nothing):

    - ``chooser(observation, legal, teacher_result, rng) -> action`` picks the move to play;
      ``teacher_result`` is ``_decision``'s tuple ``(root, choice, values, cpu_seconds,
      refusals, simulations)`` and ``rng`` the fight's teacher Random. The move must be legal.
      A decision row's ``choice`` is the move actually played; the distillation label entry keeps the
      teacher's choice (from ``teacher_result``) separately from the played move.
    - ``observer(observation, decision, turn)`` sees each decision row with the decision's own
      observation and public turn, before the move is stepped.
    - ``finisher(final_observation, info, outcome, censor_reason)`` sees the fight's end state
      (the observation potion kinds, max HP and cleanup HP are read from), once, before return.
    """
    wall_start, cpu_start = perf_counter(), process_time()
    observation, info = env.reset(case)
    # Read the reset resources before the first step.
    start = _resources(view, observation)
    rng = random.Random(research_seed)
    decisions = []
    turn_actions = 0
    censor_reason = None
    reward = None
    while info.get("phase") != "terminal":
        step = len(decisions)
        if step == max_steps:
            censor_reason = "step_cap"
            break
        turn = _public_turn(observation, turn_column)
        legal = view.legal(observation)
        if not legal:
            raise RuntimeError(f"Live fight has no legal action at step {step}")
        seed = (research_seed * max_steps + step) % (1 << 64)
        teacher_result = _decision(env, observation, legal, view, teacher, rng, seed)
        root, action, values, cpu_seconds, refusals, simulations = teacher_result
        if chooser is not None:
            action = chooser(observation, legal, teacher_result, rng)
            if action not in legal:
                raise RuntimeError(f"Chooser played an action outside the legal set at step {step}")
        decision = {
            "case_id": case["case_id"], "step": step, "action_prefix_length": step,
            "root": root, "values": values, "choice": action, "cpu_seconds": cpu_seconds,
            "research_seed": seed, "branch_refusals": refusals, "simulations": simulations,
        }
        if observer is not None:
            observer(observation, decision, turn)
        observation, reward, done, truncated, info = env.step(action)
        decisions.append(decision)
        if truncated:
            raise RuntimeError("Unexpected native Standalone combat truncation")
        if done:
            break
        turn_actions = turn_actions + 1 if _public_turn(observation, turn_column) == turn else 0
        if turn_actions == max_turn_actions:
            censor_reason = "turn_cap"
            break

    outcome = "censored" if censor_reason else info.get("terminal_reason")
    if censor_reason is None and (
        outcome not in TERMINAL_REWARDS or reward is not None and reward != TERMINAL_REWARDS[outcome]
    ):
        raise RuntimeError("Terminal fight has an invalid public outcome or reward")
    if finisher is not None:
        finisher(observation, info, outcome, censor_reason)
    hp = info.get("hp_after_cleanup") if censor_reason is None else None
    return {
        "case_id": case["case_id"], "outcome": outcome, "censor_reason": censor_reason,
        "steps": len(decisions), "hp_after_cleanup": None if hp is None else int(hp),
        **_resource_fields(view, start, _resources(view, observation)),
        "cpu_seconds": process_time() - cpu_start, "wall_seconds": perf_counter() - wall_start,
    }, decisions


def _run_case(job):
    case, split, teacher, seed, max_steps, max_turn_actions = job
    with _fight_env(case, split) as (env, descriptor):
        player = next(tensor for tensor in descriptor["tensors"] if tensor["key"] == "player")
        turn_column = [column["field"] for column in player["columns"]].index("turn")
        return play_fight(env, case, View(descriptor), turn_column, teacher, research_seed=seed,
                          max_steps=max_steps, max_turn_actions=max_turn_actions)


def _results(jobs, parallel):
    if parallel == 1:
        yield from map(_run_case, jobs)
        return
    pool = ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("spawn"))
    try:
        yield from pool.map(_run_case, jobs)
    finally:
        pool.shutdown(cancel_futures=True)


def _value_fields(fields):
    """Parse optional value_rule and playout_policy; the v1 defaults are dropped so v1 identities hold."""
    parsed = {}
    if "value_rule" in fields:
        if not isinstance(fields["value_rule"], dict):
            raise ValueError("value_rule must be a value rule identity object")
        rule = ValueRule.from_identity(fields["value_rule"])
        if rule != _FLAT:
            parsed["value_rule"] = rule
    if fields.get("playout_policy", "uniform-v1") != "uniform-v1":
        parsed["playout_policy"] = fields["playout_policy"]
    return parsed


def _uncalibrated(name, panel, calibrated, allow_uncalibrated):
    if panel and not calibrated and not allow_uncalibrated:
        raise ValueError(f"{name} is uncalibrated; use --train-sample during calibration "
                         "or --allow-uncalibrated")


def _teacher(path, name, panel, *, allow_uncalibrated=False):
    teachers = json.loads(path.read_text())["teachers"]
    if name not in teachers:
        raise ValueError(f"unknown teacher {name!r}; valid teachers: {', '.join(sorted(teachers))}")
    row = teachers[name]
    if row.get("kind") == "one-move":
        current = {"samples": 32, "playout_steps": 80}
        expected = {"kind": "one-move", **current}
        extra = {key: row[key] for key in ("calibrated", "value_rule", "playout_policy") if key in row}
        backend = row.get("playout_backend", "python")
        if backend not in ("python", "native"):
            raise ValueError(f"{name} playout_backend must be python or native")
        if backend == "native":
            current["playout_backend"] = expected["playout_backend"] = "native"
        elif "playout_backend" in row:
            expected["playout_backend"] = "python"
        if extra:
            expected.update(extra)
        if row != expected:
            raise ValueError(f"{name} must use one-move search with 32 samples and 80 playout steps")
        values = _value_fields(row)
        check_value_options(values.get("value_rule", _FLAT), values.get("playout_policy", "uniform-v1"), backend)
        current.update(values)
        calibrated = row.get("calibrated", True) is True
        _uncalibrated(name, panel, calibrated, allow_uncalibrated)
        identity = {key: value for key, value in row.items()
                    if key not in ("calibrated", "value_rule", "playout_policy")}
        if backend == "python":
            identity.pop("playout_backend", None)
        identity.update({"branch_layout": "compact", "workers": 1})
        if "playout_backend" in current:
            identity["playout_rule_id"] = playout_rule_id(current.get("playout_policy", "uniform-v1"))
        identity.update(_json_values(values))
        return current, content_sha256(identity), calibrated
    calibrated = row["calibrated"] is True
    _uncalibrated(name, panel, calibrated, allow_uncalibrated)
    config = {key: value for key, value in row["config"].items() if key not in ("value_rule", "playout_policy")}
    config.update(_value_fields(row["config"]))
    if config.get("widening") is not None:
        config = {**config, "widening": tuple(config["widening"])}
    teacher = TeacherConfig(**config)
    if teacher.branch_layout != "compact" or teacher.workers != 1:
        raise ValueError("Tree teachers require compact search with one worker")
    return teacher, config_sha256(teacher), calibrated


def _json_values(fields):
    """JSON-safe value fields: a value rule as its identity object."""
    return {key: value.as_identity() if isinstance(value, ValueRule) else value for key, value in fields.items()}


def _teacher_config(teacher):
    return config_identity(teacher) if isinstance(teacher, TeacherConfig) else _json_values(teacher)


def _native_backend(teacher):
    backend = teacher.playout_backend if isinstance(teacher, TeacherConfig) else teacher.get("playout_backend")
    return backend == "native"


def _run(args, output):
    # Callers that build args without the flag keep today's calibration guard.
    allow_uncalibrated = getattr(args, "allow_uncalibrated", False) is True
    teacher, teacher_sha, calibrated = _teacher(args.teachers, args.teacher, args.panel,
                                                allow_uncalibrated=allow_uncalibrated)

    from spire_lab import _native
    recipe = load_recipe(args.recipe)
    gym_build = dict(_native.build_info())
    if not isinstance(recipe, StandardRecipe):
        raise ValueError("Teacher entry requires a Standard recipe")
    try:
        contract = contract_for_revision(recipe.gym_revision)
    except ValueError as error:
        raise ValueError(f"Teacher entry requires a supported Gym revision: {error}") from error
    revision = contract.source_revision
    # The live Gym may be the table's revision or a drop-in in its rules epoch; its own contract
    # decides the compact view, and its revision and build are what the provenance records.
    live_revision = gym_build.get("revision")
    if not accepts_recorded_revision(live_revision, revision):
        raise ValueError(f"Teacher entry requires a Standard recipe and Gym revision {revision} "
                         "(or one in its rules epoch) with compact views")
    live = contract_for_revision(live_revision)
    if STANDARD_COMPACT_PRESET not in live.standard_compact_fingerprints:
        raise ValueError(f"Teacher entry requires a Standard recipe and Gym revision {revision} "
                         "(or one in its rules epoch) with compact views")

    if args.panel:
        split = "selection"
        panel = build_panel(recipe, split, kind=args.panel)
        cases = panel["manifest"]["cases"]
        identity = {key: panel[key] for key in ("panel_identity", "panel_sha256", "manifest_sha256")}
    else:
        split = "train"
        cases = sample_train_cases(recipe.states_for(split), args.train_sample, args.seed)
        identity = {"train_sample": args.train_sample, "level": 0, "sample_seed": args.seed}
    if args.limit:
        cases = cases[::max(1, len(cases) // args.limit)][:args.limit]
    if not cases:
        raise ValueError("No fights selected")
    _, playout_policy = _value_options(teacher)
    if _native_backend(teacher) and playout_policy != "uniform-v1":
        # Native v2 playouts need the live Gym's tiered rule; refuse before any preflight or output.
        require_tiered_playouts(live)
    descriptor = preflight(cases, split)

    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / filename for filename in ("fights.jsonl", "decisions.jsonl", "summary.json")]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Teacher output files already exist")
    jobs = [(case, split, teacher, (args.seed + index) % (1 << 64),
             args.max_steps, args.max_turn_actions) for index, case in enumerate(cases)]
    outcomes, roots = Counter(), Counter()
    search_cpu = fight_cpu = refusals = 0
    wall_start = perf_counter()
    with destinations[0].open("x", encoding="utf-8") as fights, destinations[1].open("x", encoding="utf-8") as decisions:
        for fight, rows in _results(jobs, args.parallel):
            outcomes[fight["outcome"]] += 1
            fight_cpu += fight["cpu_seconds"]
            fights.write(json.dumps(fight, sort_keys=True, allow_nan=False) + "\n")
            for row in rows:
                roots[row["root"]] += 1
                if row["root"] == "searched":
                    search_cpu += row["cpu_seconds"]
                refusals += row["branch_refusals"]
                decisions.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    searched = roots["searched"]
    summary = {
        "schema": "spire-learn-search-teacher-entry-v1", "cases": len(cases), "wins": outcomes["victory"],
        "outcomes": dict(outcomes), "roots": dict(roots), "branch_refusals": refusals,
        "searched_decisions": searched, "cpu_seconds_per_searched_decision": search_cpu / searched if searched else None,
        "fight_cpu_seconds": fight_cpu, "fight_run_wall_seconds": perf_counter() - wall_start,
        "provenance": {
            "gym_revision": live_revision, "gym_build": gym_build,
            "recipe_sha256": sha256(args.recipe.read_bytes()).hexdigest(), "states_table_sha256": recipe.table_sha256,
            "panel": args.panel, "split": split, **identity, "case_manifest_sha256": content_sha256(cases),
            "teacher": args.teacher, "calibrated": calibrated, "config_sha256": teacher_sha,
            "teacher_config": _teacher_config(teacher),
            **({"uncalibrated_allowed": True} if allow_uncalibrated else {}),
            "seed": args.seed, "environment_seeds": [case["environment_seed"] for case in cases],
            "research_seed_rule": "((seed + case_index) * max_steps + step) modulo 2**64",
            "character": recipe.character, "ascension": recipe.ascension, "profile": descriptor["profile_id"],
            "observation_schema": descriptor["schema_id"], "observation_fingerprint": descriptor["fingerprint"],
            "rules_epoch": live.rules_epoch, "capacity_preset": STANDARD_COMPACT_PRESET,
            "reward": TERMINAL_REWARDS, "branch_layout": "compact",
            **({"playout_backend": "native", "playout_rule_id": playout_rule_id(playout_policy)}
               if _native_backend(teacher) else {}),
            "workers": 1, "parallel": args.parallel, "max_steps": args.max_steps,
            "max_turn_actions": args.max_turn_actions, "gpu_seconds": 0,
        },
    }
    with destinations[2].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--panel", choices=("selection", "boss"))
    scope.add_argument("--train-sample", type=_positive, metavar="N")
    parser.add_argument("--teacher", required=True, help="a teacher name in --teachers")
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=_positive, default=1024)
    parser.add_argument("--max-turn-actions", type=_positive, default=500)
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--parallel", type=_positive, default=1)
    parser.add_argument("--limit", type=_positive)
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="let a teacher marked uncalibrated run on --panel (recorded in provenance)")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if not output.is_relative_to((_ROOT / "artifacts").resolve()):
        parser.error("--output must resolve under artifacts/")
    expected = os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT")
    if expected and not output.is_relative_to(Path(expected).resolve()):
        parser.error("--output must be inside SPIRE_LEARN_RUNNER_OUTPUT")
    if not 0 <= args.seed < 1 << 64:
        parser.error("--seed must be an unsigned 64-bit integer")
    args.recipe = _ROOT / args.recipe
    args.teachers = _ROOT / args.teachers
    print(json.dumps(_run(args, output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
