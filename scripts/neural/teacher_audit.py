#!/usr/bin/env python3
"""Search-teacher audit: self-consistency replays, strength against budget and gate G1.

consistency
  Samples searched decisions from one finished selection-split teacher run
  (search_teacher_entry.py output: decisions.jsonl with summary.json beside it),
  deterministically from --seed. The audit set is never training data. Each
  sampled decision's state is rebuilt by resetting its case exactly as the entry
  did (panel case, case environment seed) and replaying that case's recorded
  choices in step order. The same teacher then searches it again with
  research_seed + 1 (run B; the recorded search is run A). A replay whose legal
  set, terminal state or root differs from the record is an error. Reported:
  top-choice agreement, and mean value lost both ways: run A's value of its own
  choice minus run A's value of run B's choice, and the same from run B's side.
  Needs the native Gym.

report
  Aggregates finished teacher runs (stdlib only, no Gym). Inputs are explicit:
    --run TEACHER:PANEL:DIR      a search_teacher_entry.py output directory
                                 (summary.json, fights.jsonl, decisions.jsonl);
                                 TEACHER in current, tree-1x, tree-3x, tree-10x;
                                 PANEL in selection, boss. Repeat per run.
    --panel PANEL:FILE           the panel document of each panel used
                                 (development-panel.json or boss-panel.json):
                                 act, room kind, level and starting max HP per case.
    --ppo NAME:PANEL:SUMMARY     optional PPO baseline summary.json, final
                                 evaluation, shown beside the teachers.
    --run-states FILE            optional run-states table; adds Baalorlord's
                                 recorded level-0 outcomes per panel.
  Selection runs of current, tree-1x and tree-3x are required. Every run must
  name the panel document's panel and manifest SHA256 and share one Gym
  revision, rules epoch, recipe, states table and caps.
  It reports strength against budget per teacher and panel (wins, mean HP kept,
  CPU seconds per searched decision); root outcomes by reason, act and room kind;
  label coverage; G1 (tree-1x and tree-3x each against current on the selection
  panel, exact McNemar, a case censored or escaped on either side dropped; a
  teacher beats current with more paired wins and p < 0.025; the boss panel is
  shown beside and not used); the teacher choice (most selection wins among
  current, tree-1x and tree-3x, ties to the lower measured selection CPU seconds
  per searched decision; tree-10x is on the strength curve only); and the stop
  rule (boss_root_share < 0.5 over the chosen teacher's boss-fight decisions on
  every panel given gives STOP). Root shares exclude single-legal decisions,
  which never open a root; labelled shares count every decision.
  HP kept is hp_after_cleanup over the case's starting max HP (panel entry), over
  completed fights (wins, deaths and escapes); PPO's figure is its summary's
  mean_hp_retention, over the same fights with end-of-fight max HP.

Both write aggregates only (JSON, to --output under artifacts/, never
overwritten); report also prints Markdown tables. Case ids never appear.
Simulator faults are errors: a run without summary.json or with an unknown
outcome is refused, never counted as a loss or censor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from scripts.neural import search_teacher_entry as entry
from scripts.neural.standard_results import KIND_ORDER, mcnemar_exact, paired, recorded_wins, root_table

ENTRY_SCHEMA = "spire-learn-search-teacher-entry-v1"
CONSISTENCY_SCHEMA = "spire-learn-teacher-consistency-v1"
REPORT_SCHEMA = "spire-learn-teacher-audit-report-v1"
TEACHERS = ("current", "tree-1x", "tree-3x", "tree-10x")
# Dataset-teacher candidates: at most 3x the current search's cost, cheapest budget first.
CANDIDATES = ("current", "tree-1x", "tree-3x")
CHALLENGERS = ("tree-1x", "tree-3x")
PANELS = ("selection", "boss")
G1_ALPHA = 0.025
STOP_BOSS_ROOT_SHARE = 0.5
# Entry outcomes in standard_results' paired vocabulary; escaped and incomplete leave a pair.
PAIRED_OUTCOME = {"victory": "win", "death": "loss", "escaped": "escaped", "censored": "incomplete"}
PPO_OUTCOMES = ("win", "loss", "escaped", "incomplete")
SHARED_PROVENANCE = ("gym_revision", "rules_epoch", "recipe_sha256", "states_table_sha256",
                     "observation_fingerprint", "max_steps", "max_turn_actions")


class AuditInputError(ValueError):
    """Audit inputs are incomplete or do not describe one panel, Gym and teacher."""


class ReplayMismatch(RuntimeError):
    """A replayed fight diverged from its recorded run."""


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _load_summary(path, what):
    path = Path(path)
    if not path.is_file():
        raise AuditInputError(f"{what} is incomplete: no summary.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema") != ENTRY_SCHEMA:
        raise AuditInputError(f"{what} summary is not a {ENTRY_SCHEMA} output")
    return summary


def case_decisions(rows):
    """Decision rows by case, in step order. Steps must run 0..n-1, each with a prefix of its step."""
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    for items in by_case.values():
        items.sort(key=lambda row: row["step"])
        for index, row in enumerate(items):
            if row["step"] != index or row["action_prefix_length"] != index:
                raise AuditInputError("A case's decision rows are not a contiguous step sequence from 0")
    return dict(by_case)


# --- consistency -------------------------------------------------------------------------------

def sample_decisions(summary, rows, count, seed):
    """Searched decisions of a selection-split run, sampled reproducibly from seed, in (case, step) order."""
    if summary.get("schema") != ENTRY_SCHEMA:
        raise AuditInputError(f"Decisions must come from a {ENTRY_SCHEMA} run")
    if (summary.get("provenance") or {}).get("split") != "selection":
        raise AuditInputError("The audit set must come from a selection-split run; it is never training data")
    pool = sorted((row for row in rows if row["root"] == "searched"),
                  key=lambda row: (row["case_id"], row["step"]))
    if len(pool) < count:
        raise AuditInputError(f"Only {len(pool)} searched decisions; cannot sample {count}")
    return sorted(random.Random(seed).sample(pool, count), key=lambda row: (row["case_id"], row["step"]))


def _values(values):
    return {int(action): float(stat[0]) for action, stat in values.items()}


def compare(recorded, choice_b, values_b):
    """Agreement and value lost between the recorded search (A) and its rerun (B)."""
    a, b = _values(recorded["values"]), _values(values_b)
    if a.keys() != b.keys():
        raise ReplayMismatch("Replayed search scored a different legal set than the recorded one")
    choice_a = recorded["choice"]
    return {"agree": choice_a == choice_b,
            "lost_a": a[choice_a] - a[choice_b],
            "lost_b": b[choice_b] - b[choice_a]}


def _check_legal(row, legal):
    legal = sorted(legal)
    root, choice = row["root"], row["choice"]
    if root == "single_legal":
        matches = legal == [choice]
    elif root == "searched":
        matches = sorted(int(action) for action in row["values"]) == legal
    else:
        matches = len(legal) > 1 and choice in legal
    if not matches:
        raise ReplayMismatch(f"Replay legal set differs from the recorded decision at step {row['step']}")


def audit_case(env, case, view, teacher, rows, sampled_steps):
    """Replay one case's recorded choices; rerun the teacher at each sampled step with research_seed + 1."""
    sampled = set(sampled_steps)
    if not sampled or min(sampled) < 0 or max(sampled) >= len(rows):
        raise AuditInputError("Sampled steps lie outside the case's recorded decisions")
    observation, info = env.reset(case)
    results = []
    for row in rows[:max(sampled) + 1]:
        step = row["step"]
        if info.get("phase") == "terminal":
            raise ReplayMismatch(f"Replay reached a terminal state before recorded step {step}")
        legal = view.legal(observation)
        _check_legal(row, legal)
        if step in sampled:
            if row["root"] != "searched":
                raise AuditInputError("Only searched decisions can be audited")
            seed = (row["research_seed"] + 1) % (1 << 64)
            root, choice, values, cpu_seconds, refusals, simulations = entry._decision(
                env, observation, legal, view, teacher, random.Random(seed), seed)
            if root != "searched":
                raise ReplayMismatch(f"Replayed decision at step {step} got no search root ({root})")
            results.append({**compare(row, choice, values), "cpu_seconds": cpu_seconds,
                            "simulations": simulations, "branch_refusals": refusals})
        observation, _, done, truncated, info = env.step(row["choice"])
        if truncated:
            raise ReplayMismatch("Unexpected truncation during replay")
        if done and step != len(rows) - 1:
            raise ReplayMismatch(f"Replay ended at step {step}, before the recorded fight did")
    return results


def summarize_consistency(results):
    n = len(results)
    if not n:
        raise AuditInputError("No audited decisions")
    agreements = sum(result["agree"] for result in results)
    disagreeing = [result for result in results if not result["agree"]]
    return {
        "decisions": n, "agreements": agreements, "top_choice_agreement": agreements / n,
        "mean_value_lost_run_a": sum(result["lost_a"] for result in results) / n,
        "mean_value_lost_run_b": sum(result["lost_b"] for result in results) / n,
        "mean_value_lost_run_a_when_disagreeing": (
            sum(result["lost_a"] for result in disagreeing) / len(disagreeing) if disagreeing else None),
        "mean_value_lost_run_b_when_disagreeing": (
            sum(result["lost_b"] for result in disagreeing) / len(disagreeing) if disagreeing else None),
        "rerun_cpu_seconds_per_decision": sum(result["cpu_seconds"] for result in results) / n,
        "rerun_branch_refusals": sum(result["branch_refusals"] for result in results),
    }


def _audit_job(job):
    case, split, teacher, rows, steps = job
    with entry._fight_env(case, split) as (env, descriptor):
        return audit_case(env, case, entry.View(descriptor), teacher, rows, steps)


def _audit_results(jobs, parallel):
    if parallel == 1:
        yield from map(_audit_job, jobs)
        return
    pool = ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("spawn"))
    try:
        yield from pool.map(_audit_job, jobs)
    finally:
        pool.shutdown(cancel_futures=True)


def run_consistency(args, output):
    summary_path = args.decisions.parent / "summary.json"
    summary = _load_summary(summary_path, "Teacher run")
    provenance = summary["provenance"]
    if provenance.get("teacher") != args.teacher:
        raise AuditInputError(f"Decisions are from teacher {provenance.get('teacher')}, not {args.teacher}")
    rows = _jsonl(args.decisions)
    by_case = case_decisions(rows)
    sample = sample_decisions(summary, rows, args.sample, args.seed)

    teacher, teacher_sha, _ = entry._teacher(args.teachers, args.teacher, provenance["panel"])
    if teacher_sha != provenance["config_sha256"]:
        raise AuditInputError("Teacher config differs from the one that produced the decisions")
    if _file_sha256(args.recipe) != provenance["recipe_sha256"]:
        raise AuditInputError("Recipe differs from the one that produced the decisions")
    from spire_lab import _native
    gym_build = dict(_native.build_info())
    recipe = entry.load_recipe(args.recipe)
    if not (gym_build.get("revision") == recipe.gym_revision == provenance["gym_revision"]):
        raise AuditInputError("Native Gym, recipe and decisions name different Gym revisions")
    panel = entry.build_panel(recipe, "selection", kind=provenance["panel"])
    cases = panel["manifest"]["cases"]
    if entry.content_sha256(cases) != provenance["case_manifest_sha256"]:
        raise AuditInputError("Rebuilt panel cases differ from the run's case manifest")
    by_id = {case["case_id"]: case for case in cases}

    steps = defaultdict(list)
    for row in sample:
        steps[row["case_id"]].append(row["step"])
    jobs = [(by_id[case_id], "selection", teacher, by_case[case_id], tuple(steps[case_id]))
            for case_id in sorted(steps)]
    entry.preflight([job[0] for job in jobs], "selection")
    results = [result for case_results in _audit_results(jobs, args.parallel) for result in case_results]
    if len(results) != len(sample):
        raise RuntimeError("Audit returned a different number of decisions than were sampled")

    report = {
        "schema": CONSISTENCY_SCHEMA, **summarize_consistency(results),
        "provenance": {
            "teacher": args.teacher, "config_sha256": teacher_sha, "sample": args.sample, "sample_seed": args.seed,
            "sample_rule": "random.Random(seed).sample of searched rows sorted by (case_id, step)",
            "sample_sha256": entry.content_sha256([[row["case_id"], row["step"]] for row in sample]),
            "rerun_research_seed_rule": "recorded research_seed + 1 modulo 2**64",
            "split": "selection", "panel": provenance["panel"], "audit_only": True,
            "source_summary_sha256": _file_sha256(summary_path),
            "source_decisions_sha256": _file_sha256(args.decisions),
            "source_seed": provenance.get("seed"), "gym_build": gym_build,
            **{key: provenance.get(key) for key in SHARED_PROVENANCE},
            "parallel": args.parallel, "workers": 1, "gpu_seconds": 0,
        },
    }
    _write_json(output, report)
    return report


# --- report ------------------------------------------------------------------------------------

def beats(only_challenger, only_current, alpha=G1_ALPHA):
    """More paired wins on the challenger's side and an exact McNemar p below alpha."""
    return only_challenger > only_current and mcnemar_exact(only_challenger, only_current) < alpha


def g1_test(challenger, current, alpha=G1_ALPHA):
    """Exact McNemar of two teachers' entry outcomes on one panel; censored or escaped cases leave the pair."""
    result = paired({cid: PAIRED_OUTCOME[o] for cid, o in challenger.items()},
                    {cid: PAIRED_OUTCOME[o] for cid, o in current.items()})
    discordant = result["only_a"] - result["only_b"]
    return {
        **result, "alpha": alpha,
        "wins_challenger": result["both_win"] + result["only_a"],
        "wins_current": result["both_win"] + result["only_b"],
        "effect_points": 100 * discordant / result["paired"] if result["paired"] else None,
        "beats": beats(result["only_a"], result["only_b"], alpha),
    }


def choose_teacher(strength):
    """Most selection-panel wins among the candidates; ties go to the lower measured CPU per searched decision."""
    candidates = {}
    for name in CANDIDATES:
        if name not in strength:
            raise AuditInputError(f"Teacher choice needs the {name} selection run")
        cpu = strength[name]["cpu_seconds_per_searched_decision"]
        if cpu is None:
            raise AuditInputError(f"{name} has no searched decisions to cost")
        candidates[name] = {"wins": strength[name]["wins"], "cpu_seconds_per_searched_decision": cpu}
    chosen = min(candidates, key=lambda name: (-candidates[name]["wins"],
                                               candidates[name]["cpu_seconds_per_searched_decision"],
                                               CANDIDATES.index(name)))
    return {"chosen": chosen, "candidates": candidates,
            "rule": "most selection-panel wins among current, tree-1x and tree-3x; ties to the lower "
                    "selection-panel CPU seconds per searched decision, then the smaller budget"}


def stop_rule(boss_roots):
    """STOP when fewer than half the boss-fight decisions that need a choice get a search root."""
    decisions = sum(count for root, count in boss_roots.items() if root != "single_legal")
    if not decisions:
        raise AuditInputError("Stop rule needs boss-fight decisions with more than one legal action")
    share = boss_roots.get("searched", 0) / decisions
    return {"searched": boss_roots.get("searched", 0), "decisions": decisions, "boss_root_share": share,
            "threshold": STOP_BOSS_ROOT_SHARE, "verdict": "STOP" if share < STOP_BOSS_ROOT_SHARE else "CONTINUE"}


def load_panel(path, kind):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = doc["manifest"]
    if manifest.get("split") != "selection":
        raise AuditInputError(f"{kind} panel document is not a selection-split panel")
    annotations = doc["annotations"]
    cases = {}
    for case in manifest["cases"]:
        cid = case["case_id"]
        if cid in cases or cid not in annotations:
            raise AuditInputError(f"{kind} panel document has a duplicate or unannotated case")
        note = annotations[cid]
        if note["room_kind"] not in KIND_ORDER:
            raise AuditInputError(f"{kind} panel document has room kind {note['room_kind']!r}")
        cases[cid] = {"act": int(note["act"]), "room_kind": note["room_kind"], "level": int(note["level"]),
                      "max_hp": int(case["entry"]["max_hp"]), "run_id": str(note["run_id"]),
                      "floor": int(note["floor"])}
    return {"kind": kind, "panel_sha256": doc["panel_sha256"], "manifest_sha256": doc["manifest_sha256"],
            "states_table_sha256": doc["panel_identity"]["states_table"]["sha256"], "cases": cases,
            "document_sha256": _file_sha256(path)}


def load_run(directory, teacher, kind, panel):
    what = f"{teacher} {kind} run"
    summary_path = Path(directory) / "summary.json"
    summary = _load_summary(summary_path, what)
    provenance = summary["provenance"]
    expected = {"teacher": teacher, "panel": kind, "split": "selection", "calibrated": True,
                "panel_sha256": panel["panel_sha256"], "manifest_sha256": panel["manifest_sha256"]}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise AuditInputError(f"{what} provenance {key} is {provenance.get(key)!r}, expected {value!r}")

    fights = {}
    for fight in _jsonl(Path(directory) / "fights.jsonl"):
        outcome = fight["outcome"]
        if outcome not in PAIRED_OUTCOME:
            raise AuditInputError(f"{what} has a fight with unknown outcome {outcome!r}")
        if (outcome == "censored") != (fight.get("censor_reason") is not None):
            raise AuditInputError(f"{what} has a censor reason that does not match its outcome")
        if outcome != "censored" and type(fight.get("hp_after_cleanup")) is not int:
            raise AuditInputError(f"{what} has a completed fight without hp_after_cleanup")
        if fight["case_id"] in fights:
            raise AuditInputError(f"{what} has a case played twice")
        fights[fight["case_id"]] = fight
    if fights.keys() != panel["cases"].keys():
        raise AuditInputError(f"{what} cases differ from the {kind} panel document's cases")
    if (len(fights) != summary["cases"]
            or Counter(f["outcome"] for f in fights.values()) != Counter(summary["outcomes"])):
        raise AuditInputError(f"{what} fights do not match its summary")

    rows = _jsonl(Path(directory) / "decisions.jsonl")
    decisions = case_decisions(rows)
    if (decisions.keys() - fights.keys()
            or any(len(decisions.get(cid, ())) != fight["steps"] for cid, fight in fights.items())
            or Counter(row["root"] for row in rows) != Counter(summary["roots"])):
        raise AuditInputError(f"{what} decisions do not match its fights and summary")
    return {"teacher": teacher, "panel": kind, "provenance": provenance, "fights": fights,
            "decisions": decisions, "summary_sha256": _file_sha256(summary_path)}


def load_ppo(path, name, panel):
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    evaluation = summary["evaluations"]["final"]
    meta = evaluation["summary"]
    if (meta.get("panel_sha256"), meta.get("panel_manifest_sha256")) != (panel["panel_sha256"], panel["manifest_sha256"]):
        raise AuditInputError(f"{name} final evaluation is not on the {panel['kind']} panel document's panel")
    if meta.get("failures") or meta["total"].get("operational_failures"):
        raise AuditInputError(f"{name} final evaluation has operational failures")
    outcomes = {cid: case["outcome"] for cid, case in evaluation["cases"].items()}
    if outcomes.keys() != panel["cases"].keys() or set(outcomes.values()) - set(PPO_OUTCOMES):
        raise AuditInputError(f"{name} final evaluation cases or outcomes do not fit the panel")
    counts = Counter(outcomes.values())
    level0 = [cid for cid, case in panel["cases"].items() if case["level"] == 0]
    return {"cases": len(outcomes), "wins": counts["win"], "deaths": counts["loss"], "escaped": counts["escaped"],
            "censored": counts["incomplete"], "wins_level0": sum(outcomes[cid] == "win" for cid in level0),
            "mean_hp_kept": meta["total"].get("mean_hp_retention"), "summary_sha256": _file_sha256(path)}


def strength_row(run, panel):
    fights = run["fights"]
    counts = Counter(fight["outcome"] for fight in fights.values())
    completed = [(cid, f) for cid, f in fights.items() if f["outcome"] != "censored"]
    kept = [f["hp_after_cleanup"] / panel["cases"][cid]["max_hp"] for cid, f in completed]
    won = [f["hp_after_cleanup"] / panel["cases"][cid]["max_hp"] for cid, f in completed if f["outcome"] == "victory"]
    rows = [row for items in run["decisions"].values() for row in items]
    searched = [row for row in rows if row["root"] == "searched"]
    return {
        "cases": len(fights), "wins": counts["victory"], "deaths": counts["death"], "escaped": counts["escaped"],
        "censored": counts["censored"],
        "wins_level0": sum(f["outcome"] == "victory" for cid, f in fights.items() if panel["cases"][cid]["level"] == 0),
        "mean_hp_kept": sum(kept) / len(kept) if kept else None,
        "mean_hp_kept_on_wins": sum(won) / len(won) if won else None,
        "decisions": len(rows), "searched_decisions": len(searched),
        "cpu_seconds_per_searched_decision": (
            sum(row["cpu_seconds"] for row in searched) / len(searched) if searched else None),
        "fight_cpu_seconds_per_decision": (
            sum(f["cpu_seconds"] for f in fights.values()) / len(rows) if rows else None),
    }


def root_groups(run, panel):
    groups = defaultdict(Counter)
    for cid, rows in run["decisions"].items():
        case = panel["cases"][cid]
        for row in rows:
            groups[case["act"], case["room_kind"]][row["root"]] += 1
    return groups


def coverage(groups):
    out = {}
    for (act, kind), counts in sorted(groups.items(), key=lambda item: (item[0][0], KIND_ORDER[item[0][1]])):
        total = sum(counts.values())
        attempted = total - counts.get("single_legal", 0)
        out[f"{act}:{kind}"] = {
            "counts": dict(sorted(counts.items())), "decisions": total,
            "root_share": counts.get("searched", 0) / attempted if attempted else None,
            "labelled_share": counts.get("searched", 0) / total if total else None,
        }
    return out


def _shared_provenance(runs):
    shared = {}
    for key in SHARED_PROVENANCE:
        values = {json.dumps(run["provenance"].get(key), sort_keys=True) for run in runs}
        if len(values) != 1:
            raise AuditInputError(f"Teacher runs disagree on {key}")
        shared[key] = runs[0]["provenance"].get(key)
    builds = {json.dumps(run["provenance"].get("gym_build"), sort_keys=True) for run in runs}
    if len(builds) != 1:
        raise AuditInputError("Teacher runs disagree on gym_build")
    shared["gym_build"] = runs[0]["provenance"].get("gym_build")
    return shared


def build_report(run_specs, panel_specs, ppo_specs=(), run_states=None):
    panels = {}
    for kind, path in panel_specs:
        if kind in panels:
            raise AuditInputError(f"Panel {kind} given twice")
        panels[kind] = load_panel(path, kind)
    runs = {}
    for teacher, kind, directory in run_specs:
        if (teacher, kind) in runs:
            raise AuditInputError(f"Run {teacher}:{kind} given twice")
        if kind not in panels:
            raise AuditInputError(f"Run {teacher}:{kind} needs --panel {kind}:FILE")
        runs[teacher, kind] = load_run(directory, teacher, kind, panels[kind])
    missing = [name for name in CANDIDATES if (name, "selection") not in runs]
    if missing:
        raise AuditInputError(f"Selection runs are required for {', '.join(missing)}")
    shared = _shared_provenance(list(runs.values()))

    strength = {kind: {} for kind in panels}
    roots = {kind: {} for kind in panels}
    for (teacher, kind), run in sorted(runs.items(), key=lambda item: (PANELS.index(item[0][1]), TEACHERS.index(item[0][0]))):
        strength[kind][teacher] = strength_row(run, panels[kind])
        roots[kind][teacher] = coverage(root_groups(run, panels[kind]))
    ppo = {}
    for name, kind, path in ppo_specs:
        if kind not in panels:
            raise AuditInputError(f"PPO {name}:{kind} needs --panel {kind}:FILE")
        if name in TEACHERS or name in strength[kind]:
            raise AuditInputError(f"PPO name {name} is a teacher name or given twice on {kind}")
        strength[kind][name] = ppo[name, kind] = load_ppo(path, name, panels[kind])

    outcomes = {key: {cid: f["outcome"] for cid, f in run["fights"].items()} for key, run in runs.items()}
    tests = {name: g1_test(outcomes[name, "selection"], outcomes["current", "selection"]) for name in CHALLENGERS}
    boss = {name: g1_test(outcomes[name, "boss"], outcomes["current", "boss"])
            for name in CHALLENGERS if (name, "boss") in runs and ("current", "boss") in runs}
    choice = choose_teacher(strength["selection"])
    chosen = choice["chosen"]
    boss_roots = Counter()
    panels_used = []
    for (teacher, kind), run in runs.items():
        if teacher == chosen:
            panels_used.append(kind)
            for (_, room_kind), counts in root_groups(run, panels[kind]).items():
                if room_kind == "boss":
                    boss_roots.update(counts)

    report = {
        "schema": REPORT_SCHEMA,
        "strength": strength,
        "roots": roots,
        "g1": {"tests": tests, "verdict": "PASS" if any(t["beats"] for t in tests.values()) else "FAIL",
               "rule": "tree-1x and tree-3x each against current on the selection panel; exact two-sided McNemar; "
                       "a case censored or escaped on either side is dropped; beats = more paired wins and p < 0.025",
               "boss_panel": boss},
        "teacher_choice": choice,
        "stop_rule": {"teacher": chosen, "panels": sorted(panels_used, key=PANELS.index), **stop_rule(boss_roots)},
        "provenance": {
            **shared,
            "panels": {kind: {key: panel[key] for key in ("panel_sha256", "manifest_sha256", "document_sha256")}
                       for kind, panel in panels.items()},
            "runs": {f"{teacher}:{kind}": {"config_sha256": run["provenance"].get("config_sha256"),
                                           "seed": run["provenance"].get("seed"),
                                           "summary_sha256": run["summary_sha256"]}
                     for (teacher, kind), run in runs.items()},
            "ppo": {f"{name}:{kind}": row["summary_sha256"] for (name, kind), row in ppo.items()},
            "hp_kept": "hp_after_cleanup / case starting max HP over completed fights; PPO: summary mean_hp_retention",
            "claim": "simulator panels of reconstructed public-run states; no original-game transfer claim",
        },
    }
    if run_states is not None:
        data = Path(run_states).read_bytes()
        digest = sha256(data).hexdigest()
        if any(panel["states_table_sha256"] != digest for panel in panels.values()):
            raise AuditInputError("Run-states table is not the panels' states table")
        recorded = recorded_wins(json.loads(data)["states"])
        level0 = {}
        for kind, panel in panels.items():
            cases = [case for case in panel["cases"].values() if case["level"] == 0]
            try:
                wins = sum(recorded[case["run_id"], case["floor"]] for case in cases)
            except KeyError:
                raise AuditInputError(f"Run-states table lacks a {kind} panel level-0 fight") from None
            level0[kind] = {"cases": len(cases), "wins": wins}
        report["baalorlord_level0"] = level0
        report["provenance"]["run_states_sha256"] = digest
    return report


def _fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render(report):
    row = lambda cells: "| " + " | ".join(str(cell) for cell in cells) + " |"
    out = []
    for kind, rows in report["strength"].items():
        out += [f"Strength against budget, {kind} panel:",
                row(["Player", "cases", "wins", "deaths", "escaped", "censored", "L0 wins", "HP kept",
                     "HP kept (wins)", "CPU s / searched decision", "fight CPU s / decision"]),
                row(["---"] * 11)]
        for name, r in rows.items():
            out.append(row([name, r["cases"], r["wins"], r["deaths"], r["escaped"], r["censored"],
                            r["wins_level0"], _fmt(r["mean_hp_kept"]), _fmt(r.get("mean_hp_kept_on_wins")),
                            _fmt(r.get("cpu_seconds_per_searched_decision")),
                            _fmt(r.get("fight_cpu_seconds_per_decision"))]))
        if kind in report.get("baalorlord_level0", {}):
            level0 = report["baalorlord_level0"][kind]
            out.append(f"Baalorlord recorded level-0 wins: {level0['wins']} of {level0['cases']}.")
        out.append("")
    for kind, teachers in report["roots"].items():
        for teacher, groups in teachers.items():
            table = {tuple(int(p) if p.isdigit() else p for p in key.split(":")): group["counts"]
                     for key, group in groups.items()}
            out += root_table(f"Root outcomes, {teacher} on the {kind} panel (shares of decisions with a root attempt):",
                              table, "Act / kind")
            out += [f"Label coverage, {teacher} on the {kind} panel (searched share of all decisions):",
                    row(["Act / kind", "decisions", "labelled"]), row(["---"] * 3)]
            out += [row([key.replace(":", " "), group["decisions"], _fmt(group["labelled_share"])])
                    for key, group in groups.items()]
            out.append("")
    g1 = report["g1"]
    out += ["G1 (selection panel, exact McNemar, alpha 0.025 each):",
            row(["Challenger", "paired", "excluded", "only challenger", "only current", "wins challenger",
                 "wins current", "effect (points)", "p", "beats"]), row(["---"] * 10)]
    for section, tests in (("", g1["tests"]), (" (boss panel, not used)", g1["boss_panel"])):
        for name, t in tests.items():
            out.append(row([name + section, t["paired"], t["excluded"], t["only_a"], t["only_b"], t["wins_challenger"],
                            t["wins_current"], _fmt(t["effect_points"], 1), f"{t['p']:.4g}", t["beats"]]))
    choice, stop = report["teacher_choice"], report["stop_rule"]
    out += [f"G1 verdict: {g1['verdict']}.", "",
            f"Teacher choice: {choice['chosen']} ({choice['rule']}).",
            f"Stop rule ({stop['teacher']}, panels {', '.join(stop['panels'])}): boss_root_share "
            f"{stop['boss_root_share']:.4f} ({stop['searched']} of {stop['decisions']}) -> {stop['verdict']}.",
            "With 256 paired cases, differences below about 4-5 percentage points read as no difference."]
    return "\n".join(out)


# --- command line ------------------------------------------------------------------------------

def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _split_spec(text, count, what):
    parts = text.split(":", count - 1)
    if len(parts) != count or not all(parts):
        raise argparse.ArgumentTypeError(f"expected {what}")
    return parts


def _run_spec(text):
    teacher, panel, path = _split_spec(text, 3, "TEACHER:PANEL:DIR")
    if teacher not in TEACHERS or panel not in PANELS:
        raise argparse.ArgumentTypeError(f"TEACHER must be one of {TEACHERS} and PANEL one of {PANELS}")
    return teacher, panel, Path(path)


def _panel_spec(text):
    panel, path = _split_spec(text, 2, "PANEL:FILE")
    if panel not in PANELS:
        raise argparse.ArgumentTypeError(f"PANEL must be one of {PANELS}")
    return panel, Path(path)


def _ppo_spec(text):
    name, panel, path = _split_spec(text, 3, "NAME:PANEL:SUMMARY")
    if panel not in PANELS:
        raise argparse.ArgumentTypeError(f"PANEL must be one of {PANELS}")
    return name, panel, Path(path)


def _output(parser, path):
    output = (_ROOT / path).resolve()
    if not output.is_relative_to((_ROOT / "artifacts").resolve()):
        parser.error("--output must resolve under artifacts/")
    expected = os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT")
    if expected and not output.is_relative_to(Path(expected).resolve()):
        parser.error("--output must be inside SPIRE_LEARN_RUNNER_OUTPUT")
    if output.exists() or output.is_symlink():
        parser.error("--output already exists")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    consistency = commands.add_parser("consistency", help="replay and re-search sampled decisions (native Gym)")
    consistency.add_argument("--decisions", type=Path, required=True,
                             help="decisions.jsonl of a finished selection-split teacher run (summary.json beside it)")
    consistency.add_argument("--sample", type=entry._positive, default=2000)
    consistency.add_argument("--seed", type=int, required=True, help="sample seed")
    consistency.add_argument("--teacher", choices=TEACHERS, required=True)
    consistency.add_argument("--teachers", type=Path, required=True, help="teachers.json the run used")
    consistency.add_argument("--recipe", type=Path, required=True, help="recipe the run used")
    consistency.add_argument("--parallel", type=entry._positive, default=1)
    consistency.add_argument("--output", type=Path, required=True, help="aggregate JSON under artifacts/")

    report = commands.add_parser("report", help="aggregate teacher runs: strength, roots, G1, choice, stop rule",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    report.add_argument("--run", action="append", type=_run_spec, required=True, metavar="TEACHER:PANEL:DIR")
    report.add_argument("--panel", action="append", type=_panel_spec, required=True, metavar="PANEL:FILE")
    report.add_argument("--ppo", action="append", type=_ppo_spec, default=[], metavar="NAME:PANEL:SUMMARY")
    report.add_argument("--run-states", type=Path, metavar="FILE")
    report.add_argument("--output", type=Path, required=True, help="aggregate JSON under artifacts/")
    args = parser.parse_args(argv)

    output = _output(parser, args.output)
    if args.command == "consistency":
        if not 0 <= args.seed < 1 << 64:
            parser.error("--seed must be an unsigned 64-bit integer")
        args.decisions, args.teachers, args.recipe = (_ROOT / path for path in (args.decisions, args.teachers, args.recipe))
        print(json.dumps(run_consistency(args, output), sort_keys=True))
        return 0
    result = build_report(
        [(teacher, panel, _ROOT / path) for teacher, panel, path in args.run],
        [(panel, _ROOT / path) for panel, path in args.panel],
        [(name, panel, _ROOT / path) for name, panel, path in args.ppo],
        None if args.run_states is None else _ROOT / args.run_states,
    )
    _write_json(output, result)
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
