#!/usr/bin/env python3
"""Aggregate Standard-baseline panel results (stdlib only).

Joins per-case panel outcomes from two baseline replicates, the search
reference and the recorded fight outcomes of the reconstructed run-states
table, and prints aggregate Markdown tables.  Only aggregates are printed:
run ids, floors and case ids never appear in the output.

Usage:
  python3 scripts/neural/standard_results.py --run1 SUMMARY --run2 SUMMARY \
      --panel standard_panel.jsonl --run-states run-states.json \
      [--search standard_panel.json] [--recipe recipe.json]

Before aggregating it checks that every evaluation in both summaries and the search
output (``standard_panel.json``, default: beside the jsonl) name one panel manifest
SHA256, that the run-states file's SHA256 is the recipe's pin, and that every panel case
is a monster, elite or boss fight; any mismatch exits with an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from math import comb
from pathlib import Path

LABELS = ("fresh-initial", "near-405504", "near-811008", "near-1216512", "final")
KIND_ORDER = {"monster": 0, "elite": 1, "boss": 2}
DEFAULT_RECIPE = Path(__file__).resolve().parents[2] / "configs/neural/baalorlord-standard-v1/recipe.json"


class InputMismatch(SystemExit):
    """The inputs do not describe one panel and one states table."""


def check_inputs(r1: dict, r2: dict, search: dict, panel: list[dict],
                 run_states_sha256: str, recipe: dict) -> str:
    """The one panel manifest SHA256 all inputs agree on; raises InputMismatch otherwise."""
    pin = (recipe.get("states_table") or {}).get("sha256")
    if run_states_sha256 != pin:
        raise InputMismatch(f"run-states SHA256 {run_states_sha256} is not the recipe pin {pin}")
    seen: dict[str, set[str]] = defaultdict(set)
    for name, summary in (("run1", r1), ("run2", r2)):
        for label, evaluation in sorted(summary["evaluations"].items()):
            sha = (evaluation.get("summary") or {}).get("panel_manifest_sha256")
            seen[str(sha)].add(f"{name}:{label}")
    seen[str(search.get("panel_manifest_sha256"))].add("search")
    if len(seen) != 1:
        detail = "; ".join(f"{sha}: {', '.join(sorted(where))}" for sha, where in sorted(seen.items()))
        raise InputMismatch(f"inputs name different panel manifests: {detail}")
    (sha,) = seen
    if sha == "None":
        raise InputMismatch("inputs do not record a panel manifest SHA256")
    kinds = {c["room_kind"] for c in panel} - set(KIND_ORDER)
    if kinds:
        raise InputMismatch(f"panel holds room kinds {sorted(kinds)}; panel strata are monster/elite/boss only")
    return sha


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2.0 * tail / 2**n)


def load_outcomes(summary: dict, label: str) -> dict[str, str]:
    """Map case id to win / loss / escaped / incomplete for one checkpoint."""
    cases = summary["evaluations"][label]["cases"]
    return {cid: str(entry["outcome"]) for cid, entry in cases.items()}


def curve_row(outcomes: dict[str, str]) -> dict[str, int]:
    row = {"win": 0, "loss": 0, "escaped": 0, "incomplete": 0}
    for outcome in outcomes.values():
        row[outcome] = row.get(outcome, 0) + 1
    return row


def paired(a: dict[str, str], b: dict[str, str]) -> dict[str, int | float]:
    """Exact McNemar on cases that are a win or a loss in both outcome maps."""
    both = only_a = only_b = neither = excluded = 0
    for cid in sorted(set(a) & set(b)):
        x, y = a[cid], b[cid]
        if x not in ("win", "loss") or y not in ("win", "loss"):
            excluded += 1
            continue
        if x == "win" and y == "win":
            both += 1
        elif x == "win":
            only_a += 1
        elif y == "win":
            only_b += 1
        else:
            neither += 1
    return {
        "paired": both + only_a + only_b + neither,
        "both_win": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "excluded": excluded,
        "p": mcnemar_exact(only_a, only_b),
    }


def recorded_wins(states: list[dict]) -> dict[tuple[str, int], bool]:
    """(run_id, floor) -> recorded fight outcome for Ironclad states."""
    return {
        (str(s["run_id"]), int(s["floor"])): bool(s["won"])
        for s in states
        if s.get("character") == "ironclad"
    }


def stratify(panel: list[dict], runs: dict[str, dict[str, str]],
             recorded: dict[tuple[str, int], bool], key) -> list[dict]:
    """Group panel cases by key(case); count wins per source."""
    groups: dict[object, dict] = defaultdict(lambda: {
        "n": 0, "search": 0, "his_n": 0, "his": 0, **{name: 0 for name in runs}})
    for case in panel:
        group = groups[key(case)]
        group["n"] += 1
        if case["search"]["outcome"] == "victory":
            group["search"] += 1
        for name, outcomes in runs.items():
            if outcomes.get(case["case_id"]) == "win":
                group[name] += 1
        if case["level"] == 0:
            group["his_n"] += 1
            group["his"] += int(recorded[(str(case["run_id"]), int(case["floor"]))])
    return [{"key": k, **v} for k, v in sorted(
        groups.items(), key=lambda item: item[0])]


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render(r1: dict, r2: dict, panel: list[dict], states: list[dict]) -> str:
    out: list[str] = []
    out.append("Learning curves (wins / escaped / censored of 256):")
    out.append(_row(["Checkpoint", "r1 wins", "r1 esc", "r1 cens", "r2 wins", "r2 esc", "r2 cens"]))
    out.append(_row(["---"] * 7))
    for label in LABELS:
        a = curve_row(load_outcomes(r1, label))
        b = curve_row(load_outcomes(r2, label))
        out.append(_row([label, a["win"], a["escaped"], a["incomplete"],
                         b["win"], b["escaped"], b["incomplete"]]))
    out.append("")
    levels = {c["case_id"]: c["level"] for c in panel}
    out.append("Wins by fuzz level per checkpoint (L0 / L1 / L2 / L4):")
    out.append(_row(["Checkpoint", "r1", "r2"]))
    out.append(_row(["---"] * 3))
    for label in LABELS:
        cells = []
        for run in (r1, r2):
            wins = [0, 0, 0, 0]
            for cid, outcome in load_outcomes(run, label).items():
                if outcome == "win" and cid in levels:
                    wins[(0, 1, 2, 4).index(levels[cid])] += 1
            cells.append(" / ".join(str(w) for w in wins))
        out.append(_row([label, *cells]))
    out.append("")
    for name, s in (("r1", r1), ("r2", r2)):
        p = paired(load_outcomes(s, "fresh-initial"), load_outcomes(s, "final"))
        out.append(f"{name} fresh vs final: {p}")
    p = paired(load_outcomes(r1, "final"), load_outcomes(r2, "final"))
    out.append(f"r1 final vs r2 final: {p}")
    out.append("")

    runs = {"r1": load_outcomes(r1, "final"), "r2": load_outcomes(r2, "final")}
    fresh = {"r1f": load_outcomes(r1, "fresh-initial"), "r2f": load_outcomes(r2, "fresh-initial")}
    recorded = recorded_wins(states)
    both = {**runs, **fresh}

    def table(title, rows, label):
        out.append(title)
        out.append(_row([label, "cases", "r1 final", "r2 final", "search",
                         "r1 fresh", "r2 fresh", "L0 cases", "his L0"]))
        out.append(_row(["---"] * 9))
        for g in rows:
            out.append(_row([g["key"], g["n"], g["r1"], g["r2"], g["search"],
                             g["r1f"], g["r2f"], g["his_n"], g["his"]]))
        out.append("")

    kinds = stratify(panel, both, recorded,
                     lambda c: (c["act"], KIND_ORDER.get(c["room_kind"], len(KIND_ORDER)), c["room_kind"]))
    table("By act x kind:", [
        {**g, "key": f"{g['key'][0]} {g['key'][2]}"} for g in kinds], "Act / kind")
    table("By fuzz level:", [
        {**g, "key": f"L{g['key']}"}
        for g in stratify(panel, both, recorded, lambda c: c["level"])], "Level")
    table("By act:", [
        {**g, "key": f"act {g['key']}"}
        for g in stratify(panel, both, recorded, lambda c: c["act"])], "Act")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run1", required=True)
    ap.add_argument("--run2", required=True)
    ap.add_argument("--panel", required=True, help="standard_panel.jsonl")
    ap.add_argument("--run-states", required=True)
    ap.add_argument("--search", help="standard_panel.json (default: beside --panel)")
    ap.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    args = ap.parse_args()
    with open(args.run1) as f:
        r1 = json.load(f)
    with open(args.run2) as f:
        r2 = json.load(f)
    with open(args.panel) as f:
        panel = [json.loads(line) for line in f if line.strip()]
    data = Path(args.run_states).read_bytes()
    states = json.loads(data)["states"]
    search_path = Path(args.search) if args.search else Path(args.panel).with_name("standard_panel.json")
    with open(search_path) as f:
        search = json.load(f)
    with open(args.recipe) as f:
        recipe = json.load(f)
    check_inputs(r1, r2, search, panel, hashlib.sha256(data).hexdigest(), recipe)
    ids = {c["case_id"] for c in panel}
    for s in (r1, r2):
        if set(s["evaluations"]["final"]["cases"]) != ids:
            raise SystemExit("panel case ids differ from run case ids")
    print(render(r1, r2, panel, states))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
