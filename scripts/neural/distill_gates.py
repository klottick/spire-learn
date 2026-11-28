"""Pre-registered distillation gates G2-G4 from baseline-format summaries.

Exact two-sided McNemar on paired selection-panel cases, final checkpoint
against final checkpoint, seed 1 with seed 1. "Beats" needs both seed pairs to
favour the challenger on paired wins and the pooled discordant test to be
significant. A case that is not a win or a loss on either side (escaped,
censored, incomplete) is dropped from that pair and counted.

Stdlib only. Usage:
  distill_gates.py --arm ppo=r1/summary.json,r2/summary.json \
      --arm s1=... --arm s2=... --arm s3=... --arm distilled=... --out gates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.neural.standard_results import load_outcomes, mcnemar_exact

LABEL = "final"
# gate -> (alpha per test, [(challenger arm, baseline arm), ...])
GATES: dict[str, tuple[float, list[tuple[str, str]]]] = {
    "G2": (0.05 / 3, [("s1", "ppo"), ("s2", "ppo"), ("s3", "ppo")]),
    "G3": (0.025, [("s2", "s1"), ("s3", "s1")]),
    "G4": (0.05, [("distilled", "ppo")]),
}


class GateInputError(ValueError):
    pass


def pair_stats(chal: dict[str, str], base: dict[str, str]) -> dict:
    """Discordant counts for one seed pair; non win/loss cases are dropped."""
    if set(chal) != set(base):
        raise GateInputError("a seed pair must cover identical case ids")
    both = only_c = only_b = neither = 0
    dropped = {"escaped": 0, "other": 0}
    for cid in sorted(chal):
        x, y = chal[cid], base[cid]
        if x not in ("win", "loss") or y not in ("win", "loss"):
            for o in (x, y):
                if o == "escaped":
                    dropped["escaped"] += 1
                elif o not in ("win", "loss"):
                    dropped["other"] += 1
            dropped.setdefault("cases", 0)
            dropped["cases"] += 1
            continue
        if x == "win" and y == "win":
            both += 1
        elif x == "win":
            only_c += 1
        elif y == "win":
            only_b += 1
        else:
            neither += 1
    n = both + only_c + only_b + neither
    return {
        "paired": n, "both_win": both, "only_challenger": only_c,
        "only_baseline": only_b, "neither": neither,
        "dropped_cases": dropped.get("cases", 0),
        "dropped_escaped_sides": dropped["escaped"],
        "dropped_censored_sides": dropped["other"],
        "win_rate_diff": (only_c - only_b) / n if n else 0.0,
        "p": mcnemar_exact(only_c, only_b),
    }


def beats(chal: list[dict[str, str]], base: list[dict[str, str]], alpha: float) -> dict:
    if len(chal) != len(base) or not chal:
        raise GateInputError("each arm needs the same non-zero number of seeds")
    pairs = [pair_stats(c, b) for c, b in zip(chal, base)]
    oc = sum(p["only_challenger"] for p in pairs)
    ob = sum(p["only_baseline"] for p in pairs)
    pooled_p = mcnemar_exact(oc, ob)
    each_favours = all(p["only_challenger"] > p["only_baseline"] for p in pairs)
    n = sum(p["paired"] for p in pairs)
    return {
        "pairs": pairs,
        "pooled": {"only_challenger": oc, "only_baseline": ob, "paired": n,
                   "p": pooled_p,
                   "win_rate_diff": (oc - ob) / n if n else 0.0},
        "alpha": alpha,
        "each_pair_favours_challenger": each_favours,
        "significant": pooled_p < alpha,
        "beats": each_favours and pooled_p < alpha,
        "dropped_cases": sum(p["dropped_cases"] for p in pairs),
    }


def evaluate(arms: dict[str, list[dict[str, str]]]) -> dict:
    """Verdict per gate; a gate runs only if all its arms are supplied."""
    out: dict[str, dict] = {}
    for gate, (alpha, tests) in GATES.items():
        needed = {a for t in tests for a in t}
        if not needed <= set(arms):
            out[gate] = {"status": "skipped", "missing_arms": sorted(needed - set(arms))}
            continue
        results = {f"{c}_vs_{b}": beats(arms[c], arms[b], alpha) for c, b in tests}
        out[gate] = {"status": "evaluated", "alpha_per_test": alpha,
                     "tests": results,
                     "pass": any(r["beats"] for r in results.values())}
    return out


def load_arm(paths: list[str]) -> list[dict[str, str]]:
    return [load_outcomes(json.loads(Path(p).read_text()), LABEL) for p in paths]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=S1,S2")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    arms: dict[str, list[dict[str, str]]] = {}
    for spec in args.arm:
        name, _, paths = spec.partition("=")
        if not paths or name in arms:
            ap.error(f"bad or duplicate --arm {spec!r}")
        arms[name] = load_arm(paths.split(","))
    text = json.dumps(evaluate(arms), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
