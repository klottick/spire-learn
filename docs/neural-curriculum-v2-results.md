# Curriculum v2: winnability gate, real boss decks and panel ceilings

Started 2026-11-19 under `docs/superpowers/specs/2026-11-19-curriculum-v2-design.md`.
This note covers the curriculum and its gate. The four training arms are added
when they finish. Everything here is a simulator-only diagnostic. The search
player sees public information only and is never a training input.

## Gate history

The gate is `scripts/neural/boss_winnability.py`. It runs Gym `PublicSearch` with 8
sampled worlds and 40-step heuristic playouts on Standard combat rebuilds of
training-shard cases, at A0. The thresholds started as the spec's search-win counts
per 10 cases:
- act 1 boss 5, act 2 boss 4, act 3 boss 3, act 3 elite 3, act 1 elite 8;
- the heuristic below 9 on boss tiers.

The gate moved to 20 cases per tier with proportional thresholds (ruling 9),
because 10 cases were too noisy: act 1 bosses moved 5 → 2 with unchanged knobs
when the recipe hash reseeded sampling.

**Prepared decks (Task 1), tuned for 6 iterations.** Search wins per tier:

| Iteration | Act 1 boss | Act 2 boss | Act 3 boss | Act 3 elite | Act 1 elite |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (initial knobs) | 5/10 | 1/10 | 1/10 | 5/10 | 9/10 |
| 2 (maxed act 2–3 knobs) | 2/10 | 4/10 | 1/10 | 4/10 | 10/10 |
| 3 (thin act 2–3 decks; from here 20 per tier) | 6/20 | 13/20 | 1/20 | 9/20 | 17/20 |
| 4 (small cores, thin act 1 and act 3 decks) | 11/20 | 7/20 | 4/20 | 10/20 | 17/20 |
| 5 (small act 2 core, full-HP act 3) | 14/20 | 5/20 | 1/20 | 7/20 | 19/20 |
| 6 | 14/20 | 7/20 | 1/20 | 7/20 | 20/20 |

Findings from the tuning:
- **Copies, not cores.** Each mechanic group has only 3 cards, so large
  `core_cards` produced piles of copies.
- **Dilution.** Developed-stage fill diluted the prepared core. The optional
  `deck_size` knob (ruling 10) addresses this.
- **The act 3 boss gate was unreachable** with prepared decks. Act 2 bosses
  swung 5–13/20 between reseeds.

**Search limit check.** A stronger search, with 32 worlds and 80-step
playouts on iteration 6, won 11/20 act 2 bosses and 3/20 act 3 bosses. This
was not paired with the gate cases. Act 2 had mostly been a search limit; act 3
stayed a ceiling.

## Real boss decks (Task 2c)

By user decision, act 2 and act 3 boss scenarios use real winning decks.

**Source.** Public Slay the Spire run files from baalorlord.tv (profile 0),
fetched politely: one request per second, with an identifying User-Agent. That
gave 676 Ironclad runs, all at A20. The table's decks come from 503 of those
runs; the rest never reached a boss floor or were excluded. `scripts/neural/real_boss_decks.py`
rebuilds each deck as it stood entering the act 2 boss (floor 33) or the act 3
bosses (floors 50 and 51):
- picks, purchases, purges, smiths and event changes after that floor are
  undone;
- decks changed by unrecorded relic effects are dropped (rulings 15 and 17);
- cards and relics outside the curriculum support are dropped, mostly Ascender's
  Bane and colorless cards.

**The table** is local only and not committed, per the user's choice. The
recipe pins it by sha256 `fd28ab41…526a`, and the script regenerates it. It
holds 1,245 decks; the won/total counts are:

| Encounter | Won/total |
| --- | ---: |
| Champ | 94/101 |
| Collector | 123/136 |
| Bronze Automaton | 104/114 |
| Awakened One | 291/301 |
| Time Eater | 300/310 |
| Donu and Deca | 280/283 |

**Reference rates.** Real winning decks with the same gate search, at A0 with
20 decks per boss, give these search wins:

| Measurement | Act 2 bosses | Act 3 bosses |
| --- | ---: | ---: |
| First extraction | — | 14/60 |
| Pre-fix table | 8/60 | 20/60 |
| Cleaned table | 8/60 | 10/60 |
| **Pooled** | **16/120 = 13%** | **44/180 = 24%** |

In the table, Baalorlord won 871 of 894 of these act 3 fights (97%) at A20.
The same decks, played by the gate search at A0, reach only the rates above.
That is consistent with the gate search, more than the decks, limiting
measured boss winnability.

The comparison is not exact:
- the decks lose their out-of-support cards and relics;
- they are played under Standard rules, at A0.

The pooled rates mix three table versions. Two of the act 3 measurements used
tables later found to include a few post-floor relic changes (rulings 15 and
17).

The act 2 and act 3 boss thresholds were therefore re-anchored at 75% of these
references (rulings 14 and 18). That is 1 and 2 per 10 cases in code, which at
the 20 cases used here means act 2 boss 2/20 and act 3 boss 4/20.

**Gate on the real-deck v2 recipe (passed).** Recipe commit `0935f03`, 20 cases
per tier:

| Tier | Random | Heuristic | Search | Threshold |
| --- | ---: | ---: | ---: | ---: |
| Act 1 elite | 9 | 17 | 18 | ≥ 16 |
| Act 1 boss | 0 | 5 | 12 | ≥ 10 |
| Act 2 elite | 0 | 2 | 5 | — |
| Act 2 boss | 0 | 0 | 5 | ≥ 2 |
| Act 3 elite | 1 | 5 | 9 | ≥ 6 |
| Act 3 boss | 0 | 0 | 7 | ≥ 4 |

The heuristic stays well below every boss-tier cap. Every gated tier reaches
at least 4/20, so `tier_weights` stay at 1.0 (ruling 20). The act 4 tiers are
not gated and are about 1% of scenarios.

## Panel ceilings

The search player played all 64 development-panel cases (`--panel`).

| Panel | Search wins | familiar_new_seed | novel_deck | reserved_pair | resource_later_tier |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 39/64 | 11/16 | 11/16 | 11/16 | 6/16 |
| v2 | 39/64 | 14/16 | 11/16 | 10/16 | 4/16 |

The v1 panel's search ceiling (39) is within one win of both v1.1.1 baselines'
final scores (39 and 38). Differences of one or two wins are not effects, and
the search plays Standard rules at A0, not M10. So this suggests, but does not
show, that v1-panel scores leave little room to improve.

## Limits

- **Standard rules, not M10.** The search plays standard-combat-v1 rebuilds at
  A0, not the M10 rules the learner trains on. Search results are lower bounds.
- **Small, noisy samples.** One 60-case reference measurement ranged 10–20/60
  for act 3 bosses.
- **Real decks are partial.** They are A20 decks mapped onto a 75-card and
  30-relic support, so out-of-support cards and relics are missing. Selection
  and final splits reuse about 10% of them through the composition partition.
