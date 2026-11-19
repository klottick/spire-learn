# Curriculum v2: winnability gate, real boss decks and panel ceilings

Started 2026-11-19 under `docs/superpowers/specs/2026-11-19-curriculum-v2-design.md`.
This note covers the curriculum, its gate and the four training arms. Everything here is a simulator-only diagnostic. The search
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

## Training arms (Task 5)

Four fresh-weight runs on Gym v1.1.1 trained on curriculum v2 with real boss
decks, all from learner `2e8ad0d`.

**Setup.**
- Each run used 405,504 learned decisions. Budgets and the model are the v1.1.1
  baselines'.
- Seed pairs are also the baselines': "r1" uses replicate 1's seeds and "r2"
  replicate 2's.
- Arms:
  - `cv2` and `cv2-r2`: win/loss objective.
  - `cv2-hp` and `cv2-hp-r2`: the HP-shaped objective
    (`spire-learn-combat-winloss-hpshaped-v1`, β = 0.5, γ = 0.99). The plan
    notes don't name it; the template's `[objective]` table and identity do.
- Each run was evaluated on the v2 panel, its primary panel, and on the v1 panel
  for continuity.
- **Re-evaluation.** Both r2 runs first stopped their fresh-initial v2-panel
  evaluation. A long real-deck fight used all 256 case actions under the untrained
  policy, and that was misclassified as an operational failure. PR #20 now
  counts clean budget exhaustion as censored. The r2 runs' recorded checkpoints
  were re-evaluated with `--reevaluate`, without retraining, into
  `…-r2-20261119-reeval/`.
- All runs ended `target_reached` with `learner_changed: false`. Training waited
  for the shared GPU for 240–810 s per run. Elapsed time to the end of
  training, including those waits, was 1,960–2,542 s.

**Panel wins.** Untrained-to-final wins on each panel, out of 64; the
fresh-vs-final columns are fresh-only / final-only cases and the exact McNemar p.

| Arm | v2: fresh → 135k → 270k → final | v2 fresh vs final | v1: fresh → final | v1 fresh vs final |
| --- | --- | ---: | --- | ---: |
| cv2 (r1) | 17 → 29 → 35 → **35** | 2 / 20, p < 0.001 | 18 → **41** | 0 / 23, p < 0.001 |
| cv2-r2 | 10* → 32 → 32 → **33** | 0 / 22, p < 0.001 | 17 → **36** | 0 / 19, p < 0.001 |
| cv2-hp (r1) | 17 → 31 → 35 → **37** | 2 / 22, p < 0.001 | 18 → **37** | 0 / 19, p < 0.001 |
| cv2-hp-r2 | 10* → 31 → 32 → **39** | 0 / 28, p < 0.001 | 17 → **39** | 0 / 22, p < 0.001 |

\* One fresh-initial case was censored, so 63 cases are paired.

For reference, the v1.1.1 baselines (curriculum v1) finished at 39 and 38 on
the v1 panel. Search ceilings are 39/64 on both panels.

**Comparisons.** These are case-level paired tests on final policies with
matched seeds:
- **Shaped vs win/loss, v2 panel.** r1: win/loss-only 3, shaped-only 5
  (p = 0.73). r2: 2 vs 8 (p = 0.11). The shaped arms score higher in both pairs
  (37 and 39 vs 35 and 33), but neither difference is significant with these
  replicates.
- **Shaped vs win/loss, v1 panel.** 5 vs 1 (p = 0.22) and 2 vs 5 (p = 0.45). No
  consistent direction.
- **Curriculum v2 vs v1 on the v1 panel**, comparing cv2 against each baseline
  with matching seeds. r1: v1-only 2, cv2-only 4 (p = 0.69). r2: 4 vs 2
  (p = 0.69). No significant difference: the v1-panel scores (36–41) stay at
  the level the v1 baselines reached (38–39), around that panel's search ceiling.
- **Replicate agreement on the v2 panel.** cv2 vs cv2-r2: 5 vs 3. cv2-hp vs
  cv2-hp-r2: 3 vs 5.

**Final v2-panel wins by tier.**

| Arm | Act 1 boss | Act 1 elite | Act 2 boss | Act 2 elite | Act 3 boss | Act 3 elite | Act 4 boss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cv2 | 4/6 | 5/6 | 1/6 | 2/6 | 1/6 | 0/4 | 0/1 |
| cv2-r2 | 3/6 | 5/6 | 0/6 | 1/6 | 1/6 | 0/4 | 0/1 |
| cv2-hp | 5/6 | 5/6 | 0/6 | 4/6 | 0/6 | 1/4 | 0/1 |
| cv2-hp-r2 | 5/6 | 6/6 | 0/6 | 2/6 | 0/6 | 1/4 | 1/1 |

**Training-time boss wins** over the last third of each training stream: prepared
shards `train-66` to `train-98`, from each segment's standalone fight outcomes.
Tiers are assigned by encounter: act 1 slime_boss/guardian/hexaghost, act 2
bronze_automaton/collector/champ, act 3 awakened_one/donu_and_deca/time_eater,
act 4 corrupt_heart. These are sampled-policy rates on different deck
distributions: v2 boss fights use real winning decks. So they measure how often
boss fights produce wins to learn from, not policy strength.

| Run | Act 1 boss | Act 2 boss | Act 3 boss | Act 4 boss |
| --- | ---: | ---: | ---: | ---: |
| v1.1.1 baseline r1 (curriculum v1) | 85/395 (22%) | 6/410 (1.5%) | 0/335 (0%) | 0/30 |
| cv2 | 158/330 (48%) | 34/350 (10%) | 27/293 (9%) | 0/27 |
| cv2-r2 | 173/344 (50%) | 25/333 (8%) | 7/333 (2%) | 0/37 |
| cv2-hp | 182/343 (53%) | 36/327 (11%) | 18/265 (7%) | 0/39 |
| cv2-hp-r2 | 167/321 (52%) | 45/326 (14%) | 24/343 (7%) | 0/43 |

**Reading.**
- **Boss fights now yield wins to learn from.** That was curriculum v2's goal.
  In training, act 1 boss wins roughly double, and act 2 and 3 boss wins rise
  about 5 to 10 times, from near zero. Curriculum v1 baseline r1 had 0 act 3 boss
  wins in 1,033 training fights; r2 had 1 in 1,069. This shows the signal
  exists, not that the policy is stronger.
- **The panels move less.** On the v2 panel, act 2 and act 3 bosses are still
  won 0–1 of 6 at the end of training. On the v1 panel, curriculum v2 neither
  helps nor hurts.
- **Shaping is a possible gain.** The HP-shaped objective is slightly higher on
  the v2 panel in both seed pairs (p = 0.73 and 0.11), but not on the v1 panel.
  More replicates would be needed to tell.
- **Both objectives learn far beyond the fresh policy** on both panels, as the
  v1 baselines also did.

## Limits

- **Standard rules, not M10.** The search plays standard-combat-v1 rebuilds at
  A0, not the M10 rules the learner trains on. Search results are lower bounds.
- **Small, noisy samples.** One 60-case reference measurement ranged 10–20/60
  for act 3 bosses.
- **Real decks are partial.** They are A20 decks mapped onto a 75-card and
  30-relic support, so out-of-support cards and relics are missing. Selection
  and final splits reuse about 10% of them through the composition partition.
- **Near-duplicates across splits.** The composition split hashes the real deck
  together with any lesson targets and coverage card. So the same recorded base
  deck can appear in train with one extra card and in selection with another,
  and a v2 "novel deck" panel case may be one card away from a training deck.
- **Two replicates per arm**, with one seed pair each. The v2 panel's
  fresh-initial r2 score has one censored case. Training ran on a GPU shared
  with other services.
