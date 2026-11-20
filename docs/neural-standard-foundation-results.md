# Standard foundation results

## Winnability reference

Measured 2026-11-20 on the 256-case selection panel of `configs/neural/baalorlord-standard-v1`
(panel SHA256 `0bf7b55e1f5ae243e05fe653456b2a982b5820e1325da7ed73c6c9f0aaef8b7a`; 64 cases at each
fuzz level 0, 1, 2 and 4, stratified by act x room kind). Gym revision
`e3c13836bd1458be8a87b93160f5c350f78fcd80` (rules epoch `gym-1.1.1`), Standard rules, Ironclad ascension 20.

**Method.** `scripts/neural/boss_winnability.py --standard-panel RECIPE` plays each panel entry directly as a
Standard catalog entry (no M10 conversion) with Gym `PublicSearch`: at every decision it samples worlds
consistent with public information and scores each legal action by mean heuristic playout value
(common random numbers across actions). A decision whose root the search cannot model (hidden intents, draw
knowledge beyond the exact model, nested targeting) falls back to a play-before-end-turn heuristic.
Search is a measurement, never a training input.

**Budget.** 32 samples, 80 playout steps, `max_steps` 1200, one search-played fight per case, 4 CPUs.
An 8-case pilot cost about 31 s wall; the full pass took about 1 h wall (0.71 h summed fight time), well under
the 6 h limit, so the budget was not reduced. Container run exit status 0, learner unchanged.

**Overall.** Search won 219 of 256 (34 deaths, 3 escapes; escapes are neither wins nor losses).
His recorded win rate on the 64 level-0 cases is 64/64: all 64 of those fights were won in his runs.
Search won 56 of the same 64 level-0 cases (the level columns below).

Search wins / cases by act x kind x fuzz level, beside his recorded wins on the level-0 cases (level-0
cases are his unmodified states; fuzzed levels have no recorded outcome):

| Act | Kind | L0 | L1 | L2 | L4 | All levels | His recorded wins, L0 |
|---|---|---|---|---|---|---|---|
| 1 | monster | 14/14 | 14/14 | 14/14 | 14/14 | 56/56 | 14/14 |
| 1 | elite | 7/7 | 6/7 | 4/7 | 6/7 | 23/28 | 7/7 |
| 1 | boss | 0/2 | 0/2 | 0/2 | 1/2 | 1/8 | 2/2 |
| 2 | monster | 11/11 | 11/11 | 11/11 | 10/11 | 43/44 | 11/11 |
| 2 | elite | 5/5 | 4/5 | 5/5 | 5/5 | 19/20 | 5/5 |
| 2 | boss | 1/2 | 1/2 | 1/2 | 2/2 | 5/8 | 2/2 |
| 3 | monster | 10/10 | 10/10 | 9/10 | 9/10 | 38/40 | 10/10 |
| 3 | elite | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 | 5/5 |
| 3 | boss | 2/4 | 2/4 | 2/4 | 1/4 | 7/16 | 4/4 |
| 4 | elite | 1/2 | 1/2 | 2/2 | 2/2 | 6/8 | 2/2 |
| 4 | boss | 0/2 | 1/2 | 0/2 | 0/2 | 1/8 | 2/2 |

A recorded win means the fight was won in his run. The table has 842 lost fights out of 49,895 states overall (1.7%), so
his rate is near the top by construction; 64/64 is a property of this sample, not a ceiling. Search is far below him on
bosses (act 1 and act 4 boss 0/2 each at level 0, against his 2/2), which is the expected underestimate of the
ceiling and not a statement that those states are unwinnable.

**Excluded strata.** None. No stratum had both 0 search wins and 0 of his wins; every stratum has at least
one search win or one recorded win, so the recipe's `excluded_strata` stays empty and the panel hash is
unchanged. (Act 1 boss and act 4 boss have a single search win each over 8 cases, but his level-0 wins in them
show they are winnable.)

**Caveats.**
- Search with heuristic playouts underestimates the achievable ceiling; the low boss numbers are a lower bound.
- One player's data: states come from Baalorlord's runs only, and his recorded win rate reflects selection
  into reaching those states.
- Simulator results only; no claim about transfer to the original game.
- The per-stratum counts are small (2 to 14 per cell), so single-cell differences are noise.
- RNG-001 and TIME-001 qualifications apply: simulator randomness and timing/turn-clock behaviour are not
  claimed to match the original game.
- Search decided 5315 decisions with the search proper and fell back to the heuristic on 570 unmodelled roots
  (329 draw knowledge beyond the exact model, 239 hidden intents, 2 nested targeting); 167 playout branches
  were refused by the engine and scored as losses.
