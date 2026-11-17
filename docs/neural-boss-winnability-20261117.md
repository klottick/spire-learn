# Boss winnability of the curriculum scenarios (2026-11-17)

On Gym v1.1.1, both baseline replicates almost never win late boss fights, in
training or on the panel. In the first replicate's training, act 3 bosses were
won in 0 of 1,033 fights and act 2 bosses in about 2%, flat across the run. A
win/loss objective gives the policy nothing to learn from fights it never wins.
This note asks whether those scenarios can be won at all. It is a simulator-only
diagnostic: nothing here trains or feeds a policy.

## Method

`scripts/neural/boss_winnability.py` takes 10 cases per tier from training shards
0–19 of `configs/neural/curriculum-v1/recipe.json`. Tiers are act 1–3 elites and
bosses, sampled with seed 20261117. Each case's entry is rebuilt as a Gym
`standard-combat-v1` catalog entry: the same encounter, HP, deck, relics and
potions, as Ironclad at A0. Each scenario is then played three ways:
- **random:** a uniformly random legal action;
- **heuristic:** a random card play or potion use before ending the turn;
- **search:** Gym's `PublicSearch`.
  - At every decision it scores each legal action over the same 8 sampled
    worlds, consistent with public information only.
  - Sample *k* uses the same world and the same playout random stream for every
    action (common random numbers).
  - Each sample plays out up to 40 steps with the heuristic.
  - A playout scores +1 for a win, −1 for a death, 0 for an escape, and half the
    HP balance at the cap.
  - If a search root is unavailable, it falls back to the heuristic.

The run used `runtime_container.py`, offline on CPU, with exit status 0 and
`learner_changed: false`. Output is in
`artifacts/development/neural-combat/boss-winnability-20261117-b/`:
`winnability.json`, `cases.jsonl` and `trace.json`. 1,212 decisions were
searched. 11 roots were unavailable and fell back; Gym failed closed with "draw
knowledge exceeds the exact model" (9) and "a nested card play holds an enemy
target" (2). An earlier run, `boss-winnability-20261117/`, had no playout common
random numbers and gave the same tier results except act 1 bosses (3/10).

## Results

| Tier | Random | Heuristic | Search | Training win rate (replicate 1, segments 66–98) |
| --- | ---: | ---: | ---: | ---: |
| Act 1 elite | 3/10 | 7/10 | **10/10** | 78% |
| Act 1 boss | 0/10 | 1/10 | 4/10 | 22% |
| Act 2 elite | 0/10 | 2/10 | 4/10 | 17% |
| Act 2 boss | 0/10 | 0/10 | 1/10 | 1.5% |
| Act 3 elite | 0/10 | 0/10 | 1/10 | 9% |
| Act 3 boss | 0/10 | 0/10 | **0/10** | 0% |

Search clearly works: it wins every act 1 elite, a tier the trained policy wins
78% of the time. On act 2 and act 3 bosses it wins 1 of 20.

**A traced act 3 boss fight** (`trace.json`, the first sampled act 3 case, played
by the heuristic).
- The player is at 74/81 HP with a 34-card deck, against Donu and Deca: 500 HP
  between them.
- The bosses' combined HP falls 500 → 476 → 459 → 438 → 371 over the first
  turns: 17–67 damage a turn.
- Their attacks grow from 10×2 to 13×2 to 16×2. The player is at 2 HP by turn 5
  and dies. Killing 500 HP at that rate would take more than ten turns.
- Search also lost this case. All 10 sampled act 3 bosses ended in deaths
  within 7–21 search decisions, from 34–80 starting HP.

The simulator's enemy stats are the real ones. The generated scenarios do not
give the player the damage or scaling a deck needs at these bosses.

Starting HP is not the issue. Across all 5,120 cases in shards 0–19
(`population` in `winnability.json`), median starting HP is 58–68% of maximum in
every tier, monsters included. Median deck size grows by act: 12 cards, then
17–18, then 24–25. The decks are random, upgraded draws, not decks built for
these fights. Bosses are 23.2% of these generated scenarios. In replicate 1's
training they were 21.5% of fights.

## Conclusion

- **Act 3 bosses** are effectively unwinnable as generated. Search won 0 of 10,
  and the traced damage arithmetic leaves no realistic path.
- **Act 2 bosses and act 3 elites** are mostly unwinnable for this search (1 of
  10 each). The search is a lower bound, so this is weaker evidence.
- **The boss failures are a curriculum problem first**, not evidence that the
  learner cannot learn bosses. A large share of boss scenarios give no win
  signal.
- **The panel is affected too.** Its 17 boss cases come from the same generator,
  so its attainable maximum is below 64. This diagnostic did not measure that
  maximum.

## Limits

- **Standard rules, not M10.** This measured `standard-combat-v1`, not the M10
  rules the baselines train on. The same encounters and enemy stats appear in
  both, but small rule differences can matter.
- **A lower bound.** The search is bounded (8 samples, 40-step heuristic
  playouts), so it underestimates the true ceiling. The traced damage arithmetic
  is what makes the act 3 conclusion firm, not the search alone.
- **Small sample.** 10 cases per tier, from training shards, not from the panel
  itself.

## Options

These change the curriculum, and so the panel's meaning. Each needs a decision
and a fresh baseline:
1. **Make late scenarios winnable.** Generate act 2–3 boss and elite decks
   meant for those fights: more upgrades, scaling cards and synergy from the
   recipe's mechanic groups, or higher HP. Then verify them with this
   diagnostic before training.
2. **Reweight the stream.** Sample fewer of the unwinnable tiers, and report
   panel scores against a winnable-case ceiling.
3. **Dense reward.** Credit damage dealt. This gives late fights a gradient, but
   it changes the objective and cannot make an unwinnable fight winnable.
