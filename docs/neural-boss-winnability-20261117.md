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
    worlds, with common random numbers.
  - The worlds are consistent with public information only.
  - Each sample plays out up to 40 steps with the heuristic.
  - A playout scores +1 for a win, −1 for a death, 0 for an escape, and half the
    HP balance at the cap.
  - If a search root is unavailable, it falls back to the heuristic.

The run used `runtime_container.py`, offline on CPU, with exit status 0. Output
is in `artifacts/development/neural-combat/boss-winnability-20261117/`. 1,194
decisions were searched; 17 roots were unavailable and fell back.

## Results

| Tier | Random | Heuristic | Search | Training win rate (replicate 1, late) |
| --- | ---: | ---: | ---: | ---: |
| Act 1 elite | 3/10 | 7/10 | **10/10** | 78% |
| Act 1 boss | 0/10 | 1/10 | 3/10 | 22% |
| Act 2 elite | 0/10 | 2/10 | 4/10 | 17% |
| Act 2 boss | 0/10 | 0/10 | 1/10 | 1% |
| Act 3 elite | 0/10 | 0/10 | 1/10 | 9% |
| Act 3 boss | 0/10 | 0/10 | **0/10** | 0% |

Search clearly works: it wins every act 1 elite, a tier the trained policy wins
78% of the time. On act 2 and act 3 bosses it wins 1 of 20.

**A traced act 3 boss fight.** The trace used a one-off script (not kept) on the
first act 3 case.
- The scenario: Time Eater, which has its real A0 stats (456 HP, attacks of 26,
  7×3, 28 and 42). The player starts at 62/98 HP with a generated 20-odd-card
  deck.
- The deck deals about 8–12 damage per turn, so killing Time Eater would take
  roughly 40 turns.
- The boss deals 20–40 per turn and wins by turn 6.

Other act 3 cases look the same: deaths in 8–19 decisions from 34–80 HP. The
simulator is behaving correctly. The generated scenarios do not give the player
anything like the damage or scaling a real deck has at these bosses.

Starting HP is not the issue. Training scenarios start at a median of about 60%
HP in every tier, monsters included. Deck size grows by act (median 12, 17–18
and 24–25 cards), but the decks are random, upgraded draws, not built decks.

## Conclusion

The curriculum's act 2 boss, act 3 boss and act 3 elite scenarios are largely
**unwinnable as generated**. The bosses' failure is a curriculum problem first,
not evidence that the learner cannot learn bosses. Two consequences:
- About 23% of training scenarios are boss fights. Most act 2 and all act 3
  ones give no win signal.
- The development panel's 17 boss cases come from the same generator, so the
  panel's practical ceiling is well below 64. The 38–39 of 64 baselines are
  probably close to it on the winnable cases.

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
