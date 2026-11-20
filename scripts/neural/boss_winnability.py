"""Diagnostic: how winnable are the curriculum's late boss and elite scenarios?

Replays curriculum scenarios as Gym ``standard-combat-v1`` combats and plays each
with three players: uniformly random legal actions, a play-before-end-turn
heuristic, and a bounded public-belief search (Gym ``PublicSearch``). The search
only ever sees worlds consistent with public information; it is a measurement of
the scenarios, never a training input. Standard rules stand in for the M10 rules
the baselines train on, so results bound winnability approximately, and a search
with heuristic playouts underestimates the true ceiling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'spire-learn-boss-winnability-v1'
PLAYERS = ('random', 'heuristic', 'search')
# Winnability gate: minimum search wins per tier, and a heuristic win ceiling on boss tiers (out of 10).
# The act 2 and act 3 boss minimums are 75% of the gate search's measured win rate on real
# (Baalorlord) boss decks, measured 2026-11-19: act2 16/120 = 13%, act3 44/180 = 24%.
GATE_SEARCH_MIN = {'act1:boss': 5, 'act2:boss': 1, 'act3:boss': 2, 'act3:elite': 3, 'act1:elite': 8}
GATE_HEURISTIC_MAX = 8


# ---------------------------------------------------------------- scenarios

def standard_entry(entry: dict, *, gold: int = 99) -> dict:
    """A curriculum (M10 common-view) entry as a standard-combat-v1 catalog entry."""
    return {'id': entry['id'], 'character': 'ironclad', 'ascension': 0, 'encounter': entry['encounter'],
            'hp': entry['hp'], 'max_hp': entry['max_hp'], 'gold': gold,
            'deck': [{'card': card['card'], 'upgrades': card['upgrades'], 'misc': 0} for card in entry['deck']],
            'relics': list(entry['relics']), 'relic_counters': dict(entry.get('relic_counters') or {}),
            'potions': list(entry['potions'])}


def select_scenarios(recipe_path: Path, *, shards: range, tiers: set[str], per_tier: int, seed: int):
    """Sample up to ``per_tier`` training cases per encounter tier from the given shards."""
    from spire_learn.neural.scenario_stream import generate_scenario_shard, load_scenario_recipe
    from spire_learn.neural.scenario_support import json_value
    recipe = load_scenario_recipe(recipe_path)
    pools: dict[str, list[dict]] = defaultdict(list)
    for index in shards:
        shard = generate_scenario_shard(recipe, split='train', shard_index=index, arm='mixed')
        for case in shard.manifest['cases']:
            tier = shard.annotations[case['case_id']]['encounter_tier']
            if tier in tiers:
                pools[tier].append({'case_id': case['case_id'], 'tier': tier,
                                    'entry': json_value(case['entry'])})
    rng = random.Random(seed)
    chosen = []
    for tier in sorted(pools):
        rng.shuffle(pools[tier])
        chosen.extend(pools[tier][:per_tier])
    return chosen


# ---------------------------------------------------------------- policies

class View:
    """Column lookups in the Standard combat view, resolved once from the descriptor."""

    def __init__(self, descriptor: dict):
        tensors = {tensor['key']: tensor for tensor in descriptor['tensors']}
        def column(key, field):
            return [c['field'] for c in tensors[key]['columns']].index(field)
        self.player_hp, self.player_max = column('player', 'hp'), column('player', 'max_hp')
        self.enemy_hp, self.enemy_max = column('enemies', 'hp'), column('enemies', 'max_hp')
        self.verb = column('actions', 'verb')
        verbs = descriptor['vocabularies']['verb']
        self.end_turn, self.discard = verbs.index('end_turn'), verbs.index('discard_potion')

    def legal(self, observation) -> list[int]:
        return [int(i) for i in observation['action_mask'].nonzero()[0]]

    def heuristic_choice(self, observation, rng: random.Random) -> int:
        legal = self.legal(observation)
        verbs = observation['actions'][legal, self.verb]
        preferred = [i for i, v in zip(legal, verbs) if v not in (self.end_turn, self.discard)]
        return rng.choice(preferred or legal)

    def balance(self, observation) -> float:
        """HP balance in [-1, 1]: player HP fraction minus remaining enemy HP fraction."""
        player = observation['player']
        enemies = observation['enemies']
        alive = enemies[:, self.enemy_max] > 0
        enemy_hp = float(enemies[alive, self.enemy_hp].clip(min=0).sum())
        enemy_max = float(enemies[alive, self.enemy_max].sum()) or 1.0
        mine = float(max(player[self.player_hp], 0)) / float(max(player[self.player_max], 1))
        return mine - enemy_hp / enemy_max


def _native_errors():
    """Gym's typed errors: (unsupported boundary, state, fault)."""
    from spire_lab import _native
    return _native.RunUnsupportedBoundaryError, _native.RunStateError, _native.RunFaultError


def search_choice(env, view: View, observation, *, samples: int, playout_steps: int,
                  rng: random.Random, research_seed: int, stats: Counter) -> int:
    """Pick the legal action with the best mean playout value over shared sampled worlds."""
    unsupported, state_error, fault = _native_errors()
    legal = view.legal(observation)
    if len(legal) == 1:
        return legal[0]
    try:
        search = env.public_search(research_seed, limits={
            'max_live_branches': samples * len(legal) + 8, 'max_branches': samples * len(legal) + 8,
            'max_steps_per_branch': playout_steps + 2, 'max_total_steps': samples * len(legal) * (playout_steps + 2)})
    except (unsupported, state_error) as error:
        stats['root_unavailable'] += 1
        stats[f'root_unavailable:{type(error).__name__}:{str(error)[:80]}'] += 1
        return view.heuristic_choice(observation, rng)
    try:
        values = {}
        for action in legal:
            total = 0.0
            for sample in range(samples):
                # Common random numbers: sample k is the same world and the same playout
                # stream for every action, so actions differ only by the action itself.
                branch = search.branch(sample)
                playout_rng = random.Random(research_seed * 1_000_003 + sample)
                try:
                    total += _playout(branch, action, view, playout_steps, playout_rng)
                except unsupported:
                    stats['branch_refused'] += 1
                    total -= 1.0
                except fault:
                    stats['branch_fault'] += 1
                    total -= 1.0
                finally:
                    try:
                        branch.close()
                    except Exception:
                        pass
            values[action] = total / samples
        stats['searched_decisions'] += 1
        best = max(values.values())
        return rng.choice([action for action, value in values.items() if value == best])
    finally:
        search.close()


def _playout(branch, action: int, view: View, steps: int, rng: random.Random) -> float:
    observation, reward, terminated, truncated, _ = branch.step(action, view='full')
    for _ in range(steps):
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, _ = branch.step(
            view.heuristic_choice(observation, rng), view='full')
    if terminated:
        return float(reward)  # +1 victory, -1 death, 0 escape
    return 0.5 * view.balance(observation)


# ---------------------------------------------------------------- driver

def play(catalog: dict, case: dict, player: str, view: View, *, seed: int, samples: int,
         playout_steps: int, max_steps: int, stats: Counter) -> dict:
    from spire_lab.gymnasium import StandardCombatEnv
    env = StandardCombatEnv(catalog)
    rng = random.Random(seed)
    try:
        observation, info = env.reset(seed=seed, options={'entry_id': case['entry']['id']})
        started = time.perf_counter()
        for step in range(max_steps):
            if player == 'random':
                action = rng.choice(view.legal(observation))
            elif player == 'heuristic':
                action = view.heuristic_choice(observation, rng)
            else:
                action = search_choice(env, view, observation, samples=samples, playout_steps=playout_steps,
                                       rng=rng, research_seed=seed * 1009 + step, stats=stats)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return {'outcome': info.get('outcome') or ('victory' if reward > 0 else 'death' if reward < 0 else 'escaped'),
                        'steps': step + 1, 'hp_after_cleanup': info.get('hp_after_cleanup'),
                        'seconds': round(time.perf_counter() - started, 2)}
        return {'outcome': 'step_cap', 'steps': max_steps, 'seconds': round(time.perf_counter() - started, 2)}
    finally:
        env.close()


def _wins(counts: dict) -> int:
    return int(counts.get('victory', 0))


def gate_report(by_tier: dict, cases_per_tier: int = 10) -> dict:
    """Apply the winnability gate to ``run()``'s ``by_tier``: {tier: {player: {outcome: count}}}.

    The thresholds are rates defined at 10 cases per tier and scale with ``cases_per_tier``.
    """
    failures = []
    heuristic_max = GATE_HEURISTIC_MAX * cases_per_tier // 10
    for tier, base in GATE_SEARCH_MIN.items():
        minimum = -(-base * cases_per_tier // 10)
        if tier not in by_tier:
            failures.append(f'{tier}: missing from results')
            continue
        search = _wins(by_tier[tier].get('search', {}))
        if search < minimum:
            failures.append(f'{tier}: search wins {search} < {minimum}')
        if tier.endswith(':boss'):
            heuristic = _wins(by_tier[tier].get('heuristic', {}))
            if heuristic > heuristic_max:
                failures.append(f'{tier}: heuristic wins {heuristic} > {heuristic_max}')
    return {'passed': not failures, 'failures': failures}


def population_stats(recipe_path: Path, *, shards: range) -> dict:
    """Starting HP fraction and deck size by tier over every case in the shards."""
    from statistics import median
    from spire_learn.neural.scenario_stream import generate_scenario_shard, load_scenario_recipe
    recipe = load_scenario_recipe(recipe_path)
    hp, deck, count = defaultdict(list), defaultdict(list), Counter()
    for index in shards:
        shard = generate_scenario_shard(recipe, split='train', shard_index=index, arm='mixed')
        for case in shard.manifest['cases']:
            tier = shard.annotations[case['case_id']]['encounter_tier']
            hp[tier].append(case['entry']['hp'] / case['entry']['max_hp'])
            deck[tier].append(len(case['entry']['deck']))
            count[tier] += 1
    total = sum(count.values())
    return {'cases': total, 'boss_share': round(sum(v for k, v in count.items() if k.endswith(':boss')) / total, 4),
            'by_tier': {tier: {'cases': count[tier], 'median_hp_fraction': round(median(hp[tier]), 3),
                               'median_deck_size': median(deck[tier])} for tier in sorted(count)}}


def trace(catalog: dict, case: dict, view: View, descriptor: dict, *, seed: int, max_steps: int) -> list[dict]:
    """One heuristic-player fight, step by step: public view values only."""
    from spire_lab.gymnasium import StandardCombatEnv
    columns = {t['key']: [c['field'] for c in t.get('columns', [])] for t in descriptor['tensors']}
    verbs = descriptor['vocabularies']['verb']
    env = StandardCombatEnv(catalog)
    rng = random.Random(seed)
    rows = []
    try:
        observation, info = env.reset(seed=seed, options={'entry_id': case['entry']['id']})
        for step in range(max_steps):
            player = dict(zip(columns['player'], observation['player'].tolist()))
            enemies = [dict(zip(columns['enemies'], row.tolist())) for row in observation['enemies']
                       if row[view.enemy_max] > 0]
            action = view.heuristic_choice(observation, rng)
            rows.append({'step': step, 'turn': player['turn'], 'hp': player['hp'], 'block': player['block'],
                         'energy': player['energy'],
                         'enemies': [{key: enemy[key] for key in ('hp', 'max_hp', 'block', 'intent_damage', 'intent_hits')}
                                     for enemy in enemies],
                         'action_verb': verbs[int(observation['actions'][action, view.verb])]})
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                rows.append({'end': info.get('outcome'), 'reward': float(reward)})
                break
    finally:
        env.close()
    return rows


def _catalog(cases: list[dict], *, native: bool = False) -> dict:
    """The catalog for ``cases``; ``native`` entries are already Standard entries and pass through unchanged."""
    return {'catalog_schema_id': 'spire-gym-standard-combat-catalog-v1', 'profile_id': 'standard-combat-v1',
            'entries': [case['entry'] if native else standard_entry(case['entry']) for case in cases]}


def _view_for(catalog: dict):
    """The (View, descriptor) for a catalog, from a throwaway environment."""
    from spire_lab.gymnasium import StandardCombatEnv
    probe = StandardCombatEnv(catalog)
    try:
        descriptor = probe.observation_spec()
    finally:
        probe.close()
    return View(descriptor), descriptor


def _build_panel(recipe_path: Path) -> dict:
    from spire_learn.neural.scenario_stream import load_scenario_recipe
    from spire_learn.neural.scenario_support import json_value
    from spire_learn.neural.stream_runner import build_development_panel
    return json_value(build_development_panel(load_scenario_recipe(recipe_path)))


def _build_standard_panel(recipe_path: Path, split: str = 'selection') -> dict:
    from spire_learn.neural.recipes import build_panel, load_recipe
    return build_panel(load_recipe(recipe_path), split)


def standard_panel_winnability(output: Path, *, recipe: Path, samples: int, playout_steps: int, max_steps: int,
                               seed: int, limit: int | None) -> dict:
    """Search-only play of a Standard selection panel's entries, passed to Gym as they are.

    ``limit`` plays an evenly strided subset (a pilot). Rows carry run id and floor for local joins only.
    """
    from spire_learn.neural.scenario_support import content_sha256
    panel = _build_standard_panel(recipe)
    cases = []
    for case in panel['manifest']['cases']:
        cases.append({'case_id': case['case_id'], 'entry': case['entry'], 'note': panel['annotations'][case['case_id']]})
    if limit is not None:
        cases = cases[::max(1, len(cases) // limit)][:limit]
    catalog = _catalog(cases, native=True)
    view, _ = _view_for(catalog)
    output.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    by_cell: dict[str, Counter] = defaultdict(Counter)
    path = output / 'standard_panel.jsonl'
    path.write_text('')
    for index, case in enumerate(cases):
        note = case['note']
        outcome = play(catalog, case, 'search', view, seed=seed + index, samples=samples,
                       playout_steps=playout_steps, max_steps=max_steps, stats=stats)
        row = {'case_id': case['case_id'], 'run_id': note['run_id'], 'floor': note['floor'], 'act': note['act'],
               'room_kind': note['room_kind'], 'level': note['level'], 'search': outcome}
        by_cell[f"{note['act']}:{note['room_kind']}:{note['level']}"][outcome['outcome']] += 1
        with path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    result = {'schema': SCHEMA, 'mode': 'standard-panel', 'cases': len(cases), 'samples': samples,
              'playout_steps': playout_steps, 'max_steps': max_steps, 'seed': seed,
              'panel_sha256': content_sha256(panel), 'panel_manifest_sha256': panel.get('manifest_sha256'),
              'search_wins': sum(c.get('victory', 0) for c in by_cell.values()),
              'search_stats': dict(stats),
              'by_stratum_level': {key: dict(value) for key, value in sorted(by_cell.items())}}
    (output / 'standard_panel.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def panel_ceiling(output: Path, *, recipe: Path, samples: int, playout_steps: int, max_steps: int,
                  seed: int) -> dict:
    """Search-only play of every development-panel case: the panel's winnability ceiling."""
    panel = _build_panel(recipe)
    cases = []
    for case in panel['manifest']['cases']:
        note = panel['annotations'][case['case_id']]
        cases.append({'case_id': case['case_id'], 'entry': case['entry'], 'tier': note['encounter_tier'],
                      'cohort': note['cohort']})
    catalog = _catalog(cases)
    view, _ = _view_for(catalog)
    output.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    by_tier: dict[str, Counter] = defaultdict(Counter)
    by_cohort: dict[str, Counter] = defaultdict(Counter)
    (output / 'panel.jsonl').write_text('')
    for index, case in enumerate(cases):
        outcome = play(catalog, case, 'search', view, seed=seed + index, samples=samples,
                       playout_steps=playout_steps, max_steps=max_steps, stats=stats)
        row = {'case_id': case['case_id'], 'tier': case['tier'], 'cohort': case['cohort'],
               'encounter': case['entry']['encounter'], 'hp': case['entry']['hp'],
               'max_hp': case['entry']['max_hp'], 'deck_size': len(case['entry']['deck']), 'search': outcome}
        by_tier[case['tier']][outcome['outcome']] += 1
        by_cohort[case['cohort']][outcome['outcome']] += 1
        with (output / 'panel.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    result = {'search_wins': sum(counts.get('victory', 0) for counts in by_tier.values()),
              'cases': len(cases),
              'by_tier': {key: dict(value) for key, value in sorted(by_tier.items())},
              'by_cohort': {key: dict(value) for key, value in sorted(by_cohort.items())}}
    (output / 'panel_ceiling.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def run(output: Path, *, recipe: Path, shards: range, tiers: set[str], per_tier: int, samples: int,
        playout_steps: int, max_steps: int, seed: int, trace_tier: str | None = None) -> dict:
    cases = select_scenarios(recipe, shards=shards, tiers=tiers, per_tier=per_tier, seed=seed)
    catalog = _catalog(cases)
    view, descriptor = _view_for(catalog)
    output.mkdir(parents=True, exist_ok=True)
    if trace_tier is not None:
        traced = next(case for case in cases if case['tier'] == trace_tier)
        (output / 'trace.json').write_text(json.dumps(
            {'case_id': traced['case_id'], 'entry': traced['entry'],
             'steps': trace(catalog, traced, view, descriptor, seed=seed, max_steps=max_steps)},
            indent=1, sort_keys=True) + '\n')
    stats: Counter = Counter()
    rows = []
    for index, case in enumerate(cases):
        row = {'case_id': case['case_id'], 'tier': case['tier'], 'encounter': case['entry']['encounter'],
               'hp': case['entry']['hp'], 'max_hp': case['entry']['max_hp'], 'deck_size': len(case['entry']['deck'])}
        for player in PLAYERS:
            row[player] = play(catalog, case, player, view, seed=seed + index, samples=samples,
                               playout_steps=playout_steps, max_steps=max_steps, stats=stats)
        rows.append(row)
        with (output / 'cases.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    by_tier = defaultdict(lambda: {player: Counter() for player in PLAYERS})
    for row in rows:
        for player in PLAYERS:
            by_tier[row['tier']][player][row[player]['outcome']] += 1
    result = {'schema': SCHEMA, 'cases': len(rows), 'samples': samples, 'playout_steps': playout_steps,
              'max_steps': max_steps, 'seed': seed, 'shards': [shards.start, shards.stop],
              'search_stats': dict(stats), 'population': population_stats(recipe, shards=shards),
              'by_tier': {tier: {player: dict(counts) for player, counts in players.items()}
                          for tier, players in sorted(by_tier.items())}}
    result['gate'] = gate_report(result['by_tier'], per_tier)
    (output / 'winnability.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--recipe', type=Path, default=Path('configs/neural/curriculum-v1/recipe.json'))
    parser.add_argument('--shards', type=int, nargs=2, default=(0, 20))
    parser.add_argument('--tiers', nargs='+', default=['act1:boss', 'act2:boss', 'act3:boss', 'act3:elite'])
    parser.add_argument('--per-tier', type=int, default=10)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--playout-steps', type=int, default=40)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--panel', action='store_true',
                        help="play the recipe's 64-case development panel with the search player only")
    parser.add_argument('--standard-panel', type=Path, metavar='RECIPE',
                        help="play a Standard recipe's selection panel entries directly with the search player")
    parser.add_argument('--limit', type=int, help='with --standard-panel: play only this many evenly strided cases')
    parser.add_argument('--trace-tier', help='also write trace.json for the first sampled case of this tier')
    arguments = parser.parse_args(argv)
    if not arguments.output.is_absolute():
        print('--output must be an absolute path', file=sys.stderr)
        return 2
    expected = os.environ.get('SPIRE_LEARN_RUNNER_OUTPUT')
    if expected is not None and Path(expected).resolve() != arguments.output.resolve():
        print('output must match SPIRE_LEARN_RUNNER_OUTPUT', file=sys.stderr)
        return 2
    recipe = arguments.recipe if arguments.recipe.is_absolute() else _ROOT / arguments.recipe
    if arguments.standard_panel is not None:
        standard = arguments.standard_panel if arguments.standard_panel.is_absolute() else _ROOT / arguments.standard_panel
        result = standard_panel_winnability(arguments.output, recipe=standard, samples=arguments.samples,
                                            playout_steps=arguments.playout_steps, max_steps=arguments.max_steps,
                                            seed=arguments.seed, limit=arguments.limit)
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.panel:
        result = panel_ceiling(arguments.output, recipe=recipe, samples=arguments.samples,
                               playout_steps=arguments.playout_steps, max_steps=arguments.max_steps,
                               seed=arguments.seed)
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run(arguments.output, recipe=recipe, shards=range(*arguments.shards), tiers=set(arguments.tiers),
                 per_tier=arguments.per_tier, samples=arguments.samples, playout_steps=arguments.playout_steps,
                 max_steps=arguments.max_steps, seed=arguments.seed, trace_tier=arguments.trace_tier)
    print(json.dumps({key: result[key] for key in ('cases', 'by_tier', 'search_stats', 'gate')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
