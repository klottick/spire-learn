#!/usr/bin/env python3.12
"""Extract Gym-validated Standard combat entries from public Slay the Spire run files.

    run_states.py extract --raw DIR [DIR...] --aliases PATH --output FILE --report FILE
    run_states.py stats --table FILE --output FILE

Each fight of each run is rebuilt with ``spire_learn.neural.run_states``, mapped to a
Gym Standard encounter row, converted to a ``standard-combat-v1`` catalog entry and
validated by the Gym in catalogs of at most 256 entries (a rejected catalog is bisected
so that only the bad entries drop). Run files are untrusted: only ``json`` parses them.
The CLI needs the Gym, so it runs in the offline runtime container
(``scripts/neural/runtime_container.py --entry run_states.py``); ``build`` takes the
Gym's validator and encounter listing as parameters so it tests without the Gym.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

from spire_learn.neural.run_states import (  # noqa: E402
    Drop, FightState, as_int, fight_floors, run_history, state_at, CHARACTERS, POTION_BELT,
)
from spire_learn.neural.scenario_support import canonical_bytes  # noqa: E402
from spire_learn.neural.standard_catalog import (  # noqa: E402
    CATALOG_SCHEMA, MAX_CATALOG, PROFILE, CatalogRejected, Validate, bind_gym, potion_slots, validate_entries,
)
from spire_learn.neural.standard_names import NameAliases, load_aliases  # noqa: E402

SCHEMA = 'spire-learn-run-states-v1'
REPORT_SCHEMA = 'spire-learn-run-states-report-v1'
GYM_REVISION = 'e3c13836bd1458be8a87b93160f5c350f78fcd80'
ASCENSION = 20
BURNING_KINDS = ('strength', 'max_hp', 'metallicize', 'regeneration')  # the Gym's order
DROP_LIMIT = 0.10
DEFAULT_ENCOUNTER_MAP = Path(__file__).resolve().parents[2] / 'configs/neural/standard/encounter-map.json'
DEFAULT_ALIASES = Path(__file__).resolve().parents[2] / 'configs/neural/standard/name-aliases.json'
STATS_SCHEMA = 'spire-learn-run-states-stats-v1'

# ---------------------------------------------------------------- input

def collect_runs(dirs: Iterable[Path]) -> list[dict]:
    """Every run file's JSON, once per run_id (``timestamp``).

    A byte-identical copy of a run in another directory is kept once; two different
    files with one run_id raise.
    """
    found: dict[int, tuple[bytes, Path, dict]] = {}
    for directory in dirs:
        for path in sorted(Path(directory).glob('*.json')):
            data = path.read_bytes()
            run = json.loads(data)
            run_id = as_int(run['timestamp'])
            if run_id in found:
                if found[run_id][0] != data:
                    raise ValueError(f'run {run_id} differs between {found[run_id][1]} and {path}')
                continue
            found[run_id] = (data, path, run)
    return [item[2] for _, item in sorted(found.items())]


# ---------------------------------------------------------------- conversion

def burning_kind(run_id: str, floor: int) -> str:
    """Runs do not record which burning buff applied: a uniform pick by hash of (run_id, floor)."""
    digest = hashlib.sha256(f'{run_id}:{floor}'.encode()).digest()
    return BURNING_KINDS[int.from_bytes(digest[:8], 'big') % len(BURNING_KINDS)]


def to_standard_entry(state: FightState, encounter: str) -> dict:
    """The ``standard-combat-v1`` catalog entry for a state fought at ``encounter``."""
    slots = potion_slots(state.relics)
    if len(state.potions) > slots:
        raise ValueError(f'{len(state.potions)} potions for {slots} slots')
    entry = {
        'id': f'{state.run_id}-{state.floor}', 'character': state.character, 'ascension': ASCENSION,
        'encounter': encounter, 'hp': state.hp, 'max_hp': state.max_hp, 'gold': state.gold,
        'deck': state.deck, 'relics': state.relics, 'relic_counters': state.relic_counters,
        'potions': list(state.potions) + [None] * (slots - len(state.potions)),
    }
    if state.bottled_cards:
        entry['bottled_cards'] = state.bottled_cards
    if state.burning:
        entry['burning'] = burning_kind(state.run_id, state.floor)
    return entry


# ---------------------------------------------------------------- extraction

def _encounter_for(fight, encounter_map: dict, rows: dict) -> str | Drop:
    """The Gym encounter id for a fight, a scope exclusion, or a drop."""
    target = encounter_map.get(fight.enemies)
    if target is None:
        if fight.room_kind == 'event':
            return Drop(f'scope_excluded:{fight.enemies}', 'event fight with no Gym encounter row')
        return Drop(f'unmapped_encounter:{fight.enemies}', 'no Gym encounter row for this name')
    row = rows[target]
    # An event fight may use a monster or elite row, never a boss row; every other fight must match act and kind.
    kind_ok = row['kind'] in ('Monster', 'Elite') if fight.room_kind == 'event' else row['kind'].lower() == fight.room_kind
    if row['act'] != fight.act or not kind_ok:
        return Drop(f'encounter_mismatch:{fight.enemies}',
                    f'{target} is act {row["act"]} {row["kind"]}, fight is act {fight.act} {fight.room_kind}')
    return target


def build(runs: Sequence[dict], names: NameAliases, encounters: Sequence[dict], encounter_map: dict,
          validate: Validate, *, gym_revision: str = GYM_REVISION) -> tuple[dict, dict]:
    """The state table and its report for ``runs``; ``validate`` is the Gym's catalog check."""
    rows = {row['id']: row for row in encounters}
    missing = sorted(set(encounter_map.values()) - set(rows))
    if missing:
        raise ValueError(f'encounter map targets without a Gym row: {missing}')
    run_ids = [as_int(run['timestamp']) for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError('duplicate run_id in the input runs')
    outcomes: list[tuple[str, int, str, str, str]] = []  # (character, act, kind, outcome, reason)
    run_counts: Counter = Counter()
    candidates: list[tuple[FightState, dict]] = []
    for run in runs:
        character = CHARACTERS.get(run.get('character_chosen'), 'unknown')
        run_counts[character] += 1
        try:
            fights = fight_floors(run)
            history = run_history(run, names)
        except (ValueError, KeyError, TypeError) as error:
            outcomes.append((character, 0, 'run', 'dropped', f'malformed_run:{str(error)[:80]}'))
            continue
        for fight in fights:
            where = (character, fight.act, fight.room_kind)
            if as_int(run.get('ascension_level', 0)) != ASCENSION:
                outcomes.append((*where, 'scope_excluded', f'scope_excluded:ascension_{run.get("ascension_level")}'))
                continue
            target = _encounter_for(fight, encounter_map, rows)
            result: FightState | Drop = target
            if not isinstance(target, Drop):
                try:
                    result = state_at(run, fight, names, history)
                except (ValueError, KeyError, TypeError) as error:
                    result = Drop('malformed_run', str(error)[:80])
            if isinstance(result, Drop):
                outcome = 'scope_excluded' if result.reason.startswith('scope_excluded:') else 'dropped'
                outcomes.append((*where, outcome, result.reason))
                continue
            state = result
            tags = list(state.tags)
            if state.burning:
                tags.append('burning_kind_sampled')
            state = FightState(**{**state.__dict__, 'tags': tags})
            candidates.append((state, to_standard_entry(state, target)))
    accepted, rejected = validate_entries([entry for _, entry in candidates], validate)
    reasons = {entry['id']: reason for entry, reason in rejected}
    kept: list[dict] = []
    for state, entry in candidates:
        where = (state.character, state.act, state.room_kind)
        if entry['id'] in reasons:
            outcomes.append((*where, 'dropped', reasons[entry['id']]))
            continue
        outcomes.append((*where, 'kept', ''))
        kept.append({**state.__dict__, 'entry': entry})
    kept.sort(key=lambda s: (s['character'], s['run_id'], s['floor']))
    ids = [s['entry']['id'] for s in kept]
    if len(ids) != len(set(ids)):
        raise ValueError('state ids are not unique')
    document = {'schema': SCHEMA, 'aliases_sha256': names.sha256, 'gym_revision': gym_revision, 'states': kept}
    return document, _report(outcomes, run_counts, kept, rejected)


def _report(outcomes: list, run_counts: Counter, kept: list[dict], rejected: list) -> dict:
    characters: dict[str, dict] = {}
    strata: dict[tuple, Counter] = defaultdict(Counter)
    for character, act, kind, outcome, reason in outcomes:
        entry = characters.setdefault(character, {
            'runs': run_counts[character], 'states': 0, 'kept': 0, 'dropped': 0, 'scope_excluded': 0,
            'drop_reasons': Counter(), 'scope_excluded_reasons': Counter(), 'potions_assumed_empty': 0})
        entry['states'] += 1
        entry[outcome] += 1
        strata[(character, act, kind)][outcome] += 1
        if outcome == 'dropped':
            entry['drop_reasons'][reason] += 1
        elif outcome == 'scope_excluded':
            entry['scope_excluded_reasons'][reason] += 1
    for state in kept:
        if 'potions_assumed_empty' in state['tags']:
            characters[state['character']]['potions_assumed_empty'] += 1
    for entry in characters.values():
        denominator = entry['kept'] + entry['dropped']
        rate = entry['dropped'] / denominator if denominator else 0.0
        entry.update(drop_denominator=denominator, drop_rate=round(rate, 6),
                     drop_rule='pass' if rate <= DROP_LIMIT else 'fail',
                     drop_reasons=dict(sorted(entry['drop_reasons'].items())),
                     scope_excluded_reasons=dict(sorted(entry['scope_excluded_reasons'].items())))
    rejections = Counter(reason for _, reason in rejected)
    return {
        'schema': REPORT_SCHEMA,
        'drop_rule': (f'informational: a character fails when more than {DROP_LIMIT:.0%} of kept + dropped '
                      'states drop (scope exclusions excluded; logless runs are kept, so they count)'),
        'characters': dict(sorted(characters.items())),
        'strata': [{'character': c, 'act': a, 'kind': k, 'kept': n['kept'], 'dropped': n['dropped'],
                    'scope_excluded': n['scope_excluded']} for (c, a, k), n in sorted(strata.items())],
        'gym_rejections': dict(sorted(rejections.items())),
    }


# ---------------------------------------------------------------- CLI

def write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


# ---------------------------------------------------------------- aggregate statistics

def _quantile(values: Sequence[int], probability: float) -> float:
    """Type-7 (linear interpolation) quantile, matching common statistical software."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError('cannot take a quantile of no values')
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _share(count: int, denominator: int) -> float:
    return round(count / denominator, 6)


def _top_shares(counts: Counter, denominator: int, key: str) -> list[dict]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:15]
    return [{key: item, 'share': _share(count, denominator)} for item, count in ranked]


def aggregate_stats(table: dict, card_types: dict[str, str], *, aliases_sha256: str | None = None) -> dict:
    """Aggregate kept states by character, act and room kind.

    Type mix and upgrade fraction are pooled over card copies. Card and relic shares
    count whether each item appears at least once in a state's deck/relic set. HP is
    averaged as a per-state ``hp / max_hp`` ratio.
    """
    if table.get('schema') != SCHEMA:
        raise ValueError(f'expected {SCHEMA}, got {table.get("schema")!r}')
    groups: dict[tuple[str, int, str], dict] = {}
    for state in table['states']:
        group_key = (state['character'], int(state['act']), state['room_kind'])
        group = groups.setdefault(group_key, {
            'states': 0, 'deck_sizes': [], 'card_count': 0, 'upgraded_count': 0,
            'card_types': Counter(), 'card_decks': Counter(), 'relic_decks': Counter(),
            'potions': 0, 'hp_fractions': [], 'wins': 0, 'potions_assumed_empty': 0,
        })
        group['states'] += 1
        deck = state['deck']
        group['deck_sizes'].append(len(deck))
        deck_cards: set[str] = set()
        for row in deck:
            card = row['card']
            kind = card_types.get(card)
            if kind is None:
                raise ValueError(f'card {card!r} has no type in gym_card_types')
            group['card_count'] += 1
            group['upgraded_count'] += int(row['upgrades'] > 0)
            group['card_types'][kind] += 1
            deck_cards.add(card)
        group['card_decks'].update(deck_cards)
        group['relic_decks'].update(set(state['relics']))
        group['potions'] += len(state['potions'])
        if state['max_hp'] <= 0:
            raise ValueError(f'invalid max_hp {state["max_hp"]!r}')
        group['hp_fractions'].append(state['hp'] / state['max_hp'])
        group['wins'] += int(bool(state['won']))
        group['potions_assumed_empty'] += int('potions_assumed_empty' in state['tags'])

    strata = []
    for (character, act, room_kind), group in sorted(groups.items()):
        count = group['states']
        card_count = group['card_count']
        type_mix = {kind: _share(amount, card_count)
                    for kind, amount in sorted(group['card_types'].items())}
        sizes = group['deck_sizes']
        strata.append({
            'character': character,
            'act': act,
            'room_kind': room_kind,
            'states': count,
            'deck_size': {
                'median': round(_quantile(sizes, 0.5), 6),
                'iqr': round(_quantile(sizes, 0.75) - _quantile(sizes, 0.25), 6),
            },
            'upgrade_fraction': _share(group['upgraded_count'], card_count),
            'card_type_mix': type_mix,
            'top_cards': _top_shares(group['card_decks'], count, 'card'),
            'top_relics': _top_shares(group['relic_decks'], count, 'relic'),
            'mean_potions': round(group['potions'] / count, 6),
            'hp_fraction': round(sum(group['hp_fractions']) / count, 6),
            'win_rate': _share(group['wins'], count),
            'potions_assumed_empty_share': _share(group['potions_assumed_empty'], count),
        })
    result = {'schema': STATS_SCHEMA, 'table_schema': SCHEMA, 'states': len(table['states']), 'strata': strata}
    if aliases_sha256 is not None:
        result['aliases_sha256'] = aliases_sha256
    if 'gym_revision' in table:
        result['gym_revision'] = table['gym_revision']
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('extract', help='rebuild and Gym-validate fight states into a canonical table')
    sub.add_argument('--raw', type=Path, nargs='+', required=True)
    sub.add_argument('--aliases', type=Path, required=True)
    sub.add_argument('--encounter-map', type=Path, default=DEFAULT_ENCOUNTER_MAP)
    sub.add_argument('--output', type=Path, required=True)
    sub.add_argument('--report', type=Path, required=True)
    stats = commands.add_parser('stats', help='aggregate the reconstructed table into per-stratum statistics')
    stats.add_argument('--table', type=Path, required=True)
    stats.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'stats':
        raw = args.table.read_bytes()
        table = json.loads(raw)
        names = load_aliases(DEFAULT_ALIASES)
        result = aggregate_stats(table, dict(names.card_types), aliases_sha256=names.sha256)
        result['table_sha256'] = hashlib.sha256(raw).hexdigest()
        write_bytes(args.output, json.dumps(result, indent=1, sort_keys=True).encode() + b'\n')
        print(json.dumps({'states': result['states'], 'strata': len(result['strata']),
                          'table_sha256': result['table_sha256']}))
        return 0

    names = load_aliases(args.aliases)
    encounter_map = json.loads(args.encounter_map.read_text())['encounters']
    validate, encounters = bind_gym()
    document, report = build(collect_runs(args.raw), names, encounters, encounter_map, validate)
    data = canonical_bytes(document)
    sha = hashlib.sha256(data).hexdigest()
    write_bytes(args.output, data)
    report = {**report, 'table_sha256': sha, 'states_kept': len(document['states']),
              'aliases_sha256': names.sha256, 'gym_revision': document['gym_revision']}
    write_bytes(args.report, json.dumps(report, indent=1, sort_keys=True).encode() + b'\n')
    print(json.dumps({'sha256': sha, 'kept': len(document['states']),
                      'drop_rule': {c: v['drop_rule'] for c, v in report['characters'].items()}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
