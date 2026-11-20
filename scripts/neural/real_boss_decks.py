"""Real act 2 and act 3 boss decks from Baalorlord's public Slay the Spire run files.

``fetch`` politely downloads profile-0 Ironclad run files (1 request per second);
``extract`` rebuilds each run's deck as it stood when the act's first boss floor
was entered and maps it to the curriculum support. Run files are untrusted data:
they are parsed only with ``json`` and never executed. The downloads and the
derived table stay local (under the git-ignored ``artifacts/``); the curriculum
v2 recipe pins the table by SHA256.

    PYTHONPATH=src python3.12 scripts/neural/real_boss_decks.py fetch \\
        --output artifacts/development/neural-combat/baalorlord-runs-20261119
    PYTHONPATH=src python3.12 scripts/neural/real_boss_decks.py extract \\
        --raw artifacts/development/neural-combat/baalorlord-runs-20261119/raw \\
        --support configs/neural/curriculum-v1/support.json \\
        --output artifacts/development/neural-combat/real-boss-decks/real-boss-decks.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

if __package__ in (None, ''):
    _repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_repo / 'src'), str(_repo)]

from spire_learn.neural.standard_names import normalize  # noqa: E402,F401
from spire_learn.neural.scenario_support import (  # noqa: E402
    ScenarioSupport, canonical_bytes, load_scenario_support,
)

SCHEMA = 'spire-learn-real-boss-decks-v1'
SOURCE = 'baalorlord.tv profile 0 (Slay the Spire)'
BASE_URL = 'https://baalorlord.tv'
USER_AGENT = 'spire-learn research crawler (https://github.com/klottick/spire-learn; 1 request/s)'
REQUEST_INTERVAL = 1.0
# Exact enemy names from the run files, plus the long forms; both map to the support encounter.
ACT2_BOSSES = {'Champ': 'champ', 'The Champ': 'champ', 'Collector': 'collector',
               'The Collector': 'collector', 'Automaton': 'bronze_automaton',
               'Bronze Automaton': 'bronze_automaton'}
ACT3_BOSSES = {'Awakened One': 'awakened_one', 'Time Eater': 'time_eater',
               'Donu and Deca': 'donu_and_deca'}
# Fight floor -> (boss names, the act's first boss floor, where the deck is rebuilt).
BOSS_FLOORS = {33: (ACT2_BOSSES, 33), 50: (ACT3_BOSSES, 50), 51: (ACT3_BOSSES, 50)}
CARD_ALIASES = {'Strike_R': 'strike', 'Defend_R': 'defend', 'Ghostly': 'ghostly_armor'}
RELIC_ALIASES = {'Paper Frog': 'paper_phrog', 'Champion Belt': 'champions_belt',
                 'Mark of the Bloom': 'mark_of_bloom'}
# Neow's Lament only lasts three combats, so it is spent long before any boss floor.
SPENT_RELICS = {'NeowsBlessing', "Neow's Lament"}
# Relics that add, copy, remove, transform or upgrade deck cards when obtained, with
# no record of the change in the run file. A deck rebuilt before such a relic was
# obtained cannot be recovered, so it is dropped. Boss relics (counted apart):
UNTRACKED_DECK_BOSS_RELICS = {'Astrolabe', "Pandora's Box", 'Empty Cage', 'Calling Bell', 'Tiny House'}
# Shop, chest/combat and event relics. Not listed, because their deck changes are
# recorded or happen only in combat: Orrery (its picks are card_choices), the eggs
# (they only upgrade later cards, whose picks are undone anyway) and Nilry's Codex.
UNTRACKED_DECK_RELICS = {'DollysMirror', 'Whetstone', 'War Paint', 'Necronomicon'}
CARD_EVENT_KEYS = ('cards_obtained', 'cards_removed', 'cards_transformed', 'cards_upgraded')


def split_card(name: str) -> tuple[str, int]:
    """'Searing Blow+3' -> ('Searing Blow', 3)."""
    match = re.fullmatch(r'(.*?)(?:\+(\d+))?', name)
    return match.group(1), int(match.group(2) or 0)


def name_maps(support: ScenarioSupport) -> tuple[dict[str, str], dict[str, str]]:
    """Normalized run-file names -> support card and relic IDs."""
    cards = {normalize(c): c for c in support.cards}
    cards.update({normalize(k): v for k, v in CARD_ALIASES.items()})
    relics = {normalize(r): r for r in support.relics if r != 'neows_lament'}
    relics.update({normalize(k): v for k, v in RELIC_ALIASES.items()})
    return cards, relics


# ---------------------------------------------------------------- rebuild

def as_int(value: object) -> int:
    """An integer field; some run files store every number as an integral float."""
    if type(value) is float and value.is_integer():
        return int(value)
    if type(value) is not int:
        raise ValueError(f'expected an integer, got {value!r}')
    return value


def _floor(row: dict) -> int:
    return as_int(row['floor'])


def _remove(deck: list[str], name: str) -> bool:
    """Remove the exact card, else a copy of the same base card with the fewest upgrades."""
    if name in deck:
        deck.remove(name)
        return True
    base = split_card(name)[0]
    same = [i for i, c in enumerate(deck) if split_card(c)[0] == base]
    if not same:
        return False
    del deck[min(same, key=lambda i: split_card(deck[i])[1])]
    return True


def _downgrade(deck: list[str], name: str) -> bool:
    """Undo one upgrade on the most-upgraded copy of a card."""
    base = split_card(name)[0]
    same = [i for i, c in enumerate(deck) if split_card(c)[0] == base and split_card(c)[1] > 0]
    if not same:
        return False
    index = max(same, key=lambda i: split_card(deck[i])[1])
    upgrades = split_card(deck[index])[1] - 1
    deck[index] = base + (f'+{upgrades}' if upgrades else '')
    return True


def _later_card_changes(run: dict, floor: int) -> list[tuple[int, int, str, str]]:
    """Deck changes on ``floor`` or later as (floor, order, kind, card) tuples."""
    changes = []
    for row in run.get('card_choices') or []:
        if _floor(row) >= floor and row['picked'] not in ('SKIP', 'Singing Bowl'):
            changes.append((_floor(row), 0, 'remove', row['picked']))
    for row in run.get('event_choices') or []:
        if _floor(row) >= floor:
            for card in row.get('cards_obtained') or []:
                changes.append((_floor(row), 1, 'remove', card))
            for card in (row.get('cards_removed') or []) + (row.get('cards_transformed') or []):
                changes.append((_floor(row), 1, 'restore', card))
            for card in row.get('cards_upgraded') or []:
                changes.append((_floor(row), 1, 'downgrade', card))
    purchases = run.get('items_purchased') or []
    purchase_floors = run.get('item_purchase_floors') or []
    for item, at in zip(purchases, map(as_int, purchase_floors), strict=True):
        if at >= floor:
            # Relics and potions are bought too; they are simply absent from the deck.
            changes.append((at, 2, 'remove_if_present', item))
    purges = run.get('items_purged') or []
    purge_floors = run.get('items_purged_floors') or []
    for item, at in zip(purges, map(as_int, purge_floors), strict=True):
        if at >= floor:
            changes.append((at, 3, 'restore', item))
    for row in run.get('campfire_choices') or []:
        if _floor(row) >= floor and row.get('data'):
            if row.get('key') == 'SMITH':
                changes.append((_floor(row), 4, 'downgrade', row['data']))
            elif row.get('key') == 'PURGE':
                changes.append((_floor(row), 4, 'restore', row['data']))
    return changes


def deck_at(run: dict, floor: int) -> tuple[list[str], int]:
    """The deck as it stood on entering ``floor``, and how many undo steps found no card.

    Undoes, newest first: card picks and purchases, shop and campfire purges,
    SMITH upgrades and event card changes on that floor or later.
    """
    deck = [str(c) for c in run['master_deck']]
    misses = 0
    for _at, _order, kind, card in sorted(_later_card_changes(run, floor), reverse=True):
        if not isinstance(card, str):
            raise ValueError('card names must be strings')
        if kind == 'restore':
            deck.append(card)
        elif kind == 'remove':
            misses += not _remove(deck, card)
        elif kind == 'remove_if_present':
            _remove(deck, card)
        else:
            misses += not _downgrade(deck, card)
    return deck, misses


def _later_relics(run: dict, floor: int) -> set[str]:
    later = {r['key'] for r in run.get('relics_obtained') or [] if _floor(r) >= floor}
    for row in run.get('event_choices') or []:
        if _floor(row) >= floor:
            later.update(row.get('relics_obtained') or [])
    for item, at in zip(run.get('items_purchased') or [], map(as_int, run.get('item_purchase_floors') or []),
                        strict=True):
        if at >= floor:
            later.add(item)
    # Boss relic i is chosen after the act i+1 boss, on floor 17, 34, 51.
    for index, row in enumerate(run.get('boss_relics') or []):
        if 17 * (index + 1) >= floor and row.get('picked'):
            later.add(row['picked'])
    return later


def relics_at(run: dict, floor: int) -> list[str]:
    """Relic names held on entering ``floor``."""
    later = _later_relics(run, floor)
    return [r for r in run['relics'] if r not in later]


def _untracked_deck_relic(run: dict, floor: int) -> bool:
    return any(17 * (index + 1) >= floor and row.get('picked') in UNTRACKED_DECK_BOSS_RELICS
               for index, row in enumerate(run.get('boss_relics') or []))


def _deck_changing_relic(run: dict, floor: int) -> bool:
    """A non-boss relic with unrecorded deck changes, obtained on ``floor`` or later."""
    return bool(_later_relics(run, floor) & UNTRACKED_DECK_RELICS)


# ---------------------------------------------------------------- extract

def _run_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in Path(raw_dir).iterdir()
                  if p.suffix == '.json' and p.name != 'listing.json' and p.is_file())


def extract(raw_dir: Path, support: ScenarioSupport) -> tuple[dict, dict]:
    """The real-deck document and extraction statistics for every run file in ``raw_dir``."""
    card_map, relic_map = name_maps(support)
    rows = []
    encounters: dict[str, list[int]] = {}
    unmatched, unmapped_cards, unmapped_relics = Counter(), Counter(), Counter()
    stats = Counter()
    for path in _run_files(raw_dir):
        run_id = path.stem
        try:
            run = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(run, dict):
                raise ValueError('run root must be an object')
            if run.get('character_chosen') != 'IRONCLAD':
                stats['skipped_other_character'] += 1
                continue
            fights = []
            for fight in run.get('damage_taken') or []:
                at, name = _floor(fight), fight['enemies']
                if at not in BOSS_FLOORS:
                    continue
                names, deck_floor = BOSS_FLOORS[at]
                if name not in names:
                    unmatched[f'{at}:{name}'] += 1
                    continue
                fights.append((at, names[name], deck_floor, name))
            run_rows = []
            for at, encounter, deck_floor, name in fights:
                if _untracked_deck_relic(run, deck_floor):
                    stats['excluded_untracked_boss_relic'] += 1
                    continue
                if _deck_changing_relic(run, deck_floor):
                    stats['excluded_deck_changing_relic'] += 1
                    continue
                deck, misses = deck_at(run, deck_floor)
                stats['rebuild_misses'] += misses
                mapped, dropped = [], 0
                for card in deck:
                    base, upgrades = split_card(card)
                    card_id = card_map.get(normalize(base))
                    if card_id is None:
                        unmapped_cards[base] += 1
                        dropped += 1
                    else:
                        mapped.append({'card': card_id, 'upgrades': upgrades})
                relics = []
                for relic in relics_at(run, deck_floor):
                    if relic in SPENT_RELICS:
                        continue
                    relic_id = relic_map.get(normalize(relic))
                    if relic_id is None:
                        unmapped_relics[relic] += 1
                    elif relic_id not in relics:
                        relics.append(relic_id)
                current = as_int(run['current_hp_per_floor'][at - 2])
                maximum = as_int(run['max_hp_per_floor'][at - 2])
                if not 0 < current <= maximum:
                    raise ValueError(f'invalid HP before floor {at}')
                won = not (run.get('killed_by') == name and as_int(run.get('floor_reached')) == at)
                ascension = as_int(run.get('ascension_level'))
                row = {'run_id': run_id, 'encounter': encounter, 'floor': at, 'won': won,
                       'ascension': ascension,
                       'deck': sorted(mapped, key=lambda r: (r['card'], r['upgrades'])),
                       'relics': sorted(relics), 'hp_fraction': current / maximum,
                       'dropped_cards': dropped}
                if not _valid_entry(support, row):
                    stats['excluded_invalid'] += 1
                    continue
                run_rows.append(row)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, UnicodeError) as exc:
            print(f'skip malformed run {run_id}: {type(exc).__name__}: {exc}', file=sys.stderr)
            stats['malformed_runs'] += 1
            continue
        for row in run_rows:
            counts = encounters.setdefault(row['encounter'], [0, 0])
            counts[0] += row['won']
            counts[1] += 1
        rows.extend(run_rows)
    rows.sort(key=lambda r: (r['encounter'], r['run_id'], r['floor']))
    document = {'schema': SCHEMA, 'source': SOURCE, 'decks': rows}
    summary = {
        'encounters': dict(sorted(encounters.items())),
        'unmatched_boss_names': dict(sorted(unmatched.items())),
        'unmapped_cards': dict(unmapped_cards.most_common()),
        'unmapped_relics': dict(unmapped_relics.most_common()),
        **{key: stats[key] for key in ('skipped_other_character', 'excluded_invalid',
                                       'excluded_untracked_boss_relic',
                                       'excluded_deck_changing_relic', 'rebuild_misses',
                                       'malformed_runs')},
        'decks': len(rows),
    }
    return document, summary


def _valid_entry(support: ScenarioSupport, row: dict) -> bool:
    relics = list(row['relics'])
    if not {'burning_blood', 'black_blood'} & set(relics):
        relics.append('burning_blood')
    try:
        support.validate_entry({'id': 'probe', 'encounter': row['encounter'], 'hp': 1, 'max_hp': 1,
                                'deck': row['deck'], 'relics': relics, 'potions': [None] * 3})
    except ValueError:
        return False
    return True


def write_document(document: dict, path: Path) -> str:
    """Write canonical JSON (sorted keys, no whitespace, no newline); return its SHA256."""
    data = canonical_bytes(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- fetch

def _get(url: str, urlopen, sleep) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    finally:
        sleep(REQUEST_INTERVAL)


def fetch(output: Path, *, urlopen=urllib.request.urlopen, sleep=time.sleep, max_pages: int = 200) -> dict:
    """Crawl the profile-0 run listing, then download raw Ironclad runs not already present."""
    output = Path(output)
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    listing: dict[str, list[str]] = {}
    for page in range(1, max_pages + 1):
        html = _get(f'{BASE_URL}/profile/0/runs/{page}', urlopen, sleep).decode('utf-8', 'replace')
        found = dict(re.findall(r'<a href="/runs/(\d+)"\s+class="([^"]*)"', html))
        new = {rid: cls.split() for rid, cls in found.items() if rid not in listing}
        if not new:
            break
        listing.update(new)
    write_document(listing, output / 'listing.json')
    ironclad = sorted((rid for rid, cls in listing.items() if cls[:1] == ['ironclad']), key=int)
    stats = Counter(listed=len(listing), ironclad=len(ironclad), fetched=0, skipped=0, failed=0)
    for rid in ironclad:
        path = raw / f'{rid}.json'
        if path.exists():
            stats['skipped'] += 1
            continue
        try:
            data = _get(f'{BASE_URL}/runs/{rid}/raw', urlopen, sleep)
            if not isinstance(json.loads(data), dict):
                raise ValueError('run root must be an object')
        except (OSError, ValueError) as exc:
            print(f'fail {rid}: {type(exc).__name__}', file=sys.stderr)
            stats['failed'] += 1
            continue
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, path)
        stats['fetched'] += 1
    return dict(stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    fetch_parser = commands.add_parser('fetch', help='download profile-0 Ironclad run files')
    fetch_parser.add_argument('--output', type=Path, required=True)
    extract_parser = commands.add_parser('extract', help='rebuild boss decks into a canonical table')
    extract_parser.add_argument('--raw', type=Path, required=True)
    extract_parser.add_argument('--support', type=Path, required=True)
    extract_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'fetch':
        print(json.dumps(fetch(args.output), sort_keys=True))
        return 0
    document, stats = extract(args.raw, load_scenario_support(args.support))
    sha256 = write_document(document, args.output)
    print(json.dumps({**stats, 'output': str(args.output), 'sha256': sha256}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
