"""Per-floor fight states rebuilt from public Slay the Spire run files.

``fight_floors`` lists every fight in a run (from ``damage_taken``);
``state_at`` rebuilds the state the player entered one of them with, or says
why it cannot be rebuilt exactly. Run files are untrusted data: callers parse
them with ``json`` only. Nothing here imports the Gym.

Floor convention: the state entering a fight on floor F applies every change
recorded on floors before F. Potions used or discarded on F were held entering
the fight; everything obtained on F (cards, relics, potions, gold) is a
post-fight reward.

Per-floor list fields (``current_hp_per_floor``, ``potion_use_per_floor``,
``potion_discard_per_floor``, ``potions_obtained_alchemize``,
``potions_obtained_entropic_brew``, ``lesson_learned_per_floor``) are indexed
by floor - 1: entry 0 is floor 1. HP and gold entries are values at the end of
that floor, so the values entering floor F are entry F - 2. This was checked
against real runs: a fight's ``damage_taken`` equals the HP drop between
entries F - 2 and F - 1 (less end-of-combat healing), and each potion use falls
on a fight floor at or after the floor that potion was obtained.

Floor numbering follows the run: ``path_per_floor`` has ``None`` on each boss
chest floor (17, 34 and, after the A20 double boss on 50 and 51, 52), so the act
of floor F is one more than the number of chest floors before it. Secret Portal
runs skip floors and still number this way.

Drop reasons: ``deck_unrecoverable`` (the recorded history does not lead back to
the starter deck, or an unrecorded non-relic change may fall after F),
``untracked_deck_relic`` (an unrecorded relic change may fall after F),
``unmapped_card:<name>``, ``unmapped_relic:<name>``, ``unmapped_potion:<name>``,
``bottle_card_missing``, ``event_fight_reordered``, ``hp_unrecorded`` (floor 1
without a consistent Neow log, or a second fight on one floor) and
``relics_unrecoverable`` (a relic with no record of where it came from).
Tags: ``potions_assumed_empty`` and ``misc_defaulted`` (ruling A).

``relic_stats`` records usage statistics, not relic counters, so the only
counter set here is Neow's Lament's (combats left); every other relic takes the
Gym's entry default.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from spire_learn.neural.standard_names import NameAliases, split_card

CHARACTERS = {'IRONCLAD': 'ironclad', 'THE_SILENT': 'silent', 'DEFECT': 'defect', 'WATCHER': 'watcher'}
STARTER_DECKS = {
    'ironclad': ('Strike_R',) * 5 + ('Defend_R',) * 4 + ('Bash',),
    'silent': ('Strike_G',) * 5 + ('Defend_G',) * 5 + ('Neutralize', 'Survivor'),
    'defect': ('Strike_B',) * 4 + ('Defend_B',) * 4 + ('Zap', 'Dualcast'),
    'watcher': ('Strike_P',) * 4 + ('Defend_P',) * 4 + ('Eruption', 'Vigilance'),
}
STARTER_RELICS = {'ironclad': 'Burning Blood', 'silent': 'Ring of the Snake',
                  'defect': 'Cracked Core', 'watcher': 'PureWater'}
# Boss relics that replace the starter relic when picked.
UPGRADED_STARTERS = {'Black Blood': 'Burning Blood', 'Ring of the Serpent': 'Ring of the Snake',
                     'FrozenCore': 'Cracked Core', 'HolyWater': 'PureWater'}
ROOM_KINDS = {'M': 'monster', 'E': 'elite', 'B': 'boss', '?': 'event'}
BOTTLES = ('Bottled Flame', 'Bottled Lightning', 'Bottled Tornado')
# Ruling A: runs do not record card ``misc``; these take the Gym minimum.
MISC_DEFAULTS = {'genetic_algorithm': 1, 'ritual_dagger': 15}
CURSES = frozenset({'AscendersBane', 'Clumsy', 'CurseOfTheBell', 'Decay', 'Doubt', 'Injury',
                    'Necronomicurse', 'Normality', 'Pain', 'Parasite', 'Pride', 'Regret', 'Shame',
                    'Writhe'})
NEOWS_LAMENT = 'NeowsBlessing'
POTION_BELT = 'Potion Belt'
CURSED_KEY = 'Cursed Key'
# Event-row fields that change the deck, relics, HP or potions. An event fight
# whose event records any of them is dropped: the run does not say whether the
# change came before the fight. Relics and gold an event grants with a fight
# (Dead Adventurer, Mind Bloom, Masked Bandits, ...) are post-fight rewards.
_EVENT_PRE_FIGHT_LISTS = ('cards_obtained', 'cards_removed', 'cards_transformed', 'cards_upgraded',
                          'relics_lost', 'potions_obtained')
_EVENT_PRE_FIGHT_NUMBERS = ('damage_taken', 'damage_healed', 'max_hp_gain', 'max_hp_loss', 'gold_loss')

# ---------------------------------------------------------------- shared rebuild helpers
# Moved from scripts/neural/real_boss_decks.py, which imports them; their
# behaviour is pinned by that script's byte-identical real boss deck table.

# Relics that add, copy, remove, transform or upgrade deck cards when obtained, with
# no record of the change in the run file. real_boss_decks drops a deck rebuilt
# before such a relic was obtained; state_at instead infers their changes from the
# run's whole history (see the deck history section). Boss relics (counted apart):
UNTRACKED_DECK_BOSS_RELICS = {'Astrolabe', "Pandora's Box", 'Empty Cage', 'Calling Bell', 'Tiny House'}
# Shop, chest/combat and event relics. Not listed, because their deck changes are
# recorded or happen only in combat: Orrery (its picks are card_choices), the eggs
# (they only upgrade later cards, whose picks are undone anyway) and Nilry's Codex.
UNTRACKED_DECK_RELICS = {'DollysMirror', 'Whetstone', 'War Paint', 'Necronomicon'}
CARD_EVENT_KEYS = ('cards_obtained', 'cards_removed', 'cards_transformed', 'cards_upgraded')


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


def _undo(deck: list[str], changes: Iterable[tuple[int, int, str, str]]) -> list[tuple[int, str, str]]:
    """Undo ``changes`` on ``deck`` in place, newest first; return the steps that found no card."""
    missed = []
    for at, _order, kind, card in sorted(changes, reverse=True):
        if not isinstance(card, str):
            raise ValueError('card names must be strings')
        if kind == 'restore':
            deck.append(card)
        elif kind == 'remove':
            if not _remove(deck, card):
                missed.append((at, kind, card))
        elif kind == 'remove_if_present':
            _remove(deck, card)
        elif not _downgrade(deck, card):
            missed.append((at, kind, card))
    return missed


def deck_at(run: dict, floor: int) -> tuple[list[str], int]:
    """The deck as it stood on entering ``floor``, and how many undo steps found no card.

    Undoes, newest first: card picks and purchases, shop and campfire purges,
    SMITH upgrades and event card changes on that floor or later.
    """
    deck = [str(c) for c in run['master_deck']]
    return deck, len(_undo(deck, _later_card_changes(run, floor)))


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


# ---------------------------------------------------------------- public types

@dataclass(frozen=True)
class FightFloor:
    """One fight of a run. ``order`` counts earlier fights on the same floor (Colosseum)."""

    floor: int
    enemies: str
    room_kind: str
    act: int
    won: bool
    order: int = 0


@dataclass(frozen=True)
class FightState:
    run_id: str
    character: str
    floor: int
    act: int
    room_kind: str
    enemies: str
    deck: list[dict]
    relics: list[str]
    relic_counters: dict
    bottled_cards: dict
    potions: list[str]
    hp: int
    max_hp: int
    gold: int
    burning: bool
    won: bool
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Drop:
    reason: str
    detail: str


# ---------------------------------------------------------------- fights

def fight_floors(run: dict) -> list[FightFloor]:
    """Every fight in ``damage_taken``, in run order."""
    path = run['path_per_floor']
    rows = run.get('damage_taken') or []
    lost_last = (not run.get('victory') and run.get('killed_by') is not None)
    reached = as_int(run['floor_reached'])
    fights, seen = [], Counter()
    for index, row in enumerate(rows):
        floor, enemies = _floor(row), row['enemies']
        if not isinstance(enemies, str) or not 1 <= floor <= len(path):
            raise ValueError(f'bad fight row on floor {floor}')
        symbol = path[floor - 1]
        if symbol not in ROOM_KINDS:
            raise ValueError(f'floor {floor} fight in a {symbol!r} room')
        act = 1 + sum(1 for s in path[:floor - 1] if s is None)
        lost = (lost_last and index == len(rows) - 1 and floor == reached
                and run.get('killed_by') == enemies)
        fights.append(FightFloor(floor, enemies, ROOM_KINDS[symbol], act, not lost, seen[floor]))
        seen[floor] += 1
    return fights


# ---------------------------------------------------------------- relic history

def _character(run: dict) -> str:
    character = CHARACTERS.get(run.get('character_chosen'))
    if character is None:
        raise ValueError(f'unknown character {run.get("character_chosen")!r}')
    return character


def _boss_relic_floors(run: dict) -> list[int]:
    """Boss relic i is picked on the i-th boss chest floor (``None`` in ``path_per_floor``)."""
    return [i + 1 for i, s in enumerate(run['path_per_floor']) if s is None]


def _relic_sources(run: dict) -> dict[str, int]:
    """Earliest recorded floor each relic name was obtained on (0 for Neow, -1 for the starter)."""
    found: dict[str, int] = {}

    def note(name: object, at: int) -> None:
        if isinstance(name, str) and (name not in found or at < found[name]):
            found[name] = at

    note(STARTER_RELICS[_character(run)], -1)
    for name in (run.get('neow_bonus_log') or {}).get('relicsObtained') or []:
        note(name, 0)
    for row in run.get('relics_obtained') or []:
        note(row['key'], _floor(row))
    for row in run.get('event_choices') or []:
        for name in row.get('relics_obtained') or []:
            note(name, _floor(row))
    for item, at in zip(run.get('items_purchased') or [], run.get('item_purchase_floors') or [], strict=True):
        note(item, as_int(at))
    chests = _boss_relic_floors(run)
    for index, row in enumerate(run.get('boss_relics') or []):
        if row.get('picked'):
            note(row['picked'], chests[index] if index < len(chests) else 17 * (index + 1))
    return found


_NEOW_RELIC_BONUSES = ('THREE_ENEMY_KILL', 'RANDOM_COMMON_RELIC', 'ONE_RARE_RELIC', 'BOSS_RELIC')


def _relic_floors(run: dict) -> tuple[dict[str, int], list[str]]:
    """Obtain floor of every relic held or lost, and the relics with no record of where they came from."""
    sources = _relic_sources(run)
    lost = [name for row in run.get('event_choices') or [] for name in row.get('relics_lost') or []]
    held = [r for r in run['relics'] if isinstance(r, str)]
    unknown = [r for r in dict.fromkeys(held + lost) if r not in sources]
    if run.get('neow_bonus_log') is None and run.get('neow_bonus') in _NEOW_RELIC_BONUSES and len(unknown) == 1:
        # Older runs do not log Neow; the one unrecorded relic is Neow's.
        sources[unknown.pop()] = 0
    return sources, unknown


def _relics_before(run: dict, floor: int, sources: dict[str, int]) -> list[str]:
    """Relic names held entering ``floor``, in acquisition order.

    ``relics`` is the run's final list in acquisition order (an upgraded starter
    takes the starter's slot). Relics lost later (N'loth, Vampires, the Golden
    Idol events) and a starter replaced later are put back where they were held.
    """
    held = [r for r in run['relics'] if sources[r] < floor]
    losses: dict[str, int] = {}
    for row in run.get('event_choices') or []:
        for name in row.get('relics_lost') or []:
            losses.setdefault(name, _floor(row))
    for name, at in sorted(losses.items(), key=lambda item: sources[item[0]]):
        if at >= floor and sources[name] < floor and name not in held:
            index = next((i for i, r in enumerate(held) if sources[r] > sources[name]), len(held))
            held.insert(index, name)
    starter = STARTER_RELICS[_character(run)]
    for upgraded, base in UPGRADED_STARTERS.items():
        if base == starter and sources.get(upgraded, -2) >= floor and starter not in held:
            held.insert(0, starter)
    return held


# ---------------------------------------------------------------- deck history
#
# The deck entering floor F is the final ``master_deck`` with every change on
# floor F or later undone, newest first. The run records most changes; some it
# does not (what Astrolabe, Pandora's Box, Empty Cage, Tiny House, Dolly's Mirror,
# Cursed Key chests, Writhing Mass and Vampires did, and Neow in older runs).
# Undoing the whole recorded history must reach the starter deck. Whatever is
# left over is what the unrecorded effects did; an item only one effect could
# have caused is put back at that effect's floor, so states either side of it
# are exact. Items several effects on different floors could have caused make
# states between the earliest and latest candidate floor unrecoverable, and
# items no effect explains make the whole run unrecoverable.

Change = tuple[int, int, str, str]  # (floor, order on the floor, kind, card)
_INFERRED = 5  # inferred changes come after recorded ones on their floor
_LAST_FLOOR = 10 ** 6


def _recorded_changes(run: dict, sources: dict[str, int]) -> list[Change]:
    """Every recorded deck change, from Neow (floor 0) on."""
    picks = Counter((_floor(row), row['picked']) for row in run.get('card_choices') or []
                    if row['picked'] not in ('SKIP', 'Singing Bowl'))
    changes = []
    for change in _later_card_changes(run, 0):
        at, order, kind, card = change
        if order == 1 and kind == 'remove' and picks[(at, card)]:
            # An event's card choice (The Library) is in card_choices and cards_obtained.
            picks[(at, card)] -= 1
            continue
        changes.append(change)
    log = run.get('neow_bonus_log')
    if isinstance(log, dict):
        changes += [(0, 1, 'remove', c) for c in log.get('cardsObtained') or []]
        changes += [(0, 1, 'restore', c) for c in (log.get('cardsRemoved') or []) + (log.get('cardsTransformed') or [])]
        changes += [(0, 1, 'downgrade', c) for c in log.get('cardsUpgraded') or []]
    for index, cards in enumerate(run.get('lesson_learned_per_floor') or []):
        # Lesson Learned upgrades a card during the fight, before its rewards.
        changes += [(index + 1, -1, 'downgrade', card) for card in cards or []]
    for relic in ('Whetstone', 'War Paint'):
        listed = _relic_stat_list(run, relic)
        if relic in sources and listed is not None:
            changes += [(sources[relic], _INFERRED - 1, 'downgrade', card) for card in listed]
    for relic, curse in (('Calling Bell', 'CurseOfTheBell'), ('Necronomicon', 'Necronomicurse')):
        if relic in sources:
            changes.append((sources[relic], _INFERRED - 1, 'remove', curse))
    return changes


def _replay(master: list[str], changes: list[Change]) -> tuple[list[str], list[int], Counter]:
    """Undo ``changes`` on ``master``, newest first.

    Returns the deck, the indices of changes that found no card, and for each
    'restore' the upgrades its card must have had: run files often name a removed
    card without its upgrades, which shows as an older SMITH finding no upgraded copy.
    """
    deck = [[name, None] for name in master]
    missed, fixes = [], Counter()
    for index in sorted(range(len(changes)), key=lambda i: changes[i], reverse=True):
        _at, _order, kind, card = changes[index]
        if not isinstance(card, str):
            raise ValueError('card names must be strings')
        base, upgrades = split_card(card)
        if kind == 'restore':
            deck.append([card, index])
            continue
        if kind in ('remove', 'remove_if_present'):
            same = [i for i, (name, _o) in enumerate(deck) if split_card(name)[0] == base]
            exact = [i for i in same if deck[i][0] == card]
            if exact or same:
                del deck[(exact or [min(same, key=lambda i: split_card(deck[i][0])[1])])[0]]
            elif kind == 'remove':
                missed.append(index)
            continue
        upgraded = [i for i, (name, _o) in enumerate(deck) if split_card(name)[0] == base and split_card(name)[1] > 0]
        if upgraded:
            i = max(upgraded, key=lambda i: split_card(deck[i][0])[1])
            level = split_card(deck[i][0])[1] - 1
            deck[i][0] = base + (f'+{level}' if level else '')
            continue
        restored = [i for i, (name, origin) in enumerate(deck) if origin is not None and name == base]
        if restored:
            fixes[deck[restored[0]][1]] += 1
        else:
            missed.append(index)
    return [name for name, _o in deck], missed, fixes


def _apply_fixes(changes: list[Change], fixes: Counter) -> list[Change]:
    fixed = list(changes)
    for index, extra in fixes.items():
        at, order, kind, card = fixed[index]
        base, upgrades = split_card(card)
        fixed[index] = (at, order, kind, f'{base}+{upgrades + extra}')
    return fixed


@dataclass
class _Effect:
    """An unrecorded deck effect on ``floor`` and what it can explain."""

    name: str
    floor: int
    add_named: Counter = field(default_factory=Counter)
    add_curse: int = 0
    add_any: int = 0
    remove_any: int = 0
    remove_basic: int = 0
    remove_strikes: bool = False
    upgrade_any: int = 0
    upgrade_named: Counter = field(default_factory=Counter)

    def take(self, item: tuple[str, str], dry: bool = False) -> int | None:
        """Use up room for ``item``; return how specific the match was (0 most), or None."""
        kind, card = item
        base = split_card(card)[0]
        slots = {'add': (('add_named', base, 0), ('add_curse', None, 1) if base in CURSES else None,
                         ('add_any', None, 2)),
                 'remove': (('remove_strikes', None, 1) if _is_basic(card, 'Strike') else None,
                            ('remove_basic', None, 1) if _is_basic(card, 'Strike', 'Defend') else None,
                            ('remove_any', None, 2)),
                 'upgrade': (('upgrade_named', base, 0), ('upgrade_any', None, 2))}[kind]
        for slot in slots:
            if slot is None:
                continue
            attr, key, rank = slot
            value = getattr(self, attr)
            if key is not None:
                if value[key]:
                    if not dry:
                        value[key] -= 1
                    return rank
            elif value is True:
                return rank
            elif value:
                if not dry:
                    setattr(self, attr, value - 1)
                return rank
        return None


def _relic_stat_list(run: dict, relic: str) -> list[str] | None:
    value = (run.get('relic_stats') or {}).get(relic)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def _is_basic(card: str, *kinds: str) -> bool:
    base = split_card(card)[0]
    return any(base.startswith(kind + '_') for kind in kinds)


_RELIC_EFFECTS = frozenset({'Astrolabe', "Pandora's Box", 'Empty Cage', 'Tiny House', 'DollysMirror', 'Whetstone',
                            'War Paint', CURSED_KEY})
_NEOW_DECK = {  # bonus -> (remove_any, add_any, upgrade_any)
    'REMOVE_CARD': (1, 0, 0), 'REMOVE_TWO': (2, 0, 0), 'TRANSFORM_CARD': (1, 1, 0),
    'TRANSFORM_TWO_CARDS': (2, 2, 0), 'UPGRADE_CARD': (0, 0, 1), 'ONE_RANDOM_RARE_CARD': (0, 1, 0),
}


def _effects(run: dict, sources: dict[str, int], changes: list[Change]) -> list[_Effect]:
    effects = []
    if run.get('neow_bonus_log') is None:
        removes, adds, upgrades = _NEOW_DECK.get(run.get('neow_bonus'), (0, 0, 0))
        if run.get('neow_bonus') in ('RANDOM_COLORLESS', 'RANDOM_COLORLESS_2', 'THREE_CARDS', 'THREE_RARE_CARDS') \
                and not any(_floor(row) == 0 for row in run.get('card_choices') or []):
            adds += 1
        neow = _Effect('Neow', 0, remove_any=removes, add_any=adds, upgrade_any=upgrades,
                       add_curse=int(run.get('neow_cost') == 'CURSE'))
        effects.append(neow)
    for relic, at in sources.items():
        listed = _relic_stat_list(run, relic)
        if relic == 'Astrolabe':
            effects.append(_Effect(relic, at, remove_any=3, add_named=Counter(listed or ()),
                                   add_any=0 if listed is not None else 3))
        elif relic == "Pandora's Box":
            effects.append(_Effect(relic, at, add_named=Counter(listed or ()),
                                   remove_basic=len(listed) if listed is not None else 0))
        elif relic == 'Empty Cage':
            effects.append(_Effect(relic, at, remove_any=2))
        elif relic == 'Tiny House':
            effects.append(_Effect(relic, at, upgrade_any=1))
        elif relic == 'DollysMirror':
            effects.append(_Effect(relic, at, add_any=1))
        elif relic in ('Whetstone', 'War Paint') and listed is None:
            effects.append(_Effect(relic, at, upgrade_any=2))
    key = sources.get(CURSED_KEY)
    curses = (run.get('relic_stats') or {}).get(CURSED_KEY)
    if key is not None and curses != 0:
        for index, symbol in enumerate(run['path_per_floor']):
            if symbol == 'T' and index + 1 > key:
                effects.append(_Effect(CURSED_KEY, index + 1, add_curse=1))
    if 'lesson_learned_per_floor' not in run:
        # Older runs do not log Lesson Learned's permanent upgrades: any fight after
        # the card was first obtained may have upgraded any card.
        obtained = [at for at, _o, kind, card in changes if kind == 'remove' and split_card(card)[0] == 'LessonLearned']
        if obtained or any(split_card(c)[0] == 'LessonLearned' for c in run['master_deck']):
            start = min(obtained, default=0)
            for row in run.get('damage_taken') or []:
                if _floor(row) > start:
                    effects.append(_Effect('Lesson Learned', _floor(row), upgrade_any=999))
    for row in run.get('damage_taken') or []:
        if row.get('enemies') == 'Writhing Mass':
            effects.append(_Effect('Writhing Mass', _floor(row), add_named=Counter(Parasite=1)))
    for row in run.get('event_choices') or []:
        # Vampires replaces every Strike with Bites but records only the Bites.
        if row.get('event_name') == 'Vampires' and str(row.get('player_choice', '')).startswith('Became') \
                and not row.get('cards_removed'):
            effects.append(_Effect('Vampires', _floor(row), remove_strikes=True))
    return effects


@dataclass(frozen=True)
class _DeckHistory:
    changes: list[Change]
    omamori: bool
    unexplained: tuple[str, ...]
    windows: tuple[tuple[int, int, tuple[str, ...]], ...]  # (after, through, effects): states in (after, through] drop


def _residual(run: dict, deck: list[str], missed: list[int], changes: list[Change],
              omamori: bool) -> list[tuple[str, str, int, int]]:
    """What undoing every change leaves unexplained, as (kind, card, earliest, latest floor) items."""
    expected = list(STARTER_DECKS[_character(run)])
    if as_int(run.get('ascension_level', 0)) >= 10:
        expected.append('AscendersBane')
    extra = Counter(deck) - Counter(expected)
    missing = Counter(expected) - Counter(deck)
    items: list[tuple[str, str, int]] = []
    for card in sorted(extra.elements()):
        base, upgrades = split_card(card)
        lower = next((m for m in sorted(+missing) if split_card(m)[0] == base and split_card(m)[1] < upgrades), None)
        if lower is not None:
            missing[lower] -= 1
            extra[card] -= 1
            items.append(('upgrade', card, 0, _LAST_FLOOR))
    items += [('add', card, 0, _LAST_FLOOR) for card in sorted((+extra).elements())]
    removed = [('remove', card, 0, _LAST_FLOOR) for card in sorted((+missing).elements())]
    for index in missed:
        at, _order, kind, card = changes[index]
        base = split_card(card)[0]
        if omamori and base in CURSES:
            continue  # Omamori negated it: the curse never entered the deck.
        if kind == 'downgrade':
            # An upgraded card that later left unrecorded: it is one of the removed
            # cards of that base if there is one (the replay fixes its upgrades).
            if any(split_card(c)[0] == base for _k, c, _a, _b in removed):
                continue
            card = base
        removed.append(('remove', card, at, _LAST_FLOOR))
    return items + removed


def _deck_history(run: dict, sources: dict[str, int]) -> _DeckHistory:
    master = [str(c) for c in run['master_deck']]
    omamori = 'Omamori' in sources
    changes = _recorded_changes(run, sources)
    deck, missed, fixes = _replay(master, changes)
    changes = _apply_fixes(changes, fixes)
    deck, missed, _fixes = _replay(master, changes)
    items = _residual(run, deck, missed, changes, omamori)
    effects = _effects(run, sources, changes)
    inferred, windows, unexplained = [], [], []
    # Most constrained first: items that can only come from later effects.
    for kind, card, earliest, latest in sorted(items, key=lambda item: (-item[2], item[3], item[0], item[1])):
        item = (kind, card)
        candidates = [(e.take(item, dry=True), e.floor, i) for i, e in enumerate(effects)
                      if earliest <= e.floor <= latest and e.take(item, dry=True) is not None]
        if kind == 'add' and candidates:
            # The card must have been in the deck for every recorded change to it
            # (a later purge or SMITH): keep only floors whose undo finds no new gap.
            fits = {f: len(_replay(master, changes + [(f, _INFERRED, 'remove', card)])[1]) <= len(missed)
                    for f in {f for _r, f, _i in candidates}}
            candidates = [c for c in candidates if fits[c[1]]]
        if not candidates:
            unexplained.append(f'{"+" if kind == "add" else "-" if kind == "remove" else "^"}{card}')
            continue
        _rank, _floor_at, chosen = min(candidates)
        effects[chosen].take(item)
        first = min(f for _r, f, _i in candidates)
        last = max(f for _r, f, _i in candidates)
        if first != last:
            windows.append((first, last, tuple(sorted({effects[i].name for _r, _f, i in candidates}))))
        undo = {'add': 'remove', 'remove': 'restore', 'upgrade': 'downgrade'}[kind]
        inferred.append((first, _INFERRED, undo, card))
    if unexplained:
        return _DeckHistory(changes, omamori, tuple(unexplained), ())
    changes = changes + inferred
    deck, missed, fixes = _replay(master, changes)
    changes = _apply_fixes(changes, fixes)
    deck, missed, _fixes = _replay(master, changes)
    left = _residual(run, deck, missed, changes, omamori)
    if left:
        return _DeckHistory(changes, omamori, tuple(f'{k}:{c}' for k, c, _a, _b in left), ())
    return _DeckHistory(changes, omamori, (), tuple(windows))


def _deck_before(run: dict, floor: int, history: _DeckHistory) -> tuple[list[str], list[Change]]:
    """Deck names entering ``floor`` and the undo steps that found no card."""
    changes = [c for c in history.changes if c[0] >= floor]
    deck, missed, fixes = _replay([str(c) for c in run['master_deck']], changes)
    missed = [i for i in missed if not (history.omamori and split_card(changes[i][3])[0] in CURSES)]
    return deck, [changes[i] for i in missed] + [changes[i] for i in fixes]


# ---------------------------------------------------------------- potions

def _per_floor(run: dict, key: str) -> list[list]:
    value = run.get(key)
    return value if isinstance(value, list) else []


def _potions_before(run: dict, floor: int, names: NameAliases, sources: dict[str, int]) -> list[str] | None:
    """Potion names held entering ``floor``, or None when the logs cannot say exactly."""
    gained: dict[int, list[str]] = {}
    for row in run.get('potions_obtained') or []:
        gained.setdefault(_floor(row), []).append(row['key'])
    for row in run.get('event_choices') or []:
        gained.setdefault(_floor(row), []).extend(row.get('potions_obtained') or [])
    for item, at in zip(run.get('items_purchased') or [], run.get('item_purchase_floors') or [], strict=True):
        if isinstance(item, str) and names.potion(item) is not None:
            gained.setdefault(as_int(at), []).append(item)
    uses = _per_floor(run, 'potion_use_per_floor')
    discards = run.get('potion_discard_per_floor')
    if not uses or not isinstance(discards, list):
        # No potion logs: exact only if no potion was obtained before this floor.
        return None if any(gained.get(at) for at in range(0, floor)) else []
    made: dict[int, list[str]] = {}
    for key in ('potions_obtained_alchemize', 'potions_obtained_entropic_brew'):
        for index, potions in enumerate(_per_floor(run, key)):
            made.setdefault(index + 1, []).extend(potions or [])
    belt_from = sources.get(POTION_BELT)
    belt_until = min((_floor(row) for row in run.get('event_choices') or []
                      if POTION_BELT in (row.get('relics_lost') or [])), default=None)
    slots_base = 2 if as_int(run.get('ascension_level', 0)) >= 11 else 3
    held = list(gained.get(0, []))
    entering = None
    # Replay the whole run: an inconsistency anywhere means some potion went unrecorded,
    # and its floor is unknown, so no state of the run has exact potions.
    for at in range(1, max(len(uses), floor) + 1):
        if at == floor:
            entering = list(held)
        available = held + gained.get(at, []) + made.get(at, [])
        out = list(uses[at - 1] if at - 1 < len(uses) else []) + list(
            discards[at - 1] if at - 1 < len(discards) else [])
        for potion in out:
            if potion not in available:
                return None
            available.remove(potion)
        held = available
        belt = belt_from is not None and belt_from <= at and (belt_until is None or at < belt_until)
        if len(held) > slots_base + 2 * belt:
            return None
    return entering


# ---------------------------------------------------------------- states

def _event_fight_reordered(run: dict, floor: int) -> str | None:
    for row in run.get('event_choices') or []:
        if _floor(row) != floor:
            continue
        if any(row.get(key) for key in _EVENT_PRE_FIGHT_LISTS) or any(
                row.get(key) not in (None, 0) for key in _EVENT_PRE_FIGHT_NUMBERS):
            return str(row.get('event_name'))
    return None


# A20 start: max HP after Ascension 14, current HP 90% of it (Ascension 6), 99 gold.
_A20_MAX_HP = {'ironclad': 75, 'silent': 66, 'defect': 71, 'watcher': 68}
_START_GOLD = 99


def _resources_entering(run: dict, floor: int, character: str) -> tuple[int, int, int] | None:
    """(HP, max HP, gold) entering ``floor``; None when the run does not pin them."""
    if floor >= 2:
        hp = as_int(run['current_hp_per_floor'][floor - 2])
        max_hp = as_int(run['max_hp_per_floor'][floor - 2])
        if not 0 < hp <= max_hp:
            raise ValueError(f'invalid HP entering floor {floor}')
        return hp, max_hp, as_int(run['gold_per_floor'][floor - 2])
    # Floor 1 is entered straight from Neow, which only newer runs log. Derive the
    # HP from the A20 start and Neow's log, and keep it only if the floor-1 fight's
    # damage (less Burning Blood's heal) lands exactly on the HP recorded after it.
    log = run.get('neow_bonus_log')
    fights = [row for row in run.get('damage_taken') or [] if _floor(row) == 1]
    if as_int(run.get('ascension_level', 0)) != 20 or not isinstance(log, dict) or len(fights) != 1:
        return None
    max_hp = _A20_MAX_HP[character] + as_int(log.get('maxHpGained', 0))
    hp = int(_A20_MAX_HP[character] * 0.9 + 0.5) + as_int(log.get('maxHpGained', 0))
    max_hp -= as_int(log.get('maxHpLost', 0))
    hp = min(hp, max_hp) - as_int(log.get('damageTaken', 0))
    gold = _START_GOLD + as_int(log.get('goldGained', 0)) - as_int(log.get('goldLost', 0))
    after = hp - as_int(fights[0]['damage'])
    if 'Burning Blood' in run['relics'][:1] or (character == 'ironclad' and 'Black Blood' in run['relics'][:1]):
        after = min(after + 6, max_hp)
    if not 0 < hp <= max_hp or max_hp != as_int(run['max_hp_per_floor'][0]) \
            or after != as_int(run['current_hp_per_floor'][0]):
        return None
    return hp, max_hp, gold


def _neows_lament(run: dict, floor: int) -> int:
    """Neow's Lament counts down three combats, then reads -2 (spent)."""
    fought = sum(1 for row in run.get('damage_taken') or [] if _floor(row) < floor)
    return 3 - fought if fought < 3 else -2


_last_run: list = [None, None]


def _run_history(run: dict) -> tuple[dict[str, int], list[str], _DeckHistory | None]:
    """Relic floors and deck history of ``run``; the last run's are reused (runs are not mutated)."""
    if _last_run[0] is not run:
        sources, unknown = _relic_floors(run)
        _last_run[:] = [run, (sources, unknown, None if unknown else _deck_history(run, sources))]
    return _last_run[1]


def state_at(run: dict, fight: FightFloor, names: NameAliases) -> FightState | Drop:
    """The state entering ``fight``, or why it cannot be rebuilt exactly.

    ``run`` must not be mutated between calls: the per-run history is computed
    once and reused while the same run object is passed.
    """
    at = fight.floor
    character = _character(run)
    if fight.order:
        # HP is recorded at floor ends, not between two fights on one floor (Colosseum).
        return Drop('hp_unrecorded', f'floor {at} fight {fight.order}')
    resources = _resources_entering(run, at, character)
    if resources is None:
        return Drop('hp_unrecorded', f'floor {at}')
    if fight.room_kind == 'event':
        event = _event_fight_reordered(run, at)
        if event is not None:
            return Drop('event_fight_reordered', event)
    sources, unknown, history = _run_history(run)
    if unknown:
        return Drop('relics_unrecoverable', ', '.join(unknown))
    if history.unexplained:
        return Drop('deck_unrecoverable', ' '.join(history.unexplained))
    for after, through, effects in history.windows:
        if after < at <= through:
            reason = 'untracked_deck_relic' if _RELIC_EFFECTS & set(effects) else 'deck_unrecoverable'
            return Drop(reason, f'{" or ".join(effects)} changed the deck unrecorded on floors {after + 1}-{through}')
    deck_names, missed = _deck_before(run, at, history)
    if missed:
        return Drop('deck_unrecoverable', f'no card to undo {missed[0][2]} {missed[0][3]} (floor {missed[0][0]})')
    deck, tags = [], []
    for name in deck_names:
        mapped = names.card(name)
        if mapped is None:
            return Drop(f'unmapped_card:{split_card(name)[0]}', name)
        card, upgrades = mapped
        deck.append({'card': card, 'upgrades': upgrades, 'misc': MISC_DEFAULTS.get(card, 0)})
        if card in MISC_DEFAULTS and 'misc_defaulted' not in tags:
            tags.append('misc_defaulted')
    deck.sort(key=lambda row: (row['card'], row['upgrades']))
    relic_names = _relics_before(run, at, sources)
    relics, counters = [], {}
    for name in relic_names:
        relic = names.relic(name)
        if relic is None:
            return Drop(f'unmapped_relic:{name}', name)
        relics.append(relic)
        if name == NEOWS_LAMENT:
            counters[relic] = _neows_lament(run, at)
    bottled = {}
    for bottle in BOTTLES:
        if bottle not in relic_names:
            continue
        card_name = (run.get('relic_stats') or {}).get(bottle)
        mapped = names.card(card_name) if isinstance(card_name, str) else None
        # relic_stats names the bottled card without upgrades; it must be one exact deck row.
        rows = {(r['card'], r['upgrades'], r['misc']) for r in deck if mapped and r['card'] == mapped[0]}
        if len(rows) != 1:
            return Drop('bottle_card_missing', f'{bottle}: {card_name!r} matches {len(rows)} deck rows')
        card, upgrades, misc = rows.pop()
        bottled[names.relic(bottle)] = {'card': card, 'upgrades': upgrades, 'misc': misc}
    potion_names = _potions_before(run, at, names, sources)
    potions = []
    if potion_names is None:
        tags.append('potions_assumed_empty')
    else:
        for name in potion_names:
            potion = names.potion(name) if isinstance(name, str) else None
            if potion is None:
                return Drop(f'unmapped_potion:{name}', str(name))
            potions.append(potion)
    hp, max_hp, gold = resources
    key_floor = run.get('green_key_taken_log')
    burning = fight.room_kind == 'elite' and key_floor is not None and as_int(key_floor) == at
    return FightState(
        run_id=str(as_int(run['timestamp'])), character=character, floor=at, act=fight.act,
        room_kind=fight.room_kind, enemies=fight.enemies, deck=deck, relics=relics,
        relic_counters=counters, bottled_cards=bottled, potions=potions, hp=hp, max_hp=max_hp,
        gold=gold, burning=burning, won=fight.won, tags=tags)
