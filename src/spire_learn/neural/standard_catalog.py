"""Spire Gym ``standard-combat-v1`` catalog validation, shared by extraction and generation.

Importing this module never imports the Gym; ``bind_gym`` imports it on call, so it
runs only in the pinned runtime. Validators are plain callables so tests inject fakes.
"""
from __future__ import annotations

from typing import Callable, Sequence

CATALOG_SCHEMA = 'spire-gym-standard-combat-catalog-v1'
PROFILE = 'standard-combat-v1'
MAX_CATALOG = 256
ASCENSION = 20

Validate = Callable[[dict], None]


class CatalogRejected(Exception):
    """The Gym refused a catalog as invalid input; the only error ``validate_entries`` bisects on.

    ``bind_gym`` translates the Gym's ``RunInputError`` into this. Any other error (a native
    fault, MemoryError, a validator bug) is a fault, not a rejection, and propagates.
    """


def catalog(entries: list[dict]) -> dict:
    """The Standard combat catalog document holding ``entries``."""
    return {'catalog_schema_id': CATALOG_SCHEMA, 'profile_id': PROFILE, 'entries': entries}


def potion_slots(relics: Sequence[str]) -> int:
    """Standard slots at A20: 2 (3 below A11), plus 2 with Potion Belt."""
    return 2 + (2 if 'potion_belt' in relics else 0)


def validate_entries(entries: list[dict], validate: Validate) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Split entries into (accepted, [(rejected, reason)]), validating catalogs of at most 256.

    A rejected catalog is bisected until each bad entry stands alone; the reason is the
    Gym's error for that entry. An entry accepted in a catalog is never revalidated. Only
    ``CatalogRejected`` is bisected; any other exception propagates at once, unretried.
    """
    accepted: list[dict] = []
    rejected: list[tuple[dict, str]] = []

    def check(chunk: list[dict]) -> None:
        try:
            validate(catalog(chunk))
        except CatalogRejected as error:
            if len(chunk) == 1:
                rejected.append((chunk[0], f'gym_rejected:{str(error)[:200]}'))
                return
            middle = len(chunk) // 2
            check(chunk[:middle])
            check(chunk[middle:])
        else:
            accepted.extend(chunk)

    for start in range(0, len(entries), MAX_CATALOG):
        check(entries[start:start + MAX_CATALOG])
    return accepted, rejected


def bind_gym() -> tuple[Validate, list[dict]]:
    """The Gym's catalog validator and encounter listing (needs the pinned runtime)."""
    from spire_lab import _native
    from spire_lab.gymnasium import StandardCombatEnv
    probe = {'id': 'probe', 'character': 'ironclad', 'ascension': ASCENSION, 'encounter': 'cultist', 'hp': 70,
             'max_hp': 80, 'gold': 0, 'deck': [{'card': 'strike', 'upgrades': 0, 'misc': 0}],
             'relics': ['burning_blood'], 'relic_counters': {}, 'potions': [None, None]}
    env = StandardCombatEnv(catalog([probe]))
    try:
        encounters = list(env.environment_spec()['support']['encounters'])
    finally:
        env.close()

    def validate(document: dict) -> None:
        # Construction checks catalog admission only, on the dense layout, and never resets: an
        # entry that overflows a compact capacity preset first shows at reset, in training or a
        # panel, as a typed Gym Capacity fault (a DriverFault, never a loss). The bounds preset
        # (standard-combat-compact-v1) is sized to the entry bounds; the play preset can overflow,
        # so baseline plans refuse it (scripts/neural/baseline_entry.py).
        try:
            StandardCombatEnv(document).close()
        except _native.RunInputError as error:  # invalid catalog input: the one recognized rejection
            raise CatalogRejected(str(error)) from error

    return validate, encounters


__all__ = ['ASCENSION', 'CATALOG_SCHEMA', 'MAX_CATALOG', 'PROFILE', 'CatalogRejected', 'Validate',
           'bind_gym', 'catalog', 'potion_slots', 'validate_entries']
