"""Public columns and heuristic policy for Standard combat search."""

from collections.abc import Mapping
import random

PLAYOUT_POLICIES = ('uniform-v1', 'no-potions-v2')


class View:
    """Column lookups in the Standard combat view, resolved once from the descriptor."""

    def __init__(self, descriptor: Mapping):
        tensors = {tensor['key']: tensor for tensor in descriptor['tensors']}
        def column(key, field):
            return [c['field'] for c in tensors[key]['columns']].index(field)
        self.player_hp, self.player_max = column('player', 'hp'), column('player', 'max_hp')
        self.enemy_hp, self.enemy_max = column('enemies', 'hp'), column('enemies', 'max_hp')
        self.verb = column('actions', 'verb')
        verbs = descriptor['vocabularies']['verb']
        self.end_turn, self.discard = verbs.index('end_turn'), verbs.index('discard_potion')
        self.use_potion = verbs.index('use_potion')
        # Columns read only by the resource value rule; resolved lazily so descriptors without them still load.
        self._cleanup_hp_column = self._optional_column(tensors, 'player', 'hp_after_cleanup')
        self._potion_kind_column = self._optional_column(tensors, 'potions', 'potion_kind')
        self._potion_vocabulary = tuple(descriptor['vocabularies'].get('potion', ()))

    @staticmethod
    def _optional_column(tensors, key, field):
        if key not in tensors:
            return None
        fields = [c['field'] for c in tensors[key]['columns']]
        return fields.index(field) if field in fields else None

    def _known_player(self, observation, column: int, name: str) -> int:
        if not observation['player_known'][column]:
            raise ValueError(f'player {name} is not known')
        return int(observation['player'][column])

    def player_hp_now(self, observation) -> int:
        """Live player HP; raises when unknown."""
        return self._known_player(observation, self.player_hp, 'hp')

    def player_max_hp(self, observation) -> int:
        """Player max HP; raises when unknown."""
        return self._known_player(observation, self.player_max, 'max_hp')

    def cleanup_hp(self, observation) -> int:
        """Player HP after combat-end hooks; raises when unknown (never falls back to ``hp``)."""
        if self._cleanup_hp_column is None:
            raise ValueError('descriptor has no player hp_after_cleanup column')
        return self._known_player(observation, self._cleanup_hp_column, 'hp_after_cleanup')

    def potion_kinds(self, observation) -> tuple[int, ...]:
        """Sorted potion-kind ids of the occupied potion rows."""
        if self._potion_kind_column is None:
            raise ValueError('descriptor has no potions potion_kind column')
        occupied = observation['potions_occupied']
        slots, column = observation['potions'], self._potion_kind_column
        return tuple(sorted(int(slots[row][column]) for row in range(len(occupied)) if occupied[row]))

    def potion_price_table(self, rule) -> dict[int, float]:
        """HP price per potion-kind id for ``rule``; raises ``ValueError`` on an unknown override id."""
        vocabulary = self._potion_vocabulary
        overrides = dict(rule.potion_overrides)
        unknown = sorted(set(overrides) - set(vocabulary))
        if unknown:
            raise ValueError(f'unknown potion ids in price overrides: {unknown}')
        return {index: float(overrides.get(name, rule.potion_hp))
                for index, name in enumerate(vocabulary) if name != 'padding'}

    def enemy_share(self, observation) -> float:
        """Remaining enemy HP share, the enemy term of ``balance``."""
        enemies = observation['enemies']
        alive = enemies[:, self.enemy_max] > 0
        enemy_hp = float(enemies[alive, self.enemy_hp].clip(min=0).sum())
        return enemy_hp / (float(enemies[alive, self.enemy_max].sum()) or 1.0)

    def playout_choice(self, observation, rng: random.Random, policy: str) -> int:
        """One playout move under ``policy`` (see ``PLAYOUT_POLICIES``)."""
        if policy == 'uniform-v1':
            return self.heuristic_choice(observation, rng)
        if policy != 'no-potions-v2':
            raise ValueError(f'unknown playout policy: {policy!r}')
        legal = self.legal(observation)
        verbs = [observation['actions'][i][self.verb] for i in legal]
        barred = (self.use_potion, self.discard, self.end_turn)
        preferred = [i for i, v in zip(legal, verbs) if v not in barred]
        if preferred:
            return rng.choice(preferred)
        for i, v in zip(legal, verbs):
            if v == self.end_turn:
                return i
        return rng.choice(legal)

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


def summary_balance(result, index: int) -> float:
    """``View.balance`` evaluated on one row of a native playout HP summary.

    Alive enemies are the rows ``enemy_present`` marks; enemy HP is clipped at zero.
    """
    present = result['enemy_present'][index]
    hp, maximum = result['enemy_hp'][index], result['enemy_max_hp'][index]
    rows = [column for column in range(len(present)) if present[column]]
    enemy_hp = float(sum(max(int(hp[column]), 0) for column in rows))
    enemy_max = float(sum(int(maximum[column]) for column in rows)) or 1.0
    mine = float(max(int(result['player_hp'][index]), 0)) / float(max(int(result['player_max_hp'][index]), 1))
    return mine - enemy_hp / enemy_max


def _native_errors():
    """Gym's typed errors: (unsupported boundary, state, fault)."""
    from spire_lab import _native
    return _native.RunUnsupportedBoundaryError, _native.RunStateError, _native.RunFaultError


def refusal_class(error: BaseException) -> str:
    """The class of a root refusal: the Gym's reason text (it names a class of reason, never a value).

    A refusal that is not an ``UnsupportedBoundary`` (a state error) is prefixed ``state: ``.
    """
    unsupported, _, _ = _native_errors()
    text = str(error)[:80]
    return text if isinstance(error, unsupported) else f'state: {text}'
