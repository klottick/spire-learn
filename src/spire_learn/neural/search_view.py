"""Public columns and heuristic policy for Standard combat search."""

from collections.abc import Mapping
import random


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
