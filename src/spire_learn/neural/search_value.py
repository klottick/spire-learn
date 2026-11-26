"""Search value rules: the flat v1 score and the resource-aware ``resource-v2`` score.

Pure arithmetic over a ``View``'s columns, or over one row of a native playout summary.
``resource-v2`` prices HP, max HP and potions on one HP scale.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math

from .search_view import summary_enemy_share

REFUSED_VALUE = -1.0
VALUE_NAMES = ('flat-v1', 'resource-v2')
HP_CURVES = ('linear', 'sqrt')
DEFAULT_POTION_HP = 12.0
DEFAULT_POTION_OVERRIDES = (('fairy_potion', 30.0), ('fruit_juice', 7.5))
_REASONS = ('victory', 'death', 'escaped')
# Native playout end reasons: a terminal row's reason and its reward; the reasons of a live row.
_SUMMARY_REWARDS = {'victory': 1, 'death': -1, 'escaped': 0}
_LIVE_REASONS = ('step_limit', 'max_steps')
_MAX_HP_WEIGHT = 1.5
_SCALE = 0.4


def _price(value, label) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a number')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be finite and non-negative')
    return value


@dataclass(frozen=True)
class ValueRule:
    name: str
    potion_hp: float = DEFAULT_POTION_HP
    potion_overrides: tuple[tuple[str, float], ...] = DEFAULT_POTION_OVERRIDES
    hp_curve: str = 'linear'

    def __post_init__(self):
        if self.name not in VALUE_NAMES:
            raise ValueError(f'unknown value rule: {self.name!r}')
        if self.hp_curve not in HP_CURVES:
            raise ValueError(f'unknown hp_curve: {self.hp_curve!r}')
        object.__setattr__(self, 'potion_hp', _price(self.potion_hp, 'potion_hp'))
        pairs = []
        for entry in self.potion_overrides:
            kind, price = entry
            if not isinstance(kind, str) or not kind or kind == 'padding':
                raise ValueError(f'invalid potion override id: {kind!r}')
            pairs.append((kind, _price(price, f'potion override {kind}')))
        kinds = [kind for kind, _ in pairs]
        if len(set(kinds)) != len(kinds):
            raise ValueError('duplicate potion override ids')
        object.__setattr__(self, 'potion_overrides', tuple(sorted(pairs)))
        if self.name == 'flat-v1' and (
                self.potion_hp != DEFAULT_POTION_HP
                or self.potion_overrides != DEFAULT_POTION_OVERRIDES
                or self.hp_curve != 'linear'):
            raise ValueError('flat-v1 takes no v2 parameters')

    def as_identity(self) -> dict:
        return {'name': self.name, 'potion_hp': self.potion_hp,
                'potion_overrides': {kind: price for kind, price in self.potion_overrides},
                'hp_curve': self.hp_curve}

    @classmethod
    def from_identity(cls, identity: Mapping) -> 'ValueRule':
        extra = set(identity) - {'name', 'potion_hp', 'potion_overrides', 'hp_curve'}
        if extra:
            raise ValueError(f'unknown value rule fields: {sorted(extra)}')
        fields = {key: identity[key] for key in ('potion_hp', 'hp_curve') if key in identity}
        if 'potion_overrides' in identity:
            overrides = identity['potion_overrides']
            if not isinstance(overrides, Mapping):
                raise ValueError('potion_overrides must be a mapping')
            fields['potion_overrides'] = tuple(overrides.items())
        return cls(identity['name'], **fields)


@dataclass(frozen=True)
class RootContext:
    max_hp: int
    potions: tuple[int, ...]
    prices: Mapping[int, float]

    @classmethod
    def from_observation(cls, view, observation, rule: ValueRule) -> 'RootContext':
        max_hp = view.player_max_hp(observation)
        if max_hp <= 0:
            raise ValueError('root max HP must be positive')
        return cls(max_hp, tuple(sorted(view.potion_kinds(observation))), view.potion_price_table(rule))


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _hp_term(rule: ValueRule, root: RootContext, hp: int) -> float:
    h = _clamp(hp / root.max_hp, 0.0, 1.0)
    return math.sqrt(h) if rule.hp_curve == 'sqrt' else h


def _charge_bonus(rule: ValueRule, root: RootContext, max_end: int, kinds) -> float:
    """Max-HP gain and potion charge, in HP at the 0.4/M scale (outside the HP clip)."""
    now, then = Counter(kinds), Counter(root.potions)
    try:
        charge = (sum(root.prices[kind] * n for kind, n in (then - now).items())
                  - sum(root.prices[kind] * n for kind, n in (now - then).items()))
    except KeyError as error:
        raise ValueError(f'potion kind {error.args[0]} has no price') from None
    return _SCALE * (_MAX_HP_WEIGHT * (max_end - root.max_hp) - charge) / root.max_hp


def _bonus(rule: ValueRule, root: RootContext, view, observation) -> float:
    return _charge_bonus(rule, root, view.player_max_hp(observation), view.potion_kinds(observation))


def _terminal_band(reason: str, h: float, b: float) -> float:
    if reason == 'victory':
        return _clamp(0.55 + _SCALE * h + b, 0.41, 1.0)
    return _clamp(-0.2 + _SCALE * h + b, -0.99, 0.4)


def _cutoff_band(h: float, e: float, b: float) -> float:
    return _clamp(_SCALE * _clamp(h - e, -1.0, 1.0) + b, -0.99, 0.4)


def terminal_value(rule: ValueRule, root: RootContext, view, observation, reward, reason) -> float:
    if rule.name == 'flat-v1':
        return float(reward)
    if reason not in _REASONS:
        raise ValueError(f'terminal reason must be one of {_REASONS}, got {reason!r}')
    if reason == 'death':
        return -1.0
    h = _hp_term(rule, root, view.cleanup_hp(observation))
    b = _bonus(rule, root, view, observation)
    return _terminal_band(reason, h, b)


def cutoff_value(rule: ValueRule, root: RootContext, view, observation) -> float:
    if rule.name == 'flat-v1':
        return 0.5 * view.balance(observation)
    h = _hp_term(rule, root, view.player_hp_now(observation))
    e = view.enemy_share(observation)
    b = _bonus(rule, root, view, observation)
    return _cutoff_band(h, e, b)


def _summary_rule(rule: ValueRule) -> None:
    if rule.name != 'resource-v2':
        raise ValueError(f'summary scoring serves resource-v2, not {rule.name}')


def _summary_bonus(rule: ValueRule, root: RootContext, result, index: int) -> float:
    present, kinds = result['potion_present'][index], result['potion_kind'][index]
    held = tuple(sorted(int(kinds[slot]) for slot in range(len(present)) if present[slot]))
    return _charge_bonus(rule, root, int(result['player_max_hp'][index]), held)


def summary_terminal_value(rule: ValueRule, root: RootContext, result, index: int) -> float:
    """``terminal_value`` of a terminated native playout row; ``player_hp`` there is HP after cleanup.

    The reason is the row's ``end_reason``, which must agree with its reward; anything else raises.
    """
    _summary_rule(rule)
    reason = result['end_reason'][index]
    if not result['terminated'][index]:
        raise ValueError(f'playout row {index} is not terminal (end_reason {reason!r})')
    if reason not in _SUMMARY_REWARDS:
        raise ValueError(f'a terminal playout row must end in one of {_REASONS}, got {reason!r}')
    reward = int(result['rewards'][index])
    if reward != _SUMMARY_REWARDS[reason]:
        raise ValueError(f'a {reason} playout row has reward {reward}')
    if reason == 'death':
        return -1.0
    h = _hp_term(rule, root, int(result['player_hp'][index]))
    b = _summary_bonus(rule, root, result, index)
    return _terminal_band(reason, h, b)


def summary_cutoff_value(rule: ValueRule, root: RootContext, result, index: int) -> float:
    """``cutoff_value`` of a live native playout row, which ended at ``step_limit`` or ``max_steps``."""
    _summary_rule(rule)
    reason = result['end_reason'][index]
    if result['terminated'][index] or reason not in _LIVE_REASONS:
        raise ValueError(f'a live playout row must end in one of {_LIVE_REASONS}, got {reason!r}')
    h = _hp_term(rule, root, int(result['player_hp'][index]))
    e = summary_enemy_share(result, index)
    b = _summary_bonus(rule, root, result, index)
    return _cutoff_band(h, e, b)
