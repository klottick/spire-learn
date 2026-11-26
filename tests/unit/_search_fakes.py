"""Descriptor and observation fakes that follow the Standard descriptor structure."""



class Arr:
    """The few numpy operations ``View`` uses, over plain lists, so unit tests need no numpy."""

    def __init__(self, data):
        self.data = list(data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __gt__(self, other):
        return Arr(x > other for x in self.data)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self.data[key]
        rows, column = key
        if isinstance(rows, slice):
            return Arr(row[column] for row in self.data[rows])
        if isinstance(rows, Arr):
            rows = [i for i, keep in enumerate(rows.data) if keep]
        return Arr(self.data[i][column] for i in rows)

    def nonzero(self):
        return ([i for i, x in enumerate(self.data) if x],)

    def clip(self, min=None):
        return Arr(x if x >= min else min for x in self.data)

    def sum(self):
        return sum(self.data)

VERBS = ['padding', 'play_card', 'end_turn', 'use_potion', 'discard_potion']
POTIONS = ['padding', 'block_potion', 'fairy_potion', 'fire_potion', 'fruit_juice', 'swift_potion', 'smoke_bomb']
PLAYER = ['hp', 'max_hp', 'hp_before_healing', 'hp_after_cleanup']
ENEMY = ['enemy_kind', 'hp', 'max_hp']


def descriptor():
    def tensor(key, fields):
        return {'key': key, 'columns': [{'field': f} for f in fields]}
    return {'tensors': [tensor('player', PLAYER), tensor('enemies', ENEMY),
                        tensor('potions', ['potion_kind']), tensor('actions', ['verb'])],
            'vocabularies': {'verb': VERBS, 'potion': POTIONS}}


def observation(*, potions_unknown=False, hp=50, max_hp=80, cleanup=None, potions=(), enemies=((20, 40),), verbs=(), known=True):
    """``cleanup`` None means unknown; ``potions`` are names; ``verbs`` are the legal action verbs."""
    player = [hp, max_hp, 0, cleanup or 0]
    player_known = [1, 1, 1, 0 if cleanup is None else 1]
    if not known:
        player_known = [0, 0, 0, 0]
    slots = [[0] for _ in range(5)]
    slots_known = [[0 if potions_unknown else 1] for _ in range(5)]
    occupied = [0] * 5
    for row, name in enumerate(potions):
        slots[row][0], occupied[row] = POTIONS.index(name), 1
    foes = [[1, hp_, max_] for hp_, max_ in enemies] or [[0, 0, 0]]
    actions = [[0] for _ in range(16)]
    mask = [0] * 16
    for row, verb in enumerate(verbs):
        actions[row][0], mask[row] = VERBS.index(verb), 1
    return {'player': Arr(player), 'player_known': Arr(player_known), 'potions': Arr(slots), 'potions_known': Arr(slots_known),
            'potions_occupied': Arr(occupied), 'enemies': Arr(foes), 'actions': Arr(actions),
            'action_mask': Arr(mask)}
