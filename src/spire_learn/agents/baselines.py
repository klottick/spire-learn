"""Simple baselines that consume only the current public decision."""

import random

from ..contracts import Choice, ContractError, PublicDecision
from .base import _legal_indices


class RandomPolicy:
    def __init__(self, seed: int):
        self._random = random.Random(seed)

    def init_state(self) -> None:
        return None

    def act(self, public: PublicDecision, state: object | None) -> Choice:
        legal = _legal_indices(public)
        if not legal:
            raise ContractError("cannot choose from an empty legal set")
        return Choice(self._random.choice(legal))


class PlayFirstPolicy:
    def init_state(self) -> None:
        return None

    def act(self, public: PublicDecision, state: object | None) -> Choice:
        legal = _legal_indices(public)
        if not legal:
            raise ContractError("cannot choose from an empty legal set")
        candidate_by_index = {candidate.index: candidate for candidate in public.candidates}
        for kind in ("play_card", "end_turn"):
            matching = [index for index in legal if candidate_by_index[index].kind == kind]
            if matching:
                return Choice(min(matching))
        raise ContractError("no legal play_card or end_turn candidate")
