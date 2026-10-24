"""Deterministic soft wall-time budgets."""

import math
from collections.abc import Callable


class Budget:
    """Track a finite wall-time allowance against an injected clock."""

    def __init__(self, max_wall_seconds: float, clock: Callable[[], float]):
        if type(max_wall_seconds) not in (int, float):
            raise TypeError("max_wall_seconds must be a primitive int or float")
        if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive and finite")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._max_wall_seconds = float(max_wall_seconds)
        self._clock = clock
        self._start = self._read_clock()

    def _read_clock(self) -> float:
        value = self._clock()
        if type(value) not in (int, float):
            raise TypeError("clock must return a primitive int or float")
        if not math.isfinite(value):
            raise ValueError("clock must return a finite value")
        return float(value)

    def remaining(self) -> float:
        elapsed = max(0.0, self._read_clock() - self._start)
        return max(0.0, self._max_wall_seconds - elapsed)

    def exhausted(self) -> bool:
        return self.remaining() <= 0.0
