"""Deterministic soft wall-time budgets."""

import math
from collections.abc import Callable


class Budget:
    """Track a finite wall-time allowance against an injected clock."""

    def __init__(self, max_wall_seconds: float, clock: Callable[[], float]):
        if type(max_wall_seconds) not in (int, float):
            raise TypeError("max_wall_seconds must be a primitive int or float")
        duration = self._finite_float(max_wall_seconds, "max_wall_seconds")
        if duration <= 0:
            raise ValueError("max_wall_seconds must be positive and finite")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._max_wall_seconds = duration
        self._clock = clock
        self._start = self._read_clock()

    def _read_clock(self) -> float:
        value = self._clock()
        if type(value) not in (int, float):
            raise TypeError("clock must return a primitive int or float")
        return self._finite_float(value, "clock")

    @staticmethod
    def _finite_float(value: int | float, name: str) -> float:
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be representable as a finite float") from exc
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        return converted

    def elapsed(self) -> float:
        return max(0.0, self._read_clock() - self._start)

    def remaining(self) -> float:
        return max(0.0, self._max_wall_seconds - self.elapsed())

    def exhausted(self) -> bool:
        return self.remaining() <= 0.0
