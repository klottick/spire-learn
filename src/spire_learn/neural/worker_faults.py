"""Picklable worker-pool faults.

Lives in an importable package module (not an entry script) so that spawn workers, which re-run an
entry script as ``__mp_main__``, still pickle a class the parent can import.
"""

from __future__ import annotations

import traceback


class WorkerFault(RuntimeError):
    """A worker job's exception as plain str/int/None fields (Gym fault classes cannot cross the pool)."""

    def __init__(self, kind, case, error_type, error_message, traceback_text):
        self.kind, self.case = kind, dict(case)
        self.error_type, self.error_message, self.traceback_text = error_type, error_message, traceback_text
        for name, value in self.case.items():
            setattr(self, name, value)
        super().__init__(str(self))

    def __str__(self):
        where = " ".join(f"{name} {value}" for name, value in self.case.items())
        return (f"{self.kind} fault in {where}: {self.error_type}: {self.error_message}\n"
                f"worker traceback:\n{self.traceback_text}")

    def __reduce__(self):
        return (type(self), (self.kind, self.case, self.error_type, self.error_message, self.traceback_text))


def _message(error):
    try:
        return str(error)
    except Exception:  # a broken __str__ must not hide the fault
        return repr(type(error))


def fault_from(error, kind, case):
    """Build the WorkerFault for ``error`` raised while handling ``case`` (call inside the except block)."""
    return WorkerFault(kind, case, f"{type(error).__module__}.{type(error).__qualname__}",
                       _message(error), traceback.format_exc())
