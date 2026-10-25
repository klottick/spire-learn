"""Deterministic accounting for scheduled evaluation case results."""

from collections.abc import Sequence

from spire_learn.contracts import CaseResult

_STARTED_STATUSES = (
    "terminal",
    "env_truncated",
    "budget_stopped",
    "agent_error",
    "simulator_fault",
    "contract_error",
    "exec_error",
)
_OUTCOMES = ("victory", "death")
_NOT_STARTED_STATUSES = ("not_started", "setup_error")


def summarize(results: Sequence[CaseResult]) -> dict[str, object]:
    """Summarize one result per scheduled execution without conflating outcome and status."""
    scheduled = len(results)
    execution_ids: set[str] = set()
    status_counts = {status: 0 for status in _STARTED_STATUSES}
    outcome_counts = {outcome: 0 for outcome in _OUTCOMES}
    started = 0
    setup_failures = 0
    secondary_failure_count = 0

    for result in results:
        if not isinstance(result, CaseResult):
            raise TypeError("results must contain CaseResult records")
        execution_id = result.execution.execution_id
        if execution_id in execution_ids:
            raise ValueError(f"duplicate execution_id: {execution_id}")
        execution_ids.add(execution_id)

        if type(result.started) is not bool:
            raise ValueError("CaseResult.started must be a boolean")
        if result.started and result.status not in status_counts:
            raise ValueError(f"unknown started execution status: {result.status}")
        if not result.started and result.status not in _NOT_STARTED_STATUSES:
            raise ValueError(f"unknown not-started execution status: {result.status}")
        if result.status == "setup_error":
            setup_failures += 1
        if result.started:
            started += 1
            status_counts[result.status] += 1
            if (
                result.status != "simulator_fault"
                and (result.primary_failure is None or result.primary_failure.kind != "simulator_fault")
                and result.outcome in outcome_counts
            ):
                outcome_counts[result.outcome] += 1
        secondary_failure_count += len(result.secondary_errors)

    outcome_denominator = outcome_counts["victory"] + outcome_counts["death"]
    return {
        "scheduled": scheduled,
        "started": started,
        "not_started": scheduled - started,
        "setup_failures": setup_failures,
        "status_counts": status_counts,
        "outcome_counts": outcome_counts,
        "completed_game_win_rate": (
            outcome_counts["victory"] / outcome_denominator if outcome_denominator else None
        ),
        "terminal_completion_coverage": status_counts["terminal"] / started if started else None,
        "secondary_failure_count": secondary_failure_count,
    }
