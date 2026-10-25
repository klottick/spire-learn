import unittest
from importlib import import_module

from spire_learn.contracts import Failure
from tests.support import case_execution, case_result


class EvaluationReportTests(unittest.TestCase):
    def _summarize(self, results):
        try:
            module = import_module("spire_learn.evaluation.report")
        except ImportError as error:
            self.fail(f"evaluation summary API is missing: {error}")
        summarize = getattr(module, "summarize", None)
        self.assertTrue(callable(summarize), "evaluation summary API is missing")
        return summarize(results)

    def test_status_and_outcome_counts_reconcile(self):
        results = [
            case_result(execution=case_execution(execution_id="victory")),
            case_result(
                execution=case_execution(execution_id="death"),
                outcome="death",
                reward_sum=-1.0,
            ),
            case_result(
                execution=case_execution(execution_id="truncated"),
                status="env_truncated",
                outcome=None,
                stop_reason="truncated",
            ),
            case_result(
                execution=case_execution(execution_id="budget"),
                status="budget_stopped",
                outcome=None,
                stop_reason="wall_time_budget",
            ),
            case_result(
                execution=case_execution(execution_id="fault"),
                status="simulator_fault",
                outcome=None,
                primary_failure=Failure("simulator_fault", "fixture", "step"),
            ),
            case_result(
                execution=case_execution(execution_id="setup"),
                status="setup_error",
                started=False,
                outcome=None,
                stop_reason="setup_error",
            ),
            case_result(
                execution=case_execution(execution_id="not-started"),
                status="not_started",
                started=False,
                outcome=None,
                stop_reason="wall_time_budget",
            ),
            case_result(
                execution=case_execution(execution_id="overflow"),
                status="contract_error",
                outcome="victory",
                primary_failure=Failure("contract_error", "reward sum overflow", "reward_sum"),
                secondary_errors=(Failure("exec_error", "trace close", "trace"),),
            ),
        ]

        summary = self._summarize(results)

        self.assertEqual(summary["scheduled"], 8)
        self.assertEqual(summary["started"], 6)
        self.assertEqual(summary["not_started"], 2)
        self.assertEqual(summary["scheduled"], summary["started"] + summary["not_started"])
        self.assertEqual(summary["setup_failures"], 1)
        self.assertEqual(
            summary["status_counts"],
            {
                "terminal": 2,
                "env_truncated": 1,
                "budget_stopped": 1,
                "agent_error": 0,
                "simulator_fault": 1,
                "contract_error": 1,
                "exec_error": 0,
            },
        )
        self.assertEqual(summary["outcome_counts"], {"victory": 2, "death": 1})
        self.assertEqual(summary["completed_game_win_rate"], 2 / 3)
        self.assertEqual(summary["terminal_completion_coverage"], 2 / 6)
        self.assertEqual(summary["secondary_failure_count"], 1)

        zero_denominator = self._summarize(
            [
                case_result(
                    execution=case_execution(execution_id="zero"),
                    status="not_started",
                    started=False,
                    outcome=None,
                )
            ]
        )
        self.assertIsNone(zero_denominator["completed_game_win_rate"])
        self.assertIsNone(zero_denominator["terminal_completion_coverage"])

    def test_unknown_and_lifecycle_inconsistent_statuses_are_rejected(self):
        malformed = (
            case_result(
                execution=case_execution(execution_id="unstarted-terminal"),
                status="terminal",
                started=False,
                outcome=None,
            ),
            case_result(
                execution=case_execution(execution_id="started-setup"),
                status="setup_error",
                started=True,
                outcome=None,
            ),
            case_result(
                execution=case_execution(execution_id="unknown"),
                status="mystery",
                started=True,
                outcome=None,
            ),
        )

        for result in malformed:
            with self.subTest(status=result.status, started=result.started):
                with self.assertRaisesRegex(ValueError, "status"):
                    self._summarize([result])

    def test_duplicate_execution_ids_are_rejected(self):
        first = case_result(execution=case_execution(execution_id="duplicate"))
        second = case_result(execution=case_execution(execution_id="duplicate"))

        with self.assertRaisesRegex(ValueError, "duplicate execution_id"):
            self._summarize([first, second])


if __name__ == "__main__":
    unittest.main()
