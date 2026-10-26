"""Focused return-arithmetic and lifecycle boundary tests."""

from __future__ import annotations

import unittest

from spire_learn.neural.contracts import (
    BufferArray,
    CombatFrame,
    ContractError,
    Lifecycle,
    PolicyChoice,
    StepRecord,
    TaskSignal,
)
from spire_learn.neural.returns import compute_gae, task_signal


FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"


def approx_tuple(actual, expected):
    return len(actual) == len(expected) and all(
        abs(left - right) < 1e-12 for left, right in zip(actual, expected)
    )


def frame(value: int) -> CombatFrame:
    return CombatFrame({"player": BufferArray("int8", (1,), bytes((value,)))}, FINGERPRINT)


def lifecycle(**changes) -> Lifecycle:
    values = dict(
        active=True,
        routing="combat",
        selection_owner=None,
        active_visit=0,
        combats_entered=1,
        combats_settled=0,
        last_exit=None,
    )
    values.update(changes)
    return Lifecycle(**values)


def exit_data(*, visit: int = 0, outcome: str = "victory") -> dict[str, object]:
    return {
        "visit": visit,
        "encounter": "cultist",
        "outcome": outcome,
        "hp_before_healing": 64,
        "hp_after_cleanup": 64,
    }


def record(
    decision_id: str,
    *,
    stream_id: str = "stream-1",
    task_id: str = "task-1",
    source_value: int = 0,
    successor_value: int = 1,
    value: float | None = 0.5,
    next_value: float | None = 0.4,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    task_terminal: bool = False,
    bootstrap: bool = True,
    continuation: bool = True,
    actor_training: bool = True,
) -> StepRecord:
    choice = PolicyChoice(
        original_action_index=0,
        log_probability=-0.25 if actor_training else None,
        value=value if actor_training else None,
        behavior_policy_version=7 if actor_training else None,
    )
    return StepRecord(
        stream_id=stream_id,
        task_id=task_id,
        decision_id=decision_id,
        source=frame(source_value),
        successor=frame(successor_value),
        native_reward=0.0,
        terminated=terminated,
        truncated=truncated,
        task_signal=TaskSignal(reward, task_terminal, bootstrap, continuation),
        choice=choice,
        actor_training_mask=actor_training,
        next_value=next_value if actor_training else None,
    )


class NeuralReturnsTests(unittest.TestCase):
    def test_gae_rejects_same_task_rows_after_natural_or_native_terminal(self):
        terminal_cases = (
            ("natural", False),
            ("native", True),
        )
        for label, terminated in terminal_cases:
            with self.subTest(end=label), self.assertRaises(ContractError):
                compute_gae(
                    (
                        record(
                            f"{label}-end",
                            source_value=0,
                            successor_value=1,
                            task_terminal=True,
                            bootstrap=False,
                            continuation=False,
                            terminated=terminated,
                        ),
                        record(
                            f"{label}-resumed",
                            source_value=1,
                            successor_value=2,
                        ),
                    ),
                    gamma=0.9,
                    lam=0.8,
                )

    def test_step_record_rejects_native_end_signal_disagreement(self):
        with self.assertRaises(ContractError):
            record("bad-terminated", terminated=True)
        with self.assertRaises(ContractError):
            record("bad-truncated", truncated=True)
        explicit_truncation = record(
            "supported-truncation-record",
            truncated=True,
            continuation=False,
        )
        self.assertTrue(explicit_truncation.task_signal.bootstrap)

    def test_gae_does_not_bootstrap_across_native_termination_if_signal_is_corrupted(self):
        ended = record(
            "native-ended",
            terminated=True,
            task_terminal=True,
            bootstrap=False,
            continuation=False,
        )
        object.__setattr__(ended, "task_signal", TaskSignal(0.0, False, True, True))
        batch = compute_gae((ended,), gamma=0.9, lam=0.8)
        self.assertEqual(batch.targets, (0.0,))

    def test_gae_hand_example(self):
        records = (
            record("d1", source_value=0, successor_value=1, value=0.5, next_value=0.4),
            record(
                "d2",
                source_value=1,
                successor_value=2,
                value=0.4,
                next_value=9.0,
                reward=1.0,
                task_terminal=True,
                bootstrap=False,
                continuation=False,
            ),
        )
        batch = compute_gae(records, gamma=0.9, lam=0.8)
        self.assertTrue(approx_tuple(batch.advantages, (0.292, 0.6)))
        self.assertTrue(approx_tuple(batch.targets, (0.792, 1.0)))
        self.assertEqual(batch.train_mask, (True, True))

    def test_cut_uses_old_successor(self):
        batch = compute_gae(
            (record("cut", source_value=0, successor_value=1, value=0.5, next_value=0.4),),
            gamma=0.9,
            lam=0.8,
        )
        self.assertLess(abs(batch.targets[0] - 0.36), 1e-12)
        self.assertNotAlmostEqual(batch.targets[0], 0.9 * 9.0)

    def test_interleaved_streams_do_not_mix(self):
        rows = (
            record("a1", stream_id="a", task_id="a-task", source_value=0, successor_value=1,
                   value=0.5, next_value=0.4),
            record("b1", stream_id="b", task_id="b-task", source_value=2, successor_value=3,
                   value=0.1, next_value=0.0, reward=0.0, task_terminal=True,
                   bootstrap=False, continuation=False),
            record("a2", stream_id="a", task_id="a-task", source_value=1, successor_value=4,
                   value=0.4, next_value=8.0, reward=1.0, task_terminal=True,
                   bootstrap=False, continuation=False),
        )
        batch = compute_gae(rows, gamma=0.9, lam=0.8)
        self.assertTrue(approx_tuple(batch.advantages, (0.292, -0.1, 0.6)))
        self.assertEqual(batch.decision_ids, ("a1", "b1", "a2"))

    def test_run_task_end_is_not_run_end(self):
        previous = lifecycle()
        current = lifecycle(
            active=False,
            routing="noncombat",
            active_visit=None,
            combats_settled=1,
            last_exit=exit_data(outcome="death"),
        )
        signal = task_signal(previous, current, terminated=False, truncated=False)
        self.assertEqual(signal, TaskSignal(-1.0, True, False, False))
        record_for_task = record(
            "task-end",
            reward=signal.reward,
            task_terminal=signal.task_terminal,
            bootstrap=signal.bootstrap,
            continuation=signal.continuation,
            next_value=9.0,
        )
        batch = compute_gae((record_for_task,), gamma=0.9, lam=0.8)
        self.assertEqual(batch.targets, (-1.0,))

    def test_outside_combat_death_has_no_new_task_loss(self):
        previous = lifecycle(
            active=False,
            routing="noncombat",
            active_visit=None,
            combats_settled=1,
            last_exit=exit_data(),
        )
        current = lifecycle(
            active=False,
            routing="noncombat",
            selection_owner=None,
            active_visit=None,
            combats_settled=1,
            last_exit=exit_data(),
        )
        signal = task_signal(previous, current, terminated=True, truncated=False)
        self.assertEqual(signal, TaskSignal(0.0, True, False, False))

    def test_fixed_noncombat_action_can_enter_and_settle_without_learned_loss(self):
        previous = Lifecycle(
            active=False,
            routing="noncombat",
            selection_owner=None,
            active_visit=None,
            combats_entered=0,
            combats_settled=0,
            last_exit=None,
        )
        current = Lifecycle(
            active=False,
            routing="noncombat",
            selection_owner=None,
            active_visit=None,
            combats_entered=1,
            combats_settled=1,
            last_exit=exit_data(outcome="death"),
        )
        signal = task_signal(previous, current, terminated=False, truncated=False)
        self.assertEqual(signal, TaskSignal(-1.0, True, False, False))
        fixed = record(
            "fixed-startup-settlement",
            actor_training=False,
            reward=signal.reward,
            task_terminal=signal.task_terminal,
            bootstrap=signal.bootstrap,
            continuation=signal.continuation,
        )
        batch = compute_gae((fixed,), gamma=0.9, lam=0.8)
        self.assertEqual(batch.train_mask, (False,))
        self.assertEqual((batch.advantages, batch.targets), ((0.0,), (0.0,)))

    def test_unsupported_truncation_errors_and_supported_branch_uses_old_final(self):
        previous = lifecycle()
        with self.assertRaises(ValueError):
            task_signal(previous, previous, terminated=False, truncated=True)
        signal = task_signal(
            previous,
            previous,
            terminated=False,
            truncated=True,
            supports_truncation=True,
        )
        self.assertEqual(signal, TaskSignal(0.0, False, True, False))
        cut = record(
            "supported-truncation",
            value=0.5,
            next_value=0.4,
            truncated=True,
            task_terminal=signal.task_terminal,
            bootstrap=signal.bootstrap,
            continuation=signal.continuation,
        )
        batch = compute_gae((cut,), gamma=0.9, lam=0.8)
        self.assertLess(abs(batch.targets[0] - 0.36), 1e-12)

    def test_fixed_controller_rows_are_zero_masked(self):
        rows = (
            record("fixed", actor_training=False),
            record("learned", source_value=1, successor_value=2, value=0.25, next_value=0.5),
        )
        batch = compute_gae(rows, gamma=0.9, lam=0.8)
        self.assertEqual(batch.train_mask, (False, True))
        self.assertEqual((batch.advantages[0], batch.targets[0]), (0.0, 0.0))
        self.assertLess(abs(batch.targets[1] - 0.45), 1e-12)


if __name__ == "__main__":
    unittest.main()
