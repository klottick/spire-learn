"""Tests for dependency-free dual-mode learner records."""

from __future__ import annotations

import sys
import unittest

from spire_learn.neural.contracts import (
    BufferArray,
    CombatFrame,
    ContractError,
    CumulativeCounters,
    Lifecycle,
    ModelConfig,
    NeuralIdentity,
    PolicyChoice,
    Rollout,
    StepRecord,
    TaskSignal,
)
from spire_learn.neural.returns import task_signal


FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"


def frame(value: int = 0) -> CombatFrame:
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


class NeuralContractsTests(unittest.TestCase):
    def test_model_config_enforces_the_ten_million_parameter_ceiling(self):
        self.assertEqual(ModelConfig(16, 64, 10_000_000).max_parameters, 10_000_000)
        with self.assertRaises(ContractError):
            ModelConfig(16, 64, 10_000_001)

    def test_buffer_array_owns_exactly_sized_immutable_native_bytes(self):
        array = BufferArray("int64", (2,), bytes(16))
        self.assertEqual((array.dtype, array.shape, len(array.data)), ("int64", (2,), 16))
        for args in (
            ("float32", (1,), bytes(4)),
            ("int8", (2,), bytes(1)),
            ("int64", (1,), bytearray(8)),
            ("int8", (True,), bytes(1)),
        ):
            with self.subTest(args=args), self.assertRaises(ContractError):
                BufferArray(*args)

    def test_combat_frame_copies_mapping_and_rejects_non_actor_records(self):
        source = {"player": BufferArray("int8", (1,), b"\x01")}
        owned = CombatFrame(source, FINGERPRINT)
        source.clear()
        self.assertIn("player", owned.tensors)
        with self.assertRaises(TypeError):
            owned.tensors["player_known"] = BufferArray("int8", (1,), b"\x00")
        with self.assertRaises(ContractError):
            CombatFrame({"privileged_seed": object()}, FINGERPRINT)

    def test_combat_frame_accepts_only_common_descriptor_actor_keys(self):
        value_groups = {
            "player", "permanent_cards", "combat_cards", "enemies", "powers", "relics",
            "potions", "selection", "selection_counts", "actions", "combat_state", "known_draw",
        }
        occupied_groups = {
            "permanent_cards", "combat_cards", "enemies", "powers", "relics", "potions",
            "selection", "actions", "known_draw",
        }
        descriptor_names = (
            value_groups
            | {f"{name}_known" for name in value_groups}
            | {f"{name}_occupied" for name in occupied_groups}
            | {"action_mask"}
        )
        self.assertEqual(len(descriptor_names), 34)
        complete_name_fixture = CombatFrame(
            {name: BufferArray("int8", (1,), b"\x00") for name in descriptor_names}, FINGERPRINT
        )
        self.assertEqual(set(complete_name_fixture.tensors), descriptor_names)
        partial = CombatFrame(
            {"enemies_occupied": BufferArray("int8", (1,), b"\x01")}, FINGERPRINT
        )
        self.assertEqual(tuple(partial.tensors), ("enemies_occupied",))
        for name, fingerprint in (
            ("seed", FINGERPRINT),
            ("routing", FINGERPRINT),
            ("player_value", FINGERPRINT),
            ("player", "f" * 64),
        ):
            with self.subTest(name=name, fingerprint=fingerprint), self.assertRaises(ContractError):
                CombatFrame({name: BufferArray("int8", (1,), b"\x00")}, fingerprint)

    def test_lifecycle_freezes_last_exit_and_checks_counters(self):
        public_exit = exit_data()
        current = lifecycle(active=False, routing="noncombat", active_visit=None,
                            combats_settled=1, last_exit=public_exit)
        public_exit["outcome"] = "death"
        self.assertEqual(current.last_exit["outcome"], "victory")
        with self.assertRaises(TypeError):
            current.last_exit["outcome"] = "death"
        with self.assertRaises(ContractError):
            lifecycle(combats_entered=0, combats_settled=1)
        with self.assertRaises(ContractError):
            lifecycle(routing="combat_selection", selection_owner="learned")
        selected = lifecycle(routing="combat_selection", selection_owner="card")
        self.assertEqual(selected.selection_owner, "card")

    def test_lifecycle_preserves_public_encounter_text(self):
        public_exit = exit_data()
        public_exit["encounter"] = "Large Slime"
        current = lifecycle(active=False, routing="noncombat", active_visit=None,
                            combats_settled=1, last_exit=public_exit)
        self.assertEqual(current.last_exit["encounter"], "Large Slime")

    def test_lifecycle_uses_native_visit_and_exit_fields(self):
        with self.assertRaises(ContractError):
            lifecycle(
                active=False,
                routing="noncombat",
                active_visit=None,
                combats_settled=1,
                last_exit={
                    "visit_id": "visit-1",
                    "outcome": "victory",
                },
            )
        with self.assertRaises(ContractError):
            lifecycle(active_visit=True)

    def test_fixed_controller_choice_has_no_fabricated_policy_statistics(self):
        choice = PolicyChoice(original_action_index=3)
        record = StepRecord(
            stream_id="stream-1",
            task_id="task-1",
            decision_id="decision-1",
            source=frame(),
            successor=frame(1),
            native_reward=0.0,
            terminated=False,
            truncated=False,
            task_signal=TaskSignal(0.0, False, True, True),
            choice=choice,
            actor_training_mask=False,
            next_value=None,
        )
        self.assertIsNone(record.choice.value)
        self.assertIsNone(record.choice.log_probability)
        self.assertIsNone(record.next_value)
        with self.assertRaises(ContractError):
            StepRecord(
                stream_id="stream-1",
                task_id="task-1",
                decision_id="decision-2",
                source=frame(),
                successor=frame(1),
                native_reward=0.0,
                terminated=False,
                truncated=False,
                task_signal=TaskSignal(0.0, False, True, True),
                choice=choice,
                actor_training_mask=True,
                next_value=0.5,
            )

    def test_neural_import_keeps_optional_runtime_packages_unloaded(self):
        self.assertFalse(
            any(name in sys.modules for name in (
                "numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab",
            ))
        )

    def test_rollout_carries_cumulative_counters_and_compatible_identity(self):
        identity = NeuralIdentity(
            model_id="model-v1",
            features_id="features-v1",
            action_id="actions-v1",
            objective_id="combat-winloss-v1",
            accepted_raw_mode_mappings={"standalone": "view-v1", "run": "view-v1"},
            config_hashes={"model": "a" * 64, "ppo": "b" * 64},
        )
        row = StepRecord(
            stream_id="stream-1",
            task_id="task-1",
            decision_id="decision-1",
            source=frame(),
            successor=frame(1),
            native_reward=0.0,
            terminated=False,
            truncated=False,
            task_signal=TaskSignal(0.0, False, True, True),
            choice=PolicyChoice(0, -0.25, 0.5, 1),
            actor_training_mask=True,
            next_value=0.4,
        )
        rollout = Rollout(
            mode="standalone",
            split="train",
            identity=identity,
            records=(row,),
            counters=CumulativeCounters(5, 4, 1, 1, 1, 0),
        )
        self.assertEqual((rollout.counters.rows, len(rollout.records)), (5, 1))

    def test_task_signal_counts_only_a_new_settled_visit(self):
        previous = lifecycle()
        current = lifecycle(
            active=False,
            routing="noncombat",
            active_visit=None,
            combats_settled=1,
            last_exit=exit_data(),
        )
        signal = task_signal(previous, current, terminated=False, truncated=False)
        self.assertEqual(signal, TaskSignal(1.0, True, False, False))
        repeated = task_signal(current, current, terminated=False, truncated=False)
        self.assertEqual(repeated.reward, 0.0)
        self.assertFalse(repeated.task_terminal)

    def test_task_signal_rejects_unknown_natural_outcome_and_requires_missing_exit(self):
        previous = lifecycle()
        settled = lifecycle(active=False, routing="noncombat", combats_settled=1,
                            active_visit=None, last_exit=exit_data())
        object.__setattr__(settled, "last_exit", None)
        with self.assertRaises(ContractError):
            task_signal(previous, settled, terminated=False, truncated=False)
        with self.assertRaises(ContractError):
            lifecycle(active=False, routing="noncombat", combats_settled=1,
                      active_visit=None, last_exit=exit_data(outcome="aborted"))

    def test_settled_lifecycle_requires_a_public_exit(self):
        with self.assertRaises(ContractError):
            lifecycle(active=False, routing="noncombat", active_visit=None,
                      combats_settled=1, last_exit=None)

    def test_lifecycle_rejects_stale_exit_visit_after_new_settlement(self):
        with self.assertRaises(ContractError):
            lifecycle(active=False, routing="noncombat", active_visit=None,
                      combats_entered=2, combats_settled=2, last_exit=exit_data(visit=0))


if __name__ == "__main__":
    unittest.main()
