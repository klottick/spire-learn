"""Pure, deterministic collector tests with scripted public driver boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

from spire_learn.experiments.budget import Budget
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.contracts import BufferArray, CombatFrame, Lifecycle, PolicyChoice
from spire_learn.neural.envs import DriverFault, DriverFaultEvidence, DriverRow, DriverTransition
from spire_learn.neural.schema import CombatCandidate


ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"


def frame(value: int) -> CombatFrame:
    return CombatFrame({"player": BufferArray("int8", (1,), bytes((value,)))}, FINGERPRINT)


def lifecycle(
    *,
    active: bool,
    entered: int = 0,
    settled: int = 0,
    visit: int | None = None,
    exit_data: dict[str, object] | None = None,
):
    return Lifecycle(
        active=active,
        routing="combat" if active else "noncombat",
        selection_owner=None,
        active_visit=visit if active else None,
        combats_entered=entered,
        combats_settled=settled,
        last_exit=exit_data,
    )


def exit_data(*, visit: int = 0, outcome: str = "victory") -> dict[str, object]:
    return {
        "visit": visit,
        "encounter": "cultist",
        "outcome": outcome,
        "hp_before_healing": 51,
        "hp_after_cleanup": 57,
    }


def row(
    value: int,
    *,
    active: bool,
    entered: int = 0,
    settled: int = 0,
    visit: int | None = None,
    exit_value: dict[str, object] | None = None,
    noncombat_indices: tuple[int, ...] = (5, 7),
) -> DriverRow:
    candidates = (
        CombatCandidate(0, "end_turn", None, None, None, False, False,
                        None, False, False, None, False, False),
    ) if active else ()
    return DriverRow(
        frame(value),
        lifecycle(active=active, entered=entered, settled=settled,
                  visit=visit, exit_data=exit_value),
        candidates,
        () if active else noncombat_indices,
    )


def transition(
    source: DriverRow,
    successor: DriverRow,
    *,
    terminated: bool = False,
    truncated: bool = False,
    reward: float = 0.0,
) -> DriverTransition:
    return DriverTransition(source, successor, reward, terminated, truncated)


class ScriptedDriver:
    """A full-width pure fake that records reset/step order and never retries."""

    def __init__(self, mode, batch_size, *, initial_rows=None, reset_rows=None, steps=()):
        self.mode = mode
        self.batch_size = batch_size
        self._rows = tuple(initial_rows) if initial_rows is not None else (None,) * batch_size
        self._reset_rows = reset_rows
        self._steps = list(steps)
        self.reset_calls = []
        self.step_calls = []
        self.events = []
        self.accepted_steps = 0
        self.close_calls = 0

    @property
    def rows(self):
        return self._rows

    def reset(self, *, seeds, entry_ids, reset_mask=None):
        mask = tuple(reset_mask) if reset_mask is not None else (True,) * self.batch_size
        before = self._rows
        call_number = len(self.reset_calls)
        self.reset_calls.append((tuple(seeds), tuple(entry_ids), mask, before))
        self.events.append(("reset", call_number, mask, before))
        updated = list(before)
        for slot, selected in enumerate(mask):
            if selected:
                if self._reset_rows is None:
                    updated[slot] = row(10 + call_number + slot, active=True,
                                        entered=1, visit=0)
                else:
                    updated[slot] = self._reset_rows(slot, call_number, seeds[slot], entry_ids[slot])
        self._rows = tuple(updated)
        return self._rows

    def step(self, actions):
        actions = tuple(actions)
        before = self._rows
        self.step_calls.append((actions, before))
        self.events.append(("step", len(self.step_calls) - 1, before))
        if not self._steps:
            raise AssertionError("unexpected extra step")
        scripted = self._steps.pop(0)
        result = scripted(before, actions) if callable(scripted) else scripted
        if len(result) != self.batch_size:
            raise AssertionError("script must return a complete native batch")
        self._rows = tuple(item.successor for item in result)
        self.accepted_steps += 1
        return tuple(result)

    def close(self):
        self.close_calls += 1


class FakePolicy:
    def __init__(self, config, *, value_scale=0.1):
        identity = config.accepted_identities[0]
        self.model_id = identity.model_id
        self.features_id = identity.features_id
        self.action_id = identity.action_id
        self.objective_id = identity.objective_id
        self.behavior_version = 0
        self.value_scale = value_scale
        self.generator_args = []
        self.act_inputs = []
        self.value_inputs = []

    def act(self, frames, *, generator, deterministic=False):
        self.generator_args.append(generator)
        self.act_inputs.append(tuple(frames))
        return tuple(PolicyChoice(0, -0.25, 0.5, self.behavior_version) for _ in frames)

    def values(self, frames):
        frozen = tuple(frames)
        self.value_inputs.append(frozen)
        return tuple(frame_value.tensors["player"].data[0] * self.value_scale
                     for frame_value in frozen)


class NeuralCollectorTests(unittest.TestCase):
    def setUp(self):
        self.standalone_config = replace(
            load_neural_config(ROOT / "configs/neural/standalone.toml"),
            rollout_rows=2,
            rollout_steps=1,
        )
        self.run_config = replace(
            load_neural_config(ROOT / "configs/neural/run.toml"),
            rollout_steps=1,
        )
        self.generator = object()

    def collector(self, driver, config, policy=None):
        from spire_learn.neural.collector import Collector

        selected_policy = policy or FakePolicy(config)
        return Collector(
            driver,
            selected_policy,
            config,
            controller_rng=random.Random(901),
            curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: 0.0),
            policy_generator=self.generator,
        )

    def test_mixed_row_finals_reset_only_finished_rows_and_preserve_successors(self):
        initial_a = row(1, active=True, entered=1, visit=0)
        initial_b = row(2, active=True, entered=1, visit=0)
        after_a = row(3, active=True, entered=1, visit=0)
        after_b = row(4, active=False, entered=1, settled=1,
                      exit_value=exit_data(outcome="death"))
        reset_b = row(20, active=True, entered=1, visit=0)
        after_a2 = row(5, active=True, entered=1, visit=0)
        after_b2 = row(21, active=True, entered=1, visit=0)

        def step_one(rows, actions):
            self.assertEqual(rows, (initial_a, initial_b))
            return (
                transition(rows[0], after_a),
                transition(rows[1], after_b, terminated=True),
            )

        def step_two(rows, actions):
            self.assertEqual(rows, (after_a, reset_b))
            return (transition(rows[0], after_a2), transition(rows[1], after_b2))

        driver = ScriptedDriver(
            "standalone", 2,
            reset_rows=lambda slot, call, seed, entry: (initial_a, initial_b)[slot]
            if call == 0 else reset_b,
            steps=(step_one, step_two),
        )
        policy = FakePolicy(self.standalone_config)
        collector = self.collector(driver, self.standalone_config, policy)

        first = collector.collect(
            max_learned_decisions=8, max_env_actions=4, policy_version=0
        )
        self.assertEqual(tuple(item.source for item in first.records),
                         (initial_a.frame, initial_b.frame))
        self.assertEqual(tuple(item.successor for item in first.records),
                         (after_a.frame, after_b.frame))
        self.assertAlmostEqual(first.records[0].next_value, 0.3)
        self.assertEqual(first.records[1].next_value, 0.0)
        self.assertEqual(first.counters.task_terminals, 1)
        self.assertEqual(first.counters.environment_terminations, 1)
        self.assertEqual(len(driver.reset_calls), 1)

        second = collector.collect(
            max_learned_decisions=8, max_env_actions=4, policy_version=0
        )
        self.assertEqual(driver.reset_calls[1][2], (False, True))
        self.assertEqual(driver.reset_calls[1][3][0], after_a)
        self.assertEqual(driver.step_calls[1][1], (after_a, reset_b))
        self.assertEqual(tuple(item.source for item in second.records),
                         (after_a.frame, reset_b.frame))
        self.assertEqual(second.counters.rows, 4)
        self.assertEqual(second.counters.actor_rows, 4)
        self.assertEqual(second.counters.fixed_rows, 0)
        self.assertEqual(second.collection.reset_operations, 2)
        self.assertEqual(second.collection.reset_slots, 3)
        self.assertEqual(second.collection.known_accepted_environment_slots, 4)
        self.assertEqual(second.collection.delivered_environment_slots, 4)
        self.assertEqual(second.collection.seed_allocations[0].authored_reference_seed,
                         next(case.environment_seed for case in self.standalone_config.cases.cases
                              if case.case_id == second.collection.seed_allocations[0].case_id))
        from spire_learn.neural.cases import environment_seed_for
        self.assertEqual(second.collection.seed_allocations[0].ordinal, 0)
        self.assertEqual(second.collection.seed_allocations[0].actual_seed,
                         environment_seed_for("train", replica=0, ordinal=0))
        self.assertEqual(driver.events.index(("reset", 1, (False, True), (after_a, after_b))),
                         2)
        self.assertEqual(driver.close_calls, 0)
        collector.close()
        collector.close()
        self.assertEqual(driver.close_calls, 1)

    def test_run_victory_closes_task_once_and_keeps_run(self):
        at_combat = row(30, active=True, entered=1, visit=0)
        settlement_exit = exit_data(outcome="victory")
        settled_inventory = row(31, active=False, entered=1, settled=1,
                                exit_value=settlement_exit)
        settlement_exit.update({
            "encounter": "mutated-after-boundary",
            "hp_before_healing": 0,
            "hp_after_cleanup": 0,
        })
        after_reward_choice = row(32, active=False, entered=1, settled=1,
                                  exit_value=exit_data(outcome="victory"))

        def settle(rows, actions):
            return (transition(rows[0], settled_inventory),)

        def choose_reward(rows, actions):
            self.assertEqual(rows, (settled_inventory,))
            return (transition(rows[0], after_reward_choice),)

        driver = ScriptedDriver(
            "run", 1, initial_rows=(at_combat,), steps=(settle, choose_reward)
        )
        policy = FakePolicy(self.run_config)
        collector = self.collector(driver, self.run_config, policy)
        first = collector.collect(
            max_learned_decisions=8, max_env_actions=2, policy_version=0
        )
        second = collector.collect(
            max_learned_decisions=8, max_env_actions=2, policy_version=0
        )

        task_reward_sum = sum(record.task_signal.reward
                              for record in first.records + second.records)
        task_end_count = sum(record.task_signal.task_terminal
                             for record in first.records + second.records)
        run_reset_count = len(driver.reset_calls)
        checkpoint = second.collection.resource_checkpoints[0]
        self.assertEqual((task_reward_sum, task_end_count, run_reset_count), (1.0, 1, 0))
        self.assertEqual(checkpoint.frame, settled_inventory.frame)
        self.assertNotEqual(checkpoint.frame, after_reward_choice.frame)
        self.assertEqual(checkpoint.encounter, "cultist")
        self.assertEqual(checkpoint.hp_before_healing, 51)
        self.assertEqual(checkpoint.hp_after_cleanup, 57)
        self.assertEqual(first.collection.resource_checkpoints, (checkpoint,))
        self.assertEqual(second.collection.resource_checkpoints, (checkpoint,))
        self.assertEqual(first.records[0].task_signal.reward, 1.0)
        self.assertEqual(second.records[0].task_signal.reward, 0.0)
        self.assertTrue(first.records[0].actor_training_mask)
        self.assertFalse(second.records[0].actor_training_mask)
        self.assertIsNone(second.records[0].choice.value)
        self.assertEqual(second.collection.tasks_completed, 1)
        self.assertEqual(second.collection.victories, 1)
        self.assertIs(collector.policy_generator, self.generator)
        self.assertEqual(policy.generator_args, [self.generator])
        collector.close()

    def test_run_noncombat_death_is_terminal_fixed_work(self):
        before = row(40, active=False)
        after = row(41, active=False)
        driver = ScriptedDriver(
            "run", 1, initial_rows=(before,),
            steps=((transition(before, after, terminated=True),),),
        )
        policy = FakePolicy(self.run_config)
        collector = self.collector(driver, self.run_config, policy)
        rollout = collector.collect(
            max_learned_decisions=0, max_env_actions=1, policy_version=0
        )
        self.assertEqual(len(rollout.records), 1)
        record = rollout.records[0]
        self.assertFalse(record.actor_training_mask)
        self.assertTrue(record.terminated)
        self.assertTrue(record.task_signal.task_terminal)
        self.assertEqual(record.task_signal.reward, 0.0)
        self.assertEqual(rollout.counters.environment_terminations, 1)
        self.assertEqual(policy.act_inputs, [])
        self.assertEqual(rollout.collection.deaths, 0)
        collector.close()

    def test_imports_stay_optional_and_fixed_run_action_leaves_torch_lazy(self):
        from spire_learn.neural.collector import Collector

        optional_prefixes = ("numpy", "gymnasium", "torch", "torchrl", "tensordict", "spire_lab")
        self.assertFalse(any(
            module == prefix or module.startswith(prefix + ".")
            for module in sys.modules for prefix in optional_prefixes
        ))
        before = row(45, active=False)
        after = row(46, active=False)
        driver = ScriptedDriver(
            "run", 1, initial_rows=(before,),
            steps=((transition(before, after, terminated=True),),),
        )
        policy = FakePolicy(self.run_config)
        collector = Collector(
            driver,
            policy,
            self.run_config,
            controller_rng=random.Random(77),
            curriculum_rng=random.Random(88),
            budget=Budget(60.0, lambda: 0.0),
        )
        self.assertIsNone(collector.policy_generator)
        collector.collect(
            max_learned_decisions=0, max_env_actions=1, policy_version=0
        )
        self.assertIsNone(collector.policy_generator)
        self.assertFalse(any(
            module == prefix or module.startswith(prefix + ".")
            for module in sys.modules for prefix in optional_prefixes
        ))
        collector.close()

    def test_public_diagnostic_policy_has_truthful_point_mass_and_distinct_identity(self):
        from tests.integration.test_neural_collection import (
            PUBLIC_DIAGNOSTIC_KIND,
            PUBLIC_DIAGNOSTIC_VERSION,
            PublicLowestLegalDiagnosticPolicy,
            make_public_diagnostic_config,
        )

        config, identity = make_public_diagnostic_config(self.run_config)
        policy = PublicLowestLegalDiagnosticPolicy(identity)
        public_frame = CombatFrame(
            {
                "action_mask": BufferArray("int8", (5,), bytes((0, 1, 0, 1, 0))),
                "player": BufferArray("int8", (1,), bytes((101,))),
            },
            FINGERPRINT,
        )
        choices = policy.act((public_frame,), generator=object(), deterministic=False)
        self.assertEqual(PUBLIC_DIAGNOSTIC_KIND, "public-diagnostic-lowest-legal-v1")
        self.assertEqual(PUBLIC_DIAGNOSTIC_VERSION, "public-diagnostic-lowest-legal-v1")
        self.assertEqual(choices, (PolicyChoice(1, 0.0, 0.0, PUBLIC_DIAGNOSTIC_VERSION),))
        self.assertEqual(policy.values((public_frame,)), (0.0,))
        self.assertNotEqual(identity.model_id, self.run_config.accepted_identities[0].model_id)
        self.assertEqual(config.accepted_identities, (identity,))
        self.assertIn("policy_spec", identity.config_hashes)

    def test_horizon_bootstraps_actual_successor_without_an_extra_action_draw(self):
        before = row(50, active=True, entered=1, visit=0)
        after = row(60, active=True, entered=1, visit=0)
        driver = ScriptedDriver(
            "standalone", 2,
            initial_rows=(before, before),
            steps=((transition(before, after), transition(before, after)),),
        )
        config = self.standalone_config
        policy = FakePolicy(config)
        collector = self.collector(driver, config, policy)
        rollout = collector.collect(
            max_learned_decisions=2, max_env_actions=2, policy_version=0
        )
        self.assertEqual(tuple(record.next_value for record in rollout.records), (6.0, 6.0))
        self.assertEqual(len(policy.act_inputs), 1)
        self.assertEqual(len(policy.value_inputs), 1)
        self.assertEqual(policy.value_inputs[0], (after.frame, after.frame))
        self.assertEqual(driver.accepted_steps, 1)
        collector.close()

    def test_wall_budget_stops_before_next_batch_and_reports_overshoot(self):
        before = row(65, active=True, entered=1, visit=0)
        after = row(66, active=True, entered=1, visit=0)
        clock = [0.0]

        def step_then_overshoot(rows, actions):
            clock[0] = 65.0
            return transition(rows[0], after), transition(rows[1], after)

        driver = ScriptedDriver(
            "standalone", 2,
            initial_rows=(before, before),
            steps=(step_then_overshoot,),
        )
        config = replace(self.standalone_config, rollout_steps=2)
        policy = FakePolicy(config)
        from spire_learn.neural.collector import Collector

        collector = Collector(
            driver,
            policy,
            config,
            controller_rng=random.Random(901),
            curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: clock[0]),
            policy_generator=self.generator,
        )
        rollout = collector.collect(
            max_learned_decisions=8, max_env_actions=8, policy_version=0
        )
        self.assertEqual(len(rollout.records), 2)
        self.assertEqual(driver.accepted_steps, 1)
        self.assertEqual(rollout.collection.stop_reason, "wall_budget")
        self.assertEqual(rollout.collection.attempted_environment_slots, 2)
        self.assertEqual(rollout.collection.elapsed_seconds, 65.0)
        self.assertEqual(rollout.collection.wall_overshoot_seconds, 5.0)
        collector.close()

    def test_wall_expiry_during_reset_prevents_actor_sampling_and_dispatch(self):
        clock = [0.0]
        after_reset = row(67, active=True, entered=1, visit=0)
        driver = ScriptedDriver(
            "standalone", 2,
            reset_rows=lambda slot, call, seed, entry: (
                clock.__setitem__(0, 61.0) or after_reset
            ),
            steps=(),
        )
        policy = FakePolicy(self.standalone_config)
        from spire_learn.neural.collector import Collector

        collector = Collector(
            driver,
            policy,
            self.standalone_config,
            controller_rng=random.Random(901),
            curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: clock[0]),
            policy_generator=self.generator,
        )
        rollout = collector.collect(
            max_learned_decisions=8, max_env_actions=4, policy_version=0
        )
        self.assertEqual(rollout.records, ())
        self.assertEqual(rollout.collection.stop_reason, "wall_budget")
        self.assertEqual(rollout.collection.reset_operations, 1)
        self.assertEqual(rollout.collection.delivered_reset_slots, 2)
        self.assertEqual(rollout.collection.attempted_environment_slots, 0)
        self.assertEqual(policy.act_inputs, [])
        self.assertEqual(driver.step_calls, [])
        collector.close()

    def test_wall_expiry_during_policy_inference_counts_undispatched_choices(self):
        clock = [0.0]
        before = row(68, active=True, entered=1, visit=0)
        driver = ScriptedDriver("standalone", 2, initial_rows=(before, before), steps=())

        class ExpiringPolicy(FakePolicy):
            def act(self, frames, *, generator, deterministic=False):
                choices = super().act(frames, generator=generator, deterministic=deterministic)
                clock[0] = 61.0
                return choices

        policy = ExpiringPolicy(self.standalone_config)
        from spire_learn.neural.collector import Collector

        collector = Collector(
            driver,
            policy,
            self.standalone_config,
            controller_rng=random.Random(901),
            curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: clock[0]),
            policy_generator=self.generator,
        )
        rollout = collector.collect(
            max_learned_decisions=8, max_env_actions=4, policy_version=0
        )
        self.assertEqual(rollout.records, ())
        self.assertEqual(rollout.collection.stop_reason, "wall_budget")
        self.assertEqual(rollout.collection.undispatched_actor_decisions, 2)
        self.assertEqual(rollout.collection.undispatched_fixed_decisions, 0)
        self.assertEqual(rollout.collection.attempted_environment_slots, 0)
        self.assertEqual(len(policy.act_inputs), 1)
        self.assertEqual(driver.step_calls, [])
        collector.close()

    def test_run_terminal_restarts_only_with_capacity_and_keeps_seed_and_task_identity(self):
        first_run = row(90, active=True, entered=1, visit=0)
        first_death = row(91, active=False, entered=1, settled=1,
                          exit_value=exit_data(outcome="death"))
        second_run = row(92, active=True, entered=1, visit=0)
        second_successor = row(93, active=True, entered=1, visit=0)
        driver = ScriptedDriver(
            "run", 1,
            reset_rows=lambda slot, call, seed, entry: first_run if call == 0 else second_run,
            steps=(
                lambda rows, actions: (transition(rows[0], first_death, terminated=True),),
                lambda rows, actions: (transition(rows[0], second_successor),),
            ),
        )
        config = replace(self.run_config, rollout_steps=1)
        policy = FakePolicy(config)
        collector = self.collector(driver, config, policy)

        first = collector.collect(
            max_learned_decisions=4, max_env_actions=2, policy_version=0
        )
        self.assertTrue(first.records[0].terminated)
        self.assertEqual(first.collection.reset_operations, 1)
        self.assertEqual(first.records[0].task_id, "task-0-run-0-visit-0")

        too_small = collector.collect(
            max_learned_decisions=1, max_env_actions=1, policy_version=0
        )
        self.assertEqual(too_small.collection.stop_reason, "environment_action_budget")
        self.assertEqual(too_small.records, ())
        self.assertEqual(len(driver.reset_calls), 1)
        self.assertEqual(len(driver.step_calls), 1)

        second = collector.collect(
            max_learned_decisions=4, max_env_actions=2, policy_version=0
        )
        self.assertEqual(len(driver.reset_calls), 2)
        self.assertEqual(len(driver.step_calls), 2)
        self.assertEqual(second.collection.reset_operations, 2)
        self.assertEqual(second.collection.attempted_environment_slots, 2)
        self.assertEqual(second.collection.next_seed_ordinal, 2)
        self.assertEqual(second.collection.tasks_started, 2)
        self.assertEqual(second.records[0].task_id, "task-0-run-1-visit-0")
        self.assertNotEqual(first.records[0].decision_id, second.records[0].decision_id)
        allocations = second.collection.seed_allocations
        self.assertEqual(tuple(item.ordinal for item in allocations), (0, 1))
        self.assertNotEqual(allocations[0].actual_seed, allocations[1].actual_seed)
        self.assertEqual(second.records[0].source, second_run.frame)
        collector.close()

    def test_clock_failure_retains_delivered_prefix_and_preserves_control_exception(self):
        class ClockFailureBudget:
            def __init__(self):
                self.failed = False

            def elapsed(self):
                if self.failed:
                    raise KeyboardInterrupt("primary clock/control failure")
                return 0.0

            def remaining(self):
                if self.failed:
                    raise KeyboardInterrupt("primary clock/control failure")
                return 60.0

        class CloseFailureDriver(ScriptedDriver):
            def close(self):
                self.close_calls += 1
                raise RuntimeError("secondary close failure")

        before_a = row(94, active=True, entered=1, visit=0)
        before_b = row(95, active=True, entered=1, visit=0)
        after_a = row(96, active=True, entered=1, visit=0)
        after_b = row(97, active=True, entered=1, visit=0)
        budget = ClockFailureBudget()

        def delivered_then_clock_fails(rows, actions):
            budget.failed = True
            return transition(rows[0], after_a), transition(rows[1], after_b)

        driver = CloseFailureDriver(
            "standalone", 2, initial_rows=(before_a, before_b),
            steps=(delivered_then_clock_fails,),
        )
        config = replace(self.standalone_config, rollout_steps=1)
        collector = self.collector(driver, config)
        collector.budget = budget
        with self.assertRaisesRegex(KeyboardInterrupt, "primary clock/control failure") as raised:
            collector.collect(
                max_learned_decisions=8, max_env_actions=4, policy_version=0
            )
        partial = collector.partial_rollout
        self.assertIsNotNone(partial)
        self.assertEqual(len(partial.records), 2)
        self.assertEqual(partial.collection.delivered_environment_slots, 2)
        self.assertEqual(partial.collection.faults, 1)
        self.assertTrue(partial.collection.elapsed_seconds_is_fallback)
        self.assertEqual(partial.collection.elapsed_seconds, 0.0)
        self.assertEqual(driver.close_calls, 1)
        notes = getattr(raised.exception, "__notes__", ())
        self.assertTrue(any("secondary close failure" in note for note in notes))
        with self.assertRaisesRegex(RuntimeError, "cannot be retried"):
            collector.collect(
                max_learned_decisions=8, max_env_actions=4, policy_version=0
            )
        self.assertEqual(len(driver.step_calls), 1)
        collector.close()
        self.assertEqual(driver.close_calls, 1)

    def test_insufficient_full_batch_capacity_stops_before_reset_or_policy_draw(self):
        driver = ScriptedDriver("standalone", 2, steps=())
        policy = FakePolicy(self.standalone_config)
        collector = self.collector(driver, self.standalone_config, policy)
        rollout = collector.collect(
            max_learned_decisions=8, max_env_actions=1, policy_version=0
        )
        self.assertEqual(rollout.records, ())
        self.assertEqual(driver.reset_calls, [])
        self.assertEqual(driver.step_calls, [])
        self.assertEqual(policy.act_inputs, [])
        self.assertEqual(rollout.collection.stop_reason, "environment_action_budget")
        collector.close()

    def test_inactive_reset_boundary_is_preserved_without_an_action_or_retry(self):
        settled = row(70, active=False, entered=1, settled=1,
                      exit_value=exit_data(outcome="victory"))
        driver = ScriptedDriver(
            "standalone", 2,
            reset_rows=lambda slot, call, seed, entry: settled,
            steps=(),
        )
        policy = FakePolicy(self.standalone_config)
        collector = self.collector(driver, self.standalone_config, policy)

        first = collector.collect(
            max_learned_decisions=8, max_env_actions=4, policy_version=0
        )
        second = collector.collect(
            max_learned_decisions=8, max_env_actions=4, policy_version=0
        )
        self.assertEqual(first.records, ())
        self.assertEqual(first.collection.stop_reason, "inactive_reset_boundary")
        self.assertEqual(first.collection.resource_checkpoints[0].frame, settled.frame)
        self.assertEqual(first.collection.resource_checkpoints[0].outcome, "victory")
        self.assertEqual(driver.step_calls, [])
        self.assertEqual(len(driver.reset_calls), 1)
        self.assertEqual(policy.act_inputs, [])
        self.assertEqual(second.collection.stop_reason, "inactive_reset_boundary")
        self.assertEqual(len(driver.reset_calls), 1)
        collector.close()

    def test_fault_retains_prefix_diagnostics_closes_once_and_rejects_retry(self):
        before_a = row(80, active=True, entered=1, visit=0)
        before_b = row(81, active=True, entered=1, visit=0)
        after_a = row(82, active=True, entered=1, visit=0)
        after_b = row(83, active=True, entered=1, visit=0)

        def known_prefix(rows, actions):
            return transition(rows[0], after_a), transition(rows[1], after_b)

        def uncertain_second_call(rows, actions):
            evidence = DriverFaultEvidence(
                phase="step",
                accepted_prefix=1,
                failed_operation=2,
                delivery="unknown",
                exception_type="RuntimeError",
                message="scripted uncertain delivery",
                accepted_prefix_operations=1,
                accepted_prefix_slot_actions=2,
                operation_slots=2,
                operation_slot_indices=(0, 1),
                uncertain_current_slots=2,
            )
            raise DriverFault(evidence)

        driver = ScriptedDriver(
            "standalone", 2,
            initial_rows=(before_a, before_b),
            steps=(known_prefix, uncertain_second_call),
        )
        collector = self.collector(driver, self.standalone_config)
        config = replace(self.standalone_config, rollout_steps=2)
        collector.config = config
        with self.assertRaises(DriverFault):
            collector.collect(
                max_learned_decisions=8, max_env_actions=8, policy_version=0
            )
        partial = collector.partial_rollout
        self.assertIsNotNone(partial)
        self.assertEqual(len(partial.records), 2)
        self.assertEqual(partial.collection.faults, 1)
        self.assertEqual(partial.collection.uncertain_environment_slots, 2)
        self.assertEqual(partial.collection.incomplete_environment_slots, 2)
        self.assertEqual(partial.collection.incomplete_decisions, 4)
        self.assertEqual(driver.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "cannot be retried"):
            collector.collect(
                max_learned_decisions=8, max_env_actions=8, policy_version=0
            )
        self.assertEqual(len(driver.step_calls), 2)
        collector.close()
        self.assertEqual(driver.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
