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


class StatelessReferencePolicy:
    def __init__(self, *, choice=0, include_statistics=False):
        self.choice = choice
        self.include_statistics = include_statistics
        self.generators = []
        self.value_calls = 0

    def act(self, frames, *, generator, deterministic=False):
        self.generators.append(generator)
        if self.include_statistics:
            return tuple(PolicyChoice(self.choice, -0.5, 0.25, 0) for _ in frames)
        return tuple(PolicyChoice(self.choice) for _ in frames)

    def values(self, frames):
        self.value_calls += 1
        raise AssertionError("reference policies must not request value estimates")


class StrictCaseSampler:
    """Only accepts the original catalog and natural outcomes, with injected faults."""
    def __init__(self, cases, *, choose_error=None, observe_error=None):
        self.cases = tuple(cases)
        self.chosen = []
        self.observed = []
        self.rngs = []
        self.choose_error = choose_error
        self.observe_error = observe_error

    def validate_cases(self, cases):
        if tuple(cases) != self.cases:
            raise ValueError("sampler catalog mismatch")

    def choose_case(self, cases, *, rng):
        self.validate_cases(cases)
        if self.choose_error is not None:
            raise self.choose_error
        case = self.cases[len(self.chosen) % len(self.cases)]
        self.chosen.append(case)
        self.rngs.append(rng)
        return case

    def observe(self, case_id, outcome):
        if case_id not in {case.case_id for case in self.cases} or outcome not in {"victory", "death"}:
            raise AssertionError("incorrect natural-settlement callback")
        if self.observe_error is not None:
            raise self.observe_error
        self.observed.append((case_id, outcome))

    def snapshot(self):
        return {}


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
        self.selection_config = replace(
            load_neural_config(ROOT / "configs/neural/standalone-selection.toml"),
            rollout_rows=2,
            rollout_steps=1,
        )

    def collector(self, driver, config, policy=None, **kwargs):
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
            **kwargs,
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

    def test_recovery_uses_next_seed_decision_and_episode_without_off_by_one(self):
        from spire_learn.neural.cases import environment_seed_for
        from spire_learn.neural.collector import CollectorRecovery

        before = row(101, active=True, entered=1, visit=0)
        successors = (row(102, active=True, entered=1, visit=0),
                      row(103, active=True, entered=1, visit=0))
        driver = ScriptedDriver(
            "standalone", 2,
            reset_rows=lambda slot, call, seed, entry: before,
            steps=(lambda rows, actions: tuple(
                transition(source, successor)
                for source, successor in zip(rows, successors, strict=True)
            ),),
        )
        recovery = CollectorRecovery(7, 11, (4, 9))
        policy = FakePolicy(self.standalone_config)
        from spire_learn.neural.collector import Collector

        collector = Collector(
            driver,
            policy,
            self.standalone_config,
            controller_rng=random.Random(901),
            curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: 0.0),
            policy_generator=self.generator,
            recovery=recovery,
        )
        self.assertEqual(driver.reset_calls, [])
        self.assertEqual(driver.step_calls, [])
        self.assertEqual(collector.recovery_state, recovery)

        rollout = collector.collect(
            max_learned_decisions=4, max_env_actions=2, policy_version=0
        )
        self.assertEqual(driver.reset_calls[0][0], (
            environment_seed_for("train", replica=0, ordinal=7),
            environment_seed_for("train", replica=0, ordinal=8),
        ))
        self.assertEqual(tuple(item.task_id for item in rollout.records),
                         ("task-0-4", "task-1-9"))
        self.assertEqual(tuple(item.decision_id for item in rollout.records),
                         ("decision-11", "decision-11"))
        self.assertEqual(collector.recovery_state, CollectorRecovery(9, 12, (5, 10)))
        collector.close()

    def test_recovery_rejects_invalid_ordinals_width_and_previously_used_driver(self):
        from spire_learn.neural.collector import Collector, CollectorRecovery

        for args in (
            (True, 0, (0,)),
            (-1, 0, (0,)),
            (0, 2**64, (0,)),
            (0, 0, ()),
            (0, 0, [0]),
            (0, 0, (False,)),
            (0, 0, (2**64,)),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                CollectorRecovery(*args)

        recovery = CollectorRecovery(0, 0, (0,))
        cases = (
            ScriptedDriver("standalone", 2),
            ScriptedDriver("standalone", 2,
                           initial_rows=(row(1, active=True, entered=1, visit=0), None)),
        )
        for driver in cases:
            with self.subTest(driver=driver), self.assertRaises(Exception):
                Collector(
                    driver, FakePolicy(self.standalone_config), self.standalone_config,
                    controller_rng=random.Random(901), curriculum_rng=random.Random(1201),
                    budget=Budget(60.0, lambda: 0.0), policy_generator=self.generator,
                    recovery=recovery,
                )
            self.assertEqual(driver.reset_calls, [])
            self.assertEqual(driver.step_calls, [])

        used = ScriptedDriver("standalone", 2)
        used.accepted_steps = 1
        with self.assertRaisesRegex(Exception, "no accepted steps"):
            Collector(
                used, FakePolicy(self.standalone_config), self.standalone_config,
                controller_rng=random.Random(901), curriculum_rng=random.Random(1201),
                budget=Budget(60.0, lambda: 0.0), policy_generator=self.generator,
                recovery=CollectorRecovery(0, 0, (0, 0)),
            )
        self.assertEqual(used.reset_calls, [])
        self.assertEqual(used.step_calls, [])

    def test_recovery_snapshot_rejects_faulted_or_closed_collector(self):
        from spire_learn.neural.collector import Collector

        driver = ScriptedDriver("standalone", 2)
        collector = self.collector(driver, self.standalone_config)
        collector.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = collector.recovery_state

        failing = ScriptedDriver("standalone", 2, steps=())
        faulted = self.collector(failing, self.standalone_config)
        with self.assertRaises(AssertionError):
            faulted.collect(max_learned_decisions=8, max_env_actions=2, policy_version=0)
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            _ = faulted.recovery_state

    def test_stateless_reference_rows_are_fixed_and_never_request_values(self):
        import random as random_module
        from spire_learn.neural.collector import Collector

        before = row(110, active=True, entered=1, visit=0)
        after = row(111, active=True, entered=1, visit=0)
        driver = ScriptedDriver(
            "standalone", 2, reset_rows=lambda slot, call, seed, entry: before,
            steps=(lambda rows, actions: tuple(transition(source, after) for source in rows),),
        )
        policy = StatelessReferencePolicy()
        rng = random_module.Random(4411)
        collector = Collector(
            driver, policy, self.selection_config,
            controller_rng=random.Random(901), curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: 0.0), policy_generator=rng,
            learned_combat_policy=False,
        )
        rollout = collector.collect(
            max_learned_decisions=0, max_env_actions=2, policy_version="reference-policy-v1"
        )
        self.assertEqual(len(rollout.records), 2)
        self.assertEqual(rollout.counters.actor_rows, 0)
        self.assertEqual(rollout.counters.fixed_rows, 2)
        self.assertEqual(rollout.collection.learned_decisions, 0)
        self.assertEqual(rollout.collection.fixed_decisions, 2)
        self.assertEqual(rollout.collection.attempted_environment_slots, 2)
        self.assertEqual(rollout.collection.undispatched_fixed_decisions, 0)
        self.assertTrue(all(not record.actor_training_mask and record.next_value is None
                            for record in rollout.records))
        self.assertEqual(policy.value_calls, 0)
        self.assertTrue(all(generator is rng for generator in policy.generators))
        collector.close()

    def test_reference_policy_rejects_statistics_and_counts_undispatched_as_fixed(self):
        import random as random_module
        from spire_learn.neural.collector import Collector

        before = row(120, active=True, entered=1, visit=0)
        driver = ScriptedDriver("standalone", 2, initial_rows=(before, before), steps=())
        bad = StatelessReferencePolicy(include_statistics=True)
        collector = Collector(
            driver, bad, self.selection_config,
            controller_rng=random.Random(901), curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: 0.0), policy_generator=random_module.Random(4411),
            learned_combat_policy=False,
        )
        with self.assertRaisesRegex(Exception, "statless"):
            collector.collect(max_learned_decisions=0, max_env_actions=2,
                              policy_version="reference-policy-v1")
        self.assertEqual(driver.step_calls, [])

        clock = [0.0]
        expiring_driver = ScriptedDriver(
            "standalone", 2, initial_rows=(before, before), steps=()
        )

        class ExpiringReference(StatelessReferencePolicy):
            def act(self, frames, *, generator, deterministic=False):
                choices = super().act(frames, generator=generator, deterministic=deterministic)
                clock[0] = 61.0
                return choices

        expiring = Collector(
            expiring_driver, ExpiringReference(), self.selection_config,
            controller_rng=random.Random(901), curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: clock[0]),
            policy_generator=random_module.Random(4411), learned_combat_policy=False,
        )
        result = expiring.collect(max_learned_decisions=0, max_env_actions=2,
                                  policy_version="reference-policy-v1")
        self.assertEqual(result.collection.learned_decisions, 0)
        self.assertEqual(result.collection.fixed_decisions, 0)
        self.assertEqual(result.collection.undispatched_fixed_decisions, 2)
        self.assertEqual(result.collection.undispatched_actor_decisions, 0)
        self.assertEqual(expiring_driver.step_calls, [])
        collector.close()
        expiring.close()

    def test_default_frozen_neural_collector_keeps_learned_attribution(self):
        from spire_learn.neural.collector import Collector

        before = row(130, active=True, entered=1, visit=0)
        after = row(131, active=True, entered=1, visit=0)
        driver = ScriptedDriver(
            "standalone", 2, reset_rows=lambda slot, call, seed, entry: before,
            steps=(lambda rows, actions: tuple(transition(source, after) for source in rows),),
        )
        policy = FakePolicy(self.selection_config)
        collector = Collector(
            driver, policy, self.selection_config,
            controller_rng=random.Random(901), curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: 0.0), policy_generator=self.generator,
        )
        rollout = collector.collect(max_learned_decisions=2, max_env_actions=2,
                                    policy_version=0)
        self.assertTrue(collector.learned_combat_policy)
        self.assertEqual(rollout.counters.actor_rows, 2)
        self.assertEqual(rollout.counters.fixed_rows, 0)
        self.assertTrue(all(record.actor_training_mask for record in rollout.records))
        self.assertEqual(len(policy.value_inputs), 1)
        collector.close()


    def test_sampler_tracks_retained_slot_case_and_only_new_natural_settlements(self):
        cases = self.standalone_config.cases.for_mode("standalone")
        sampler = StrictCaseSampler(cases)
        start = (row(1, active=True, entered=1, visit=0), row(2, active=True, entered=1, visit=0))
        win = row(3, active=False, entered=1, settled=1, exit_value=exit_data())
        ongoing = row(4, active=True, entered=1, visit=0)
        fresh = row(5, active=True, entered=1, visit=0)
        death = row(6, active=False, entered=1, settled=1, exit_value=exit_data(outcome="death"))
        driver = ScriptedDriver("standalone", 2,
            reset_rows=lambda slot, call, seed, entry: start[slot] if call == 0 else fresh,
            steps=(lambda rows, actions: (transition(rows[0], win, terminated=True), transition(rows[1], ongoing)),
                   lambda rows, actions: (transition(rows[0], fresh), transition(rows[1], death, terminated=True))))
        policy = FakePolicy(self.standalone_config)
        collector = self.collector(driver, self.standalone_config, policy, case_sampler=sampler)
        first = collector.collect(max_learned_decisions=8, max_env_actions=4, policy_version=0)
        second = collector.collect(max_learned_decisions=8, max_env_actions=4, policy_version=0)
        third = collector.collect(max_learned_decisions=8, max_env_actions=4, policy_version=0)
        self.assertEqual(sampler.observed, [(cases[0].case_id, "victory"), (cases[1].case_id, "death")])
        self.assertEqual(len(sampler.chosen), 3)
        self.assertTrue(all(rng is collector.curriculum_rng for rng in sampler.rngs))
        self.assertEqual(tuple(item.source for item in first.records), tuple(item.frame for item in start))
        self.assertEqual(tuple(item.source for item in second.records), (fresh.frame, ongoing.frame))
        self.assertEqual(third.records, ())
        self.assertEqual(policy.act_inputs, [tuple(item.frame for item in start), (fresh.frame, ongoing.frame)])
        self.assertTrue(all(set(item.tensors) == {"player"} for frames in policy.act_inputs for item in frames))
        self.assertTrue(all(generator is self.generator for generator in policy.generator_args))
        self.assertEqual(second.collection.known_accepted_reset_slots, 3)
        self.assertEqual(second.collection.tasks_completed, 2)
        collector.close()

    def test_sampler_never_observes_censoring_truncation_invalid_batch_or_driver_fault(self):
        cases = self.standalone_config.cases.for_mode("standalone")
        active = row(1, active=True, entered=1, visit=0)
        ended = row(2, active=False, entered=1, settled=1, exit_value=exit_data())
        def fault(rows, actions):
            raise RuntimeError("native delivery unknown")
        scripts = (
            (lambda rows, actions: tuple(transition(source, active) for source in rows), None, 2),
            (lambda rows, actions: tuple(transition(source,
                row(3, active=False, entered=1), terminated=True) for source in rows), None, 0),
            (lambda rows, actions: tuple(transition(source, active, truncated=True) for source in rows), ValueError, None),
            (lambda rows, actions: (transition(rows[0], ended, terminated=True), transition(ended, active)), Exception, None),
            (fault, RuntimeError, None),
        )
        for script, error, incomplete in scripts:
            with self.subTest(error=error):
                sampler = StrictCaseSampler(cases)
                driver = ScriptedDriver("standalone", 2, steps=(script,))
                collector = self.collector(driver, self.standalone_config, case_sampler=sampler)
                if error is None:
                    result = collector.collect(max_learned_decisions=2, max_env_actions=2, policy_version=0)
                    self.assertEqual(result.collection.tasks_completed, 0)
                    self.assertEqual(result.collection.incomplete_decisions, incomplete)
                else:
                    with self.assertRaises(error):
                        collector.collect(max_learned_decisions=2, max_env_actions=2, policy_version=0)
                self.assertEqual(sampler.observed, [])
                collector.close()
        # A sticky exit returned directly by reset is not a newly delivered settlement.
        sampler = StrictCaseSampler(cases)
        driver = ScriptedDriver("standalone", 2, reset_rows=lambda *args: ended)
        collector = self.collector(driver, self.standalone_config, case_sampler=sampler)
        for _ in range(2):
            collector.collect(max_learned_decisions=4, max_env_actions=4, policy_version=0)
        self.assertEqual(sampler.observed, [])
        self.assertEqual(len(driver.reset_calls), 1)
        self.assertEqual(driver.step_calls, [])
        collector.close()

    def test_sampler_observer_error_preserves_delivered_batch_and_control_exception(self):
        cases = self.standalone_config.cases.for_mode("standalone")
        ended = row(3, active=False, entered=1, settled=1, exit_value=exit_data())
        for error in (RuntimeError("observer failure"), KeyboardInterrupt("observer control failure")):
            with self.subTest(error=error):
                sampler = StrictCaseSampler(cases, observe_error=error)
                driver = ScriptedDriver("standalone", 2, steps=(lambda rows, actions:
                    (transition(rows[0], ended, terminated=True), transition(rows[1], rows[1])),))
                def close_failure():
                    driver.close_calls += 1
                    raise ValueError("secondary close failure")
                driver.close = close_failure
                collector = self.collector(driver, self.standalone_config, case_sampler=sampler)
                with self.assertRaises(type(error)) as raised:
                    collector.collect(max_learned_decisions=8, max_env_actions=8, policy_version=0)
                self.assertIs(raised.exception, error)
                partial = collector.partial_rollout
                self.assertEqual(len(partial.records), 2)
                self.assertEqual(partial.counters.rows, 2)
                self.assertEqual(partial.collection.tasks_completed, 1)
                self.assertEqual(partial.collection.known_accepted_environment_slots, 2)
                self.assertEqual(partial.collection.delivered_environment_slots, 2)
                self.assertEqual(partial.collection.uncertain_environment_slots, 0)
                self.assertEqual(partial.collection.incomplete_environment_slots, 0)
                self.assertEqual(partial.collection.fault.uncertain_slots, 0)
                self.assertEqual(partial.collection.fault.operation_slots, 0)
                self.assertEqual(partial.collection.fault.phase, "case-sampler-observe")
                self.assertEqual(partial.collection.incomplete_decisions, 1)
                with self.assertRaisesRegex(RuntimeError, "cannot be retried"):
                    collector.collect(max_learned_decisions=8, max_env_actions=8, policy_version=0)
                self.assertEqual(len(driver.step_calls), 1)
                self.assertEqual(len(driver.reset_calls), 1)
                self.assertEqual(driver.close_calls, 1)
                collector.close()

    def test_sampler_choice_error_or_invalid_case_stops_before_reset(self):
        cases = self.standalone_config.cases.for_mode("standalone")
        for sampler in (StrictCaseSampler(cases, choose_error=RuntimeError("choice failure")),
                        StrictCaseSampler(cases)):
            if sampler.choose_error is None:
                sampler.choose_case = lambda cases, *, rng: replace(cases[0], case_id="foreign")
            driver = ScriptedDriver("standalone", 2)
            collector = self.collector(driver, self.standalone_config, case_sampler=sampler)
            with self.assertRaises(Exception):
                collector.collect(max_learned_decisions=4, max_env_actions=4, policy_version=0)
            self.assertEqual(driver.reset_calls, [])
            self.assertEqual(driver.step_calls, [])
            self.assertEqual(collector.partial_rollout.collection.known_accepted_reset_slots, 0)
            self.assertEqual(collector.partial_rollout.collection.uncertain_reset_slots, 0)
            with self.assertRaisesRegex(RuntimeError, "cannot be retried"):
                collector.collect(max_learned_decisions=4, max_env_actions=4, policy_version=0)

    def test_allocated_choices_do_not_claim_accepted_resets_on_driver_failure(self):
        sampler = StrictCaseSampler(self.standalone_config.cases.for_mode("standalone"))
        driver = ScriptedDriver("standalone", 2)
        def reset_failure(*, seeds, entry_ids, reset_mask):
            driver.reset_calls.append((seeds, entry_ids, reset_mask))
            raise RuntimeError("reset delivery unknown")
        driver.reset = reset_failure
        collector = self.collector(driver, self.standalone_config, case_sampler=sampler)
        with self.assertRaisesRegex(RuntimeError, "reset delivery unknown"):
            collector.collect(max_learned_decisions=4, max_env_actions=4, policy_version=0)
        partial = collector.partial_rollout
        self.assertEqual(len(sampler.chosen), 2)
        self.assertEqual(partial.collection.known_accepted_reset_slots, 0)
        self.assertEqual(partial.collection.delivered_reset_slots, 0)
        self.assertEqual(partial.collection.uncertain_reset_slots, 2)
        self.assertEqual(partial.records, ())
        self.assertEqual(sampler.observed, [])
        self.assertEqual(driver.step_calls, [])
        self.assertEqual(driver.close_calls, 1)

    def test_sampler_constructor_rejects_wrong_mode_operation_or_catalog_before_effects(self):
        cases = self.standalone_config.cases.for_mode("standalone")
        for config in (self.selection_config, self.run_config):
            driver = ScriptedDriver(config.mode, config.rollout_rows)
            with self.assertRaises(Exception):
                self.collector(driver, config, case_sampler=StrictCaseSampler(cases))
            self.assertEqual(driver.reset_calls, [])
            self.assertEqual(driver.step_calls, [])
        driver = ScriptedDriver("standalone", 2)
        with self.assertRaisesRegex(ValueError, "catalog mismatch"):
            self.collector(driver, self.standalone_config, case_sampler=StrictCaseSampler(cases[:-1]))
        self.assertEqual(driver.reset_calls, [])
        driver = ScriptedDriver("standalone", 2, initial_rows=(row(1, active=True, entered=1, visit=0), None))
        with self.assertRaises(Exception):
            self.collector(driver, self.standalone_config, case_sampler=StrictCaseSampler(cases))
        self.assertEqual(driver.reset_calls, [])

    def test_real_sampler_validates_and_receives_collector_natural_outcomes(self):
        from spire_learn.neural.curriculum import CurriculumSampler
        from spire_learn.neural.scenario_support import content_sha256
        cases = self.standalone_config.cases.for_mode("standalone")
        annotations = {case.case_id: {
            "family": "lesson", "deck_stage": "early", "lesson_kind": "group", "category": "group:all",
            "target_card_ids": (case.entry["deck"][0]["card"],), "encounter_tier": "act1:monster",
            "lineage_id": case.lineage_id, "composition_sha256": content_sha256(sorted(
                case.entry["deck"], key=lambda card: (card["card"], card["upgrades"]))),
            "composition_scope": "novel_composition"} for case in cases}
        sampler = CurriculumSampler(annotations, distribution_sha256="a" * 64, categories=("group:all",))
        ended = row(3, active=False, entered=1, settled=1, exit_value=exit_data())
        driver = ScriptedDriver("standalone", 2, steps=(lambda rows, actions:
            tuple(transition(source, ended, terminated=True) for source in rows),))
        collector = self.collector(driver, self.standalone_config, case_sampler=sampler)
        result = collector.collect(max_learned_decisions=2, max_env_actions=2, policy_version=0)
        self.assertEqual(sampler.snapshot()["completed"], 2)
        self.assertEqual(sampler.snapshot()["category_stats"]["group:all"]["recent"], ["victory", "victory"])
        self.assertEqual(result.collection.tasks_completed, 2)
        collector.close()

    def test_explicit_none_sampler_preserves_default_rng_frames_actions_and_counters(self):
        collectors, policies, drivers = [], [], []
        for kwargs in ({}, {"case_sampler": None}):
            driver = ScriptedDriver("standalone", 2, steps=(lambda rows, actions:
                tuple(transition(source, source) for source in rows),))
            policy = FakePolicy(self.standalone_config)
            collector = self.collector(driver, self.standalone_config, policy, **kwargs)
            collectors.append(collector)
            drivers.append(driver)
            policies.append(policy)
        outputs = [collector.collect(max_learned_decisions=2, max_env_actions=2, policy_version=0)
                   for collector in collectors]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(drivers[0].events, drivers[1].events)
        self.assertEqual(policies[0].act_inputs, policies[1].act_inputs)
        self.assertEqual(policies[0].value_inputs, policies[1].value_inputs)
        expected_rng = random.Random(1201)
        cases = self.standalone_config.cases.for_mode("standalone")
        expected_ids = [cases[expected_rng.randrange(len(cases))].case_id for _ in range(2)]
        self.assertEqual([item.case_id for item in outputs[0].collection.seed_allocations], expected_ids)
        for collector in collectors:
            self.assertEqual(collector.curriculum_rng.getstate(), expected_rng.getstate())
            collector.close()


if __name__ == "__main__":
    unittest.main()
