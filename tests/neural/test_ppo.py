"""Opt-in direct-PPO checks over learner-authored frames and frozen rollouts."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.neural._support import CandidateInput, accepted_schema_and_config, authored_frame


_TENSOR_GATE = "SPIRE_LEARN_RUN_NEURAL_TENSOR"


def setUpModule() -> None:
    if os.environ.get(_TENSOR_GATE) == "1":
        from tests.neural._sample_guard import activate_from_environment

        activate_from_environment()


def _api(test: unittest.TestCase) -> SimpleNamespace:
    try:
        from spire_learn.neural import distribution, features, model, ppo
        from spire_learn.neural.contracts import (
            ContractError,
            CumulativeCounters,
            NeuralIdentity,
            PolicyChoice,
            Rollout,
            StepRecord,
            TaskSignal,
        )
        from spire_learn.neural.ppo import UpdateFault
        from spire_learn.experiments.budget import Budget
    except ImportError as error:
        test.fail(f"opt-in neural PPO API is unavailable: {error}")
    return SimpleNamespace(
        distribution=distribution,
        features=features,
        model=model,
        ppo=ppo,
        ContractError=ContractError,
        CumulativeCounters=CumulativeCounters,
        NeuralIdentity=NeuralIdentity,
        PolicyChoice=PolicyChoice,
        Rollout=Rollout,
        StepRecord=StepRecord,
        TaskSignal=TaskSignal,
        UpdateFault=UpdateFault,
        Budget=Budget,
    )


def _test_identity(identity, ppo_config):
    encoded = json.dumps(
        asdict(ppo_config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    hashes = dict(identity.config_hashes)
    hashes["ppo"] = hashlib.sha256(encoded).hexdigest()
    return replace(identity, config_hashes=hashes)


def _tensor_bytes(tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _named_parameter_byte_hashes(policy) -> dict[str, object]:
    """Hash named contiguous tensor bytes directly, without a pickle archive."""
    parameter_hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for name, parameter in sorted(policy.named_parameters()):
        name_bytes = name.encode("utf-8")
        parameter_bytes = _tensor_bytes(parameter)
        parameter_hashes[name] = hashlib.sha256(parameter_bytes).hexdigest()
        aggregate.update(len(name_bytes).to_bytes(4, "big"))
        aggregate.update(name_bytes)
        aggregate.update(len(parameter_bytes).to_bytes(8, "big"))
        aggregate.update(parameter_bytes)
    return {
        "aggregate_sha256": aggregate.hexdigest(),
        "parameters_sha256": parameter_hashes,
    }


def _record(api, *, frame, choice, index: int, reward: float, training: bool = True,
            id_prefix: str = "clean"):
    if training:
        signal = api.TaskSignal(reward, True, False, False)
        return api.StepRecord(
            stream_id=f"{id_prefix}-stream-{index}",
            task_id=f"{id_prefix}-task-{index}",
            decision_id=f"{id_prefix}-decision-{index}",
            source=frame,
            successor=frame,
            native_reward=0.0,
            terminated=False,
            truncated=False,
            task_signal=signal,
            choice=choice,
            actor_training_mask=True,
            next_value=choice.value,
        )
    return api.StepRecord(
        stream_id=f"{id_prefix}-fixed-stream-{index}",
        task_id=f"{id_prefix}-fixed-task-{index}",
        decision_id=f"{id_prefix}-fixed-decision-{index}",
        source=frame,
        successor=frame,
        native_reward=0.0,
        terminated=False,
        truncated=False,
        task_signal=api.TaskSignal(1_000_000.0, False, True, True),
        choice=api.PolicyChoice(choice.original_action_index),
        actor_training_mask=False,
        next_value=None,
    )


def _rollout(api, *, identity, mode: str, split: str, records, counters=None):
    records = tuple(records)
    actor_rows = sum(record.actor_training_mask for record in records)
    fixed_rows = len(records) - actor_rows
    if counters is None:
        counters = api.CumulativeCounters(
            rows=len(records),
            actor_rows=actor_rows,
            fixed_rows=fixed_rows,
            task_terminals=sum(record.task_signal.task_terminal for record in records),
            environment_terminations=sum(record.terminated for record in records),
            environment_truncations=sum(record.truncated for record in records),
        )
    return api.Rollout(mode=mode, split=split, identity=identity, records=records, counters=counters)


class NeuralPPOTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get(_TENSOR_GATE) != "1":
            raise unittest.SkipTest("set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
        cls.schema, cls.config = accepted_schema_and_config()
        cls.frame_candidates = (
            CandidateInput(
                index=2,
                verb="play_card",
                source_group="combat_cards",
                source_row=0,
                target_group="enemies",
                target_row=0,
            ),
            CandidateInput(
                index=9,
                verb="play_card",
                source_group="combat_cards",
                source_row=0,
                target_group="enemies",
                target_row=1,
            ),
        )
        cls.frame = authored_frame(cls.schema, candidates=cls.frame_candidates)
        from spire_learn.neural.schema import validate_frame

        validate_frame(cls.frame, cls.schema)
        enemy_spec = cls.schema.tensors["enemies"]
        enemy_row_bytes = len(enemy_spec.columns) * 8
        enemy_data = cls.frame.tensors["enemies"].data
        if enemy_data[:enemy_row_bytes] == enemy_data[enemy_row_bytes:2 * enemy_row_bytes]:
            raise AssertionError("PPO target rows must have distinct public enemy features")
        cls.mode = cls.config.mode

    def setUp(self):
        import torch

        self.torch = torch
        self.api = _api(self)
        self.policy = self.api.model.build_policy(
            self.schema,
            self.config.model,
            seed=2207,
            device="cpu",
        )
        self.ppo_config = replace(self.config.ppo, epochs=1, minibatch_rows=64)
        self.identity = _test_identity(self.config.accepted_identities[0], self.ppo_config)

    def _choices(self, count: int = 2):
        actions = tuple(candidate.index for candidate in self.frame_candidates[:count])
        if count < 1 or len(actions) != count:
            raise ValueError("PPO authored-frame queries require one or two legal candidates")
        return self._choices_for_actions(actions)

    def _choices_for_actions(self, actions, *, policy=None):
        policy = self.policy if policy is None else policy
        # Deterministic act supplies the frame's value/version; the alternate
        # legal rows below use exact current-policy log-probs and no sampling.
        mode_choice = policy.act(
            (self.frame,),
            generator=self.torch.Generator(device="cpu").manual_seed(77),
            deterministic=True,
        )[0]
        frames = tuple(self.frame for _ in actions)
        batch = self.api.features.encode_frames(frames, self.schema, device="cpu")
        output = policy.forward(batch)
        distribution = self.api.distribution.masked_categorical(output.logits, batch.action_mask)
        action_tensor = self.torch.tensor(actions, dtype=self.torch.int64)
        row_tensor = self.torch.arange(len(actions), dtype=self.torch.int64)
        legal = batch.action_mask[row_tensor, action_tensor].to(dtype=self.torch.bool)
        self.assertTrue(self.torch.all(legal).item(), "PPO fixture action must be legal")
        old_log_probs = distribution.log_prob(action_tensor).detach().cpu().tolist()
        return tuple(
            replace(
                mode_choice,
                original_action_index=action,
                log_probability=float(old_log_probability),
            )
            for action, old_log_probability in zip(actions, old_log_probs, strict=True)
        )

    def _optimizer(self, policy=None):
        policy = self.policy if policy is None else policy
        return self.torch.optim.Adam(
            policy.parameters(),
            lr=self.ppo_config.learning_rate,
            eps=self.ppo_config.adam_eps,
        )

    def _budget(self):
        return self.api.Budget(30.0, lambda: 0.0)

    def _update(self, policy, optimizer, rollout, *, seed=91, budget=None):
        return self.api.ppo.ppo_update(
            policy,
            optimizer,
            rollout,
            self.ppo_config,
            shuffle_generator=self.torch.Generator(device="cpu").manual_seed(seed),
            budget=self._budget() if budget is None else budget,
        )

    def test_actor_only_update_has_nonzero_gradient_and_changes_named_parameter_hash(self):
        self.ppo_config = replace(
            self.config.ppo,
            epochs=1,
            minibatch_rows=2,
            value_coefficient=0.0,
            entropy_coefficient=0.0,
        )
        self.identity = _test_identity(self.config.accepted_identities[0], self.ppo_config)
        choices = self._choices()
        frames = (self.frame, self.frame)
        batch = self.api.features.encode_frames(frames, self.schema, device="cpu")
        output = self.policy.forward(batch)
        distribution = self.api.distribution.masked_categorical(output.logits, batch.action_mask)
        actions = self.torch.tensor(
            [choice.original_action_index for choice in choices], dtype=self.torch.int64
        )
        new_logp = distribution.log_prob(actions)
        old_logp = self.torch.tensor(
            [choice.log_probability for choice in choices], dtype=self.torch.float32
        )
        ratios_before_update = self.torch.exp(new_logp - old_logp)
        self.assertTrue(self.torch.allclose(ratios_before_update, self.torch.ones_like(ratios_before_update)))
        self.assertEqual(int(batch.action_mask[0].sum().item()), 2)

        actor_advantages = self.torch.tensor((1.0, -1.0), dtype=self.torch.float32)
        actor_objective = -(new_logp * actor_advantages).mean()
        actor_gradients = self.torch.autograd.grad(
            actor_objective,
            tuple(self.policy.parameters()),
            allow_unused=True,
        )
        present_gradients = [gradient for gradient in actor_gradients if gradient is not None]
        self.assertTrue(present_gradients)
        self.assertTrue(all(self.torch.isfinite(gradient).all() for gradient in present_gradients))
        self.assertTrue(any(
            self.torch.count_nonzero(gradient).item() > 0
            for gradient in present_gradients
        ), "actor objective must produce a nonzero gradient")

        original_hashes = _named_parameter_byte_hashes(self.policy)

        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=float(choice.value) + (1.0 if index == 0 else -1.0),
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api, identity=self.identity, mode=self.mode, split="train", records=records
        )
        metrics = self._update(self.policy, self._optimizer(), rollout)
        updated_hashes = _named_parameter_byte_hashes(self.policy)
        changed_parameters = [
            name for name, digest in original_hashes["parameters_sha256"].items()
            if updated_hashes["parameters_sha256"][name] != digest
        ]
        self.assertNotEqual(original_hashes["aggregate_sha256"], updated_hashes["aggregate_sha256"])
        self.assertTrue(changed_parameters, "actor-only optimizer update must change parameter bytes")
        self.assertEqual((metrics.attempted_steps, metrics.accepted_steps), (1, 1))
        self.assertIsNone(metrics.stop_reason)
        print("TASK4_PPO_NAMED_PARAMETER_BYTE_HASHES=" + json.dumps({
            "before": original_hashes,
            "after": updated_hashes,
            "changed_parameters": changed_parameters,
        }, sort_keys=True))

    def test_positive_and_negative_advantages_clip_in_the_correct_directions(self):
        choices = self._choices()
        ratios = (1.3, 0.7)
        # This arithmetic case imposes known ratios through stored log-probs;
        # the actor-only test uses exact current-distribution log-probs.
        rewards = (float(choices[0].value) + 1.0, float(choices[1].value) - 1.0)
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=replace(
                    choice,
                    log_probability=choice.log_probability - math.log(ratios[index]),
                ),
                index=index,
                reward=rewards[index],
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api, identity=self.identity, mode=self.mode, split="train", records=records
        )
        metrics = self._update(self.policy, self._optimizer(), rollout)

        self.assertIs(type(metrics.policy_loss), float)
        self.assertIs(type(metrics.value_loss), float)
        self.assertIs(type(metrics.entropy), float)
        self.assertTrue(all(math.isfinite(value) for value in (
            metrics.policy_loss, metrics.value_loss, metrics.entropy,
        )))
        # For normalized advantages (+1,-1), ratios (1.3,0.7), and epsilon
        # 0.2, the clipped surrogate mean is 0.2 and its loss is -0.2.
        self.assertAlmostEqual(metrics.policy_loss, -0.2, places=4)
        self.assertGreaterEqual(metrics.value_loss, 0.0)
        self.assertGreaterEqual(metrics.entropy, 0.0)
        self.assertIs(type(metrics.attempted_steps), int)
        self.assertIs(type(metrics.accepted_steps), int)
        self.assertGreaterEqual(metrics.attempted_steps, 0)
        self.assertGreaterEqual(metrics.accepted_steps, 0)
        self.assertEqual((metrics.attempted_steps, metrics.accepted_steps), (1, 1))
        self.assertIsNone(metrics.stop_reason)
        self.assertTrue(all(
            parameter.grad is None or self.torch.isfinite(parameter.grad).all()
            for parameter in self.policy.parameters()
        ))

    def test_tiny_nonconstant_advantages_use_epsilon_normalization_and_keep_actor_signal(self):
        from spire_learn.neural.returns import compute_gae

        self.ppo_config = replace(
            self.config.ppo,
            epochs=1,
            minibatch_rows=2,
            value_coefficient=0.0,
            entropy_coefficient=0.0,
            max_grad_norm=100.0,
        )
        self.identity = _test_identity(self.config.accepted_identities[0], self.ppo_config)
        choices = self._choices(2)
        deltas = (1.0e-8, -1.0e-8)
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=float(choice.value) + deltas[index],
                id_prefix="tiny-advantages",
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=records,
        )

        returns = compute_gae(
            records,
            gamma=self.ppo_config.gamma,
            lam=self.ppo_config.gae_lambda,
        )
        raw_advantages = self.torch.tensor(returns.advantages, dtype=self.torch.float32)
        population_std = raw_advantages.std(unbiased=False)
        self.assertGreater(population_std.item(), 0.0)
        self.assertLess(population_std.item(), self.torch.finfo(raw_advantages.dtype).eps)
        approved_normalization = (raw_advantages - raw_advantages.mean()) / (population_std + 1e-8)
        expected_normalized = self.torch.tensor((0.5, -0.5), dtype=self.torch.float32)
        self.torch.testing.assert_close(approved_normalization, expected_normalized, rtol=1e-3, atol=1e-3)

        frames = (self.frame, self.frame)
        batch = self.api.features.encode_frames(frames, self.schema, device="cpu")
        output = self.policy.forward(batch)
        distribution = self.api.distribution.masked_categorical(output.logits, batch.action_mask)
        actions = self.torch.tensor(
            [choice.original_action_index for choice in choices], dtype=self.torch.int64
        )
        expected_actor_loss = -(
            distribution.log_prob(actions) * approved_normalization
        ).mean()
        target_values = self.torch.tensor(returns.targets, dtype=self.torch.float32)
        expected_actor_loss = (
            expected_actor_loss
            + self.ppo_config.value_coefficient * self.torch.square(output.values - target_values).mean()
            - self.ppo_config.entropy_coefficient * distribution.entropy.mean()
        )
        expected_gradients = self.torch.autograd.grad(
            expected_actor_loss,
            tuple(self.policy.parameters()),
            allow_unused=True,
        )
        self.assertTrue(any(
            gradient is not None and self.torch.count_nonzero(gradient).item() > 0
            for gradient in expected_gradients
        ), "epsilon-normalized tiny advantages must preserve a nonzero actor signal")

        optimizer = self._optimizer()
        self._update(self.policy, optimizer, rollout)
        for parameter, expected_gradient in zip(
            self.policy.parameters(), expected_gradients, strict=True
        ):
            self.assertEqual(parameter.grad is None, expected_gradient is None)
            if expected_gradient is not None:
                self.torch.testing.assert_close(
                    parameter.grad,
                    expected_gradient,
                    rtol=1e-4,
                    atol=1e-7,
                )

    def test_first_post_admission_budget_clock_error_is_fault_and_consumes_rollout(self):
        choice = self._choices(1)[0]
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=(_record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=0,
                reward=float(choice.value),
                id_prefix="budget-clock-fault",
            ),),
        )
        clock_error = RuntimeError("injected post-admission clock failure")
        clock_calls = 0

        def clock():
            nonlocal clock_calls
            clock_calls += 1
            if clock_calls == 1:
                return 0.0
            raise clock_error

        budget = self.api.Budget(30.0, clock)
        optimizer = self._optimizer()
        caught = None
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            try:
                self._update(self.policy, optimizer, rollout, budget=budget)
            except BaseException as error:
                caught = error
            else:
                self.fail("a post-admission clock error must not become a normal wall-time stop")

            self.assertIsInstance(caught, self.api.UpdateFault)
            if not isinstance(caught, self.api.UpdateFault):
                return
            self.assertIs(caught.__cause__, clock_error)
            self.assertEqual(caught.phase, "budget-check")
            self.assertEqual(
                (caught.attempted_steps, caught.accepted_steps, caught.uncertain_steps),
                (0, 0, 0),
            )
            self.assertEqual(caught.admitted_behavior_policy_version, 0)
            self.assertEqual(caught.resulting_behavior_policy_version, 0)
            self.assertEqual(caught.consumed_decisions, 1)
            self.assertEqual(step.call_count, 0)

        with self.assertRaises(self.api.ContractError):
            self._update(self.policy, optimizer, rollout, budget=self._budget())

    def test_fixed_rows_and_metadata_poison_do_not_change_training_gradients(self):
        choices = self._choices()
        rewards = (float(choices[0].value) + 1.0, float(choices[1].value) - 1.0)
        base_records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=rewards[index],
                id_prefix="base",
            )
            for index, choice in enumerate(choices)
        )
        base_rollout = _rollout(
            self.api, identity=self.identity, mode=self.mode, split="train", records=base_records
        )

        poison_policy = self.api.model.build_policy(
            self.schema, self.config.model, seed=2207, device="cpu"
        )
        poison_choices = self._choices_for_actions((2, 9), policy=poison_policy)
        poison_actor = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=rewards[index],
                id_prefix="seed-routing-visit-rng-poison",
            )
            for index, choice in enumerate(poison_choices)
        )
        fixed_record = _record(
            self.api,
            frame=self.frame,
            choice=poison_choices[0],
            index=2,
            reward=0.0,
            training=False,
            id_prefix="privileged-fixed-controller-metadata",
        )
        poison_counters = self.api.CumulativeCounters(
            rows=30,
            actor_rows=20,
            fixed_rows=10,
            task_terminals=15,
            environment_terminations=2,
            environment_truncations=1,
        )
        poison_rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=poison_actor + (fixed_record,),
            counters=poison_counters,
        )

        self._update(self.policy, self._optimizer(self.policy), base_rollout)
        self._update(
            poison_policy,
            self._optimizer(poison_policy),
            poison_rollout,
            seed=91,
        )
        base_gradients = dict(self.policy.named_parameters())
        poison_gradients = dict(poison_policy.named_parameters())
        self.assertEqual(base_gradients.keys(), poison_gradients.keys())
        self.assertTrue(any(
            parameter.grad is not None and self.torch.count_nonzero(parameter.grad).item() > 0
            for parameter in base_gradients.values()
        ), "fixed-row comparison requires a nonzero actor gradient")
        self.assertTrue(any(
            parameter.grad is not None and self.torch.count_nonzero(parameter.grad).item() > 0
            for parameter in poison_gradients.values()
        ), "poisoned fixed-row comparison requires a nonzero actor gradient")
        for name in base_gradients:
            left = base_gradients[name].grad
            right = poison_gradients[name].grad
            self.assertEqual(left is None, right is None, name)
            if left is not None:
                self.assertTrue(self.torch.isfinite(left).all(), name)
                self.torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)

    def test_selection_rollout_is_rejected_before_optimizer_mutation(self):
        choices = self._choices()
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=0.0,
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api, identity=self.identity, mode=self.mode, split="selection", records=records
        )
        optimizer = self._optimizer()
        before = [parameter.detach().clone() for parameter in self.policy.parameters()]
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(self.api.ContractError):
                self._update(self.policy, optimizer, rollout)
        self.assertEqual(step.call_count, 0)
        for original, current in zip(before, self.policy.parameters(), strict=True):
            self.assertTrue(self.torch.equal(original, current))

    def test_multiple_epochs_admit_one_behavior_version_for_every_minibatch(self):
        self.ppo_config = replace(self.config.ppo, epochs=2, minibatch_rows=1)
        self.identity = _test_identity(self.config.accepted_identities[0], self.ppo_config)
        choices = self._choices(2)
        self.assertEqual(len({choice.behavior_policy_version for choice in choices}), 1)
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=float(choice.value),
                id_prefix="multi-epoch",
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=records,
        )

        metrics = self._update(self.policy, self._optimizer(), rollout)

        self.assertEqual((metrics.attempted_steps, metrics.accepted_steps), (4, 4))
        self.assertIsNone(metrics.stop_reason)

    def test_stale_behavior_version_is_rejected_before_optimizer_step(self):
        choice = self._choices(1)[0]
        version = choice.behavior_policy_version
        stale_version = version + 1 if type(version) is int else f"{version}-stale"
        stale_choice = replace(choice, behavior_policy_version=stale_version)
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=(_record(
                self.api,
                frame=self.frame,
                choice=stale_choice,
                index=0,
                reward=float(stale_choice.value),
                id_prefix="stale-version",
            ),),
        )
        optimizer = self._optimizer()

        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(self.api.ContractError):
                self._update(self.policy, optimizer, rollout)

        self.assertEqual(step.call_count, 0)

    def test_wall_time_stop_consumes_admitted_rollout_and_blocks_a_copied_replay(self):
        choices = self._choices(2)
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=float(choice.value),
                id_prefix="wall-stop",
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=records,
        )
        copied_rollout = replace(rollout, records=tuple(rollout.records))
        now = [0.0]
        budget = self.api.Budget(1.0, lambda: now[0])
        now[0] = 2.0
        optimizer = self._optimizer()

        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            stopped = self._update(self.policy, optimizer, rollout, budget=budget)
            self.assertEqual(step.call_count, 0)
            self.assertEqual(stopped.stop_reason, "wall_time")
            self.assertEqual((stopped.attempted_steps, stopped.accepted_steps), (0, 0))

            with self.assertRaises(self.api.ContractError):
                self._update(self.policy, optimizer, copied_rollout)

        self.assertEqual(step.call_count, 0)

    def test_uncertain_step_fault_keeps_mutated_prefix_and_never_retries(self):
        self.ppo_config = replace(self.config.ppo, epochs=2, minibatch_rows=1)
        self.identity = _test_identity(self.config.accepted_identities[0], self.ppo_config)
        choices = self._choices(2)
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=float(choice.value) + (1.0 if index == 0 else -1.0),
                id_prefix="uncertain-step",
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=records,
        )
        optimizer = self._optimizer()
        real_step = optimizer.step
        calls = 0
        before_update_hashes = _named_parameter_byte_hashes(self.policy)
        accepted_prefix_hashes: dict[str, object] | None = None
        after_uncertain_hashes: dict[str, object] | None = None
        forced_mutation_before: str | None = None
        forced_mutation_after: str | None = None
        forced_parameter_name = next(iter(dict(self.policy.named_parameters())))
        forced_parameter = dict(self.policy.named_parameters())[forced_parameter_name]

        def mutate_then_raise(*args, **kwargs):
            nonlocal calls, accepted_prefix_hashes, after_uncertain_hashes
            nonlocal forced_mutation_before, forced_mutation_after
            calls += 1
            real_step(*args, **kwargs)
            if calls == 1:
                accepted_prefix_hashes = _named_parameter_byte_hashes(self.policy)
            if calls == 2:
                forced_mutation_before = hashlib.sha256(_tensor_bytes(forced_parameter)).hexdigest()
                # Make a concrete tensor-byte mutation after the real step and
                # before reporting its uncertain return.
                with self.torch.no_grad():
                    forced_parameter.view(-1)[0].add_(1.0)
                forced_mutation_after = hashlib.sha256(_tensor_bytes(forced_parameter)).hexdigest()
                after_uncertain_hashes = _named_parameter_byte_hashes(self.policy)
                raise RuntimeError("injected optimizer mutation followed by uncertain return")

        with patch.object(optimizer, "step", new=mutate_then_raise):
            with self.assertRaises(self.api.UpdateFault) as caught:
                self._update(self.policy, optimizer, rollout)
            fault = caught.exception
            self.assertEqual((fault.attempted_steps, fault.accepted_steps, fault.uncertain_steps),
                             (2, 1, 1))
            self.assertTrue(fault.phase)
            self.assertIsInstance(fault.__cause__, RuntimeError)
            self.assertIn("injected optimizer mutation", str(fault.__cause__))
            self.assertEqual(calls, 2)
            self.assertIsNotNone(forced_mutation_before)
            self.assertIsNotNone(forced_mutation_after)
            self.assertNotEqual(forced_mutation_before, forced_mutation_after)
            self.assertNotEqual(
                before_update_hashes["parameters_sha256"][forced_parameter_name],
                forced_mutation_after,
                "the uncertainly mutated parameter bytes must differ from pre-update bytes",
            )
            self.assertIsNotNone(accepted_prefix_hashes)
            self.assertNotEqual(
                before_update_hashes["aggregate_sha256"],
                accepted_prefix_hashes["aggregate_sha256"],
                "the accepted optimizer prefix must change named parameter bytes",
            )
            self.assertIsNotNone(after_uncertain_hashes)
            self.assertNotEqual(
                before_update_hashes["aggregate_sha256"],
                after_uncertain_hashes["aggregate_sha256"],
            )
            self.assertEqual(_named_parameter_byte_hashes(self.policy), after_uncertain_hashes)

            with self.assertRaises((self.api.ContractError, self.api.UpdateFault)):
                self._update(self.policy, optimizer, rollout)
            self.assertEqual(calls, 2)

    def test_process_control_exception_propagates_with_optimizer_prefix_note(self):
        self.ppo_config = replace(self.config.ppo, epochs=2, minibatch_rows=1)
        self.identity = _test_identity(self.config.accepted_identities[0], self.ppo_config)
        choices = self._choices(2)
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=float(choice.value),
                id_prefix="process-control",
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=records,
        )
        optimizer = self._optimizer()
        real_step = optimizer.step
        calls = 0
        interrupt = KeyboardInterrupt("injected process stop")

        def interrupt_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise interrupt
            return real_step(*args, **kwargs)

        with patch.object(optimizer, "step", new=interrupt_second):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self._update(self.policy, optimizer, rollout)

        self.assertIs(caught.exception, interrupt)
        self.assertEqual(calls, 2)
        notes = getattr(interrupt, "__notes__", ())
        self.assertTrue(any(
            "phase='optimizer.step'" in note
            and "attempted_steps=2" in note
            and "accepted_steps=1" in note
            and "uncertain_steps=1" in note
            and "admitted_behavior_policy_version=0" in note
            and "resulting_behavior_policy_version=1" in note
            and "consumed_decisions=2" in note
            for note in notes
        ))

    def test_later_forward_interrupt_preserves_an_accepted_optimizer_prefix_note(self):
        self.ppo_config = replace(self.config.ppo, epochs=1, minibatch_rows=1)
        self.identity = _test_identity(self.config.accepted_identities[0], self.ppo_config)
        choices = self._choices(2)
        records = tuple(
            _record(
                self.api,
                frame=self.frame,
                choice=choice,
                index=index,
                reward=float(choice.value),
                id_prefix="forward-control",
            )
            for index, choice in enumerate(choices)
        )
        rollout = _rollout(
            self.api,
            identity=self.identity,
            mode=self.mode,
            split="train",
            records=records,
        )
        optimizer = self._optimizer()
        original_forward = self.policy.forward
        forward_calls = 0
        interrupt = KeyboardInterrupt("injected later forward interruption")

        def interrupt_second_minibatch(batch):
            nonlocal forward_calls
            forward_calls += 1
            # PPO first validates stored behavior data, then forwards each
            # minibatch. Interrupt the second minibatch after the first step.
            if forward_calls == 3:
                raise interrupt
            return original_forward(batch)

        with (
            patch.object(self.policy, "forward", new=interrupt_second_minibatch),
            patch.object(optimizer, "step", wraps=optimizer.step) as step,
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self._update(self.policy, optimizer, rollout)

        self.assertIs(caught.exception, interrupt)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(self.policy._behavior_policy_version, 1)
        notes = getattr(interrupt, "__notes__", ())
        self.assertTrue(any(
            "phase='forward-loss'" in note
            and "attempted_steps=1" in note
            and "accepted_steps=1" in note
            and "uncertain_steps=0" in note
            and "admitted_behavior_policy_version=0" in note
            and "resulting_behavior_policy_version=1" in note
            and "consumed_decisions=2" in note
            for note in notes
        ))


if __name__ == "__main__":
    unittest.main()
