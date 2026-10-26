"""One gated, bounded CPU training smoke for each accepted neural mode."""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from spire_learn.neural.config import load_neural_config


ROOT = Path(__file__).resolve().parents[2]
_RUN_SMOKE = (
    os.environ.get("SPIRE_LEARN_RUN_NEURAL_NATIVE") == "1"
    and os.environ.get("SPIRE_LEARN_RUN_NEURAL_TENSOR") == "1"
)


def _assert_choice_parity(actual_choices, expected_choices) -> None:
    """Check deterministic action, value, and log probability for every choice."""
    if len(actual_choices) != len(expected_choices):
        raise AssertionError(
            f"choice count differs: {len(actual_choices)} != {len(expected_choices)}"
        )
    for index, (actual_choice, expected_choice) in enumerate(
        zip(actual_choices, expected_choices, strict=True)
    ):
        if actual_choice.original_action_index != expected_choice.original_action_index:
            raise AssertionError(f"choice {index} action differs")
        if not math.isclose(actual_choice.value, expected_choice.value, rel_tol=0.0, abs_tol=1e-7):
            raise AssertionError(f"choice {index} value differs")
        if not math.isclose(actual_choice.log_probability, expected_choice.log_probability,
                            rel_tol=0.0, abs_tol=1e-7):
            raise AssertionError(f"choice {index} log_probability differs")


@unittest.skipUnless(
    _RUN_SMOKE,
    "Task 7 CPU training smoke is held for parent and fresh source/control review",
)
class NeuralTrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.monotonic()
        cls.shared_deadline = cls.started + 120.0
        import torch

        from scripts.neural.task7_training_run_helpers import record_training_process_identity

        cls.torch = torch
        cls.training_process_identity = record_training_process_identity(
            torch, ROOT / "artifacts/development/neural-combat/runtime-identity.json",
        )
        sys.stderr.write(
            "TASK7_TRAINING_PROCESS_IDENTITY="
            + json.dumps(cls.training_process_identity, sort_keys=True) + "\n"
        )
        sys.stderr.flush()

    def _run_smoke(self, mode: str) -> None:
        import torch

        from spire_learn.neural.checkpoint import load_checkpoint
        from spire_learn.neural.model import build_policy
        from spire_learn.neural.runner import RuntimeOperationLimits, create_runtime, train

        config = load_neural_config(ROOT / f"configs/neural/{mode}.toml")
        self.assertEqual(config.mode, mode)
        self.assertEqual(config.rollout_rows, 4 if mode == "standalone" else 1)
        self.assertEqual(config.rollout_steps, 32 if mode == "standalone" else 128)
        self.assertEqual(config.budgets.max_training_decisions, 256)
        self.assertEqual(config.budgets.max_environment_actions, 1024)
        self.assertEqual(config.budgets.max_optimizer_steps, 16)
        self.assertLessEqual(config.rollout_rows * config.rollout_steps, 128)

        runtime = create_runtime(config, purpose="train")
        runtime.operation_limits = RuntimeOperationLimits(
            max_updates=2, monotonic_deadline=self.shared_deadline,
        )
        initial_parameters = {
            name: value.detach().cpu().clone()
            for name, value in runtime.policy.named_parameters()
        }
        public_frames = []
        collect = runtime.collector.collect

        def collect_and_retain_public_frames(**kwargs):
            rollout = collect(**kwargs)
            public_frames.extend(
                record.source for record in rollout.records if record.actor_training_mask
            )
            return rollout

        runtime.collector.collect = collect_and_retain_public_frames
        report = train(config, runtime=runtime)
        self.assertEqual(report.exit_code, 0, report.summary)
        self.assertEqual(report.summary["optimizer"]["updates"], 2, report.summary)
        self.assertLessEqual(report.summary["collection"]["learned_decisions"], 256)
        self.assertLessEqual(report.summary["collection"]["attempted_environment_slots"], 1024)
        self.assertLessEqual(report.summary["optimizer"]["attempted_steps"], 16)
        self.assertGreater(report.summary["optimizer"]["accepted_steps"], 0)
        self.assertEqual(report.summary["optimizer"]["uncertain_steps"], 0)
        self.assertEqual(report.summary["optimizer"]["attempted_steps"],
                         report.summary["optimizer"]["accepted_steps"])
        self.assertLessEqual(time.monotonic(), self.shared_deadline)
        self.assertTrue(public_frames, "the smoke must retain accepted public frames for reload parity")
        self.assertIsNotNone(report.run_dir)

        final_parameters = {
            name: value.detach().cpu().clone()
            for name, value in runtime.policy.named_parameters()
        }
        self.assertEqual(set(initial_parameters), set(final_parameters))
        self.assertTrue(all(torch.isfinite(value).all().item() for value in final_parameters.values()))
        changed_parameters = [
            name for name in final_parameters
            if not torch.equal(initial_parameters[name], final_parameters[name])
        ]
        self.assertTrue(changed_parameters, "at least one finite model parameter must change")

        checkpoint = report.run_dir / "checkpoint-final"
        loaded = load_checkpoint(
            checkpoint, config.accepted_identities[0], purpose="evaluate", device="cpu"
        )
        restored = build_policy(
            runtime.collector.driver.schema, config.model,
            seed=config.seed_streams.policy, device="cpu",
        )
        restored.load_state_dict(dict(loaded.model_tensors), strict=True)
        frames = tuple(public_frames[: min(8, len(public_frames))])
        first_generator = torch.Generator(device="cpu").manual_seed(31337)
        second_generator = torch.Generator(device="cpu").manual_seed(31337)
        expected = runtime.policy.act(frames, generator=first_generator, deterministic=True)
        actual = restored.act(frames, generator=second_generator, deterministic=True)
        _assert_choice_parity(actual, expected)

        from spire_learn.neural.collector import CollectorRecovery

        saved = load_checkpoint(
            checkpoint, config.accepted_identities[0], purpose="resume", device="cpu"
        )
        saved_recovery = saved.metadata["run_metadata"]["collector_recovery"]
        recovery = CollectorRecovery(
            saved_recovery["next_seed_ordinal"], saved_recovery["next_decision_sequence"],
            tuple(saved_recovery["next_episode_ordinals"]),
        )

        class FreshResumeDriver:
            def __init__(self):
                self.mode = mode
                self.batch_size = config.rollout_rows
                self.rows = (None,) * config.rollout_rows
                self.accepted_steps = 0
                self.schema = runtime.collector.driver.schema
                self.reset_calls = []
                self.step_calls = []
                self.close_calls = 0

            def reset(self, *args, **kwargs):
                self.reset_calls.append((args, kwargs))
                raise AssertionError("resume comparison must not reset a native row")

            def step(self, *args, **kwargs):
                self.step_calls.append((args, kwargs))
                raise AssertionError("resume comparison must not step a native row")

            def close(self):
                self.close_calls += 1

        fake_driver = FreshResumeDriver()
        with patch("spire_learn.neural.envs.make_driver", return_value=fake_driver):
            resumed = create_runtime(config, purpose="resume", checkpoint=checkpoint)
        try:
            def trees_equal(left, right):
                if torch.is_tensor(left):
                    return torch.is_tensor(right) and torch.equal(left, right)
                if isinstance(left, dict):
                    return isinstance(right, dict) and left.keys() == right.keys() and all(
                        trees_equal(left[key], right[key]) for key in left
                    )
                if isinstance(left, (tuple, list)):
                    return isinstance(right, type(left)) and len(left) == len(right) and all(
                        trees_equal(a, b) for a, b in zip(left, right, strict=True)
                    )
                return left == right

            self.assertTrue(trees_equal(runtime.policy.state_dict(), resumed.policy.state_dict()))
            self.assertTrue(trees_equal(runtime.optimizer.state_dict(), resumed.optimizer.state_dict()))
            self.assertTrue(torch.equal(runtime.policy_generator.get_state(),
                                        resumed.policy_generator.get_state()))
            self.assertTrue(torch.equal(runtime.shuffle_generator.get_state(),
                                        resumed.shuffle_generator.get_state()))
            self.assertEqual(runtime.curriculum_rng.getstate(), resumed.curriculum_rng.getstate())
            self.assertEqual(runtime.controller_rng.getstate(), resumed.controller_rng.getstate())
            self.assertEqual(resumed.recovered_progress, saved.metadata["training_state"]["counters"])
            self.assertEqual(resumed.update_count, saved.metadata["training_state"]["counters"]["update_count"])
            self.assertEqual(resumed.policy._behavior_policy_version,
                             saved.metadata["training_state"]["behavior_policy_version"])
            self.assertEqual(resumed.collector.recovery_state, recovery)
            self.assertEqual(fake_driver.reset_calls, [])
            self.assertEqual(fake_driver.step_calls, [])
            self.assertEqual(fake_driver.accepted_steps, 0)
        finally:
            resumed.close()
        self.assertEqual(fake_driver.close_calls, 1)

        smoke_checks = {
            "schema_id": "spire-learn-task7-smoke-checks-v1",
            "all_final_weights_finite": True,
            "at_least_one_weight_changed": True,
            "changed_parameter_count": len(changed_parameters),
            "deterministic_reload_parity": True,
            "resume_round_trip": True,
            "resume_driver_reset_calls": 0,
            "resume_driver_step_calls": 0,
            "accepted_optimizer_steps": report.summary["optimizer"]["accepted_steps"],
            "uncertain_optimizer_steps": report.summary["optimizer"]["uncertain_steps"],
        }
        with (report.run_dir / "smoke-checks.json").open("x", encoding="utf-8") as stream:
            json.dump(smoke_checks, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()

    def test_run_bounded_training_and_reload_parity(self):
        self._run_smoke("run")
        self.assertLessEqual(time.monotonic() - self.started, 300.0)

    def test_standalone_bounded_training_and_reload_parity(self):
        self._run_smoke("standalone")
        self.assertLessEqual(time.monotonic() - self.started, 300.0)


if __name__ == "__main__":
    unittest.main()
