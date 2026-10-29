"""Pure tests for the projected Task 9 proof orchestration."""

from __future__ import annotations

import importlib
from pathlib import Path
import random
import sys
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


integration = importlib.import_module("tests.integration.test_neural_gpu_runtime")
proof = integration._HELPER
SOURCE_ROOT = str(integration.ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)


class _Driver:
    def reset(self, *, seeds, entry_ids, reset_mask):
        self.resets = getattr(self, "resets", 0) + sum(bool(value) for value in reset_mask)

    def step(self, actions):
        self.steps = getattr(self, "steps", 0) + len(actions)


class _Policy:
    def __init__(self):
        self.rows = 0

    def act(self, frames, *, generator, deterministic=False):
        self.rows += len(frames)
        return tuple(None for _ in frames)


class _Optimizer:
    def step(self):
        return None


def _valid_result(counters):
    return {
        "schema_id": proof.MODE_RESULT_SCHEMA,
        "status": "passed",
        "mode": "standalone",
        "test_id": "projection-test",
        "exit_code": 0,
        "skipped": False,
        "updates": {"accepted": 1, "finite": True, "weights_changed": True},
        "counters": counters,
        "checks": {
            "one_shared_deadline": True,
            "accepted_finite_update_and_changed_weights": True,
            "final_checkpoint_saved": True,
            "same_backend_weights_and_adam": True,
            "adam_scalar_steps_cpu_moments_cuda": True,
            "same_backend_torch_rng_and_python_rng": True,
            "same_backend_counters_version_and_recovery": True,
            "same_frame_deterministic_and_sampled_full_choice_parity": True,
            "portable_cpu_weights_and_tolerant_outputs": True,
            "warm_start_fresh_state_and_frozen_evaluation": True,
            "cpu_comparisons_used_inert_drivers_without_steps": True,
            "training_driver_closed_once": True,
            "actor_critic_and_shared_hashes_recorded": True,
            "zero_fault_uncertainty_and_incomplete_effects": True,
        },
        "weights": {
            "model_sha256": "1" * 64,
            "shared_body_sha256": "2" * 64,
            "actor_head_sha256": "3" * 64,
            "critic_head_sha256": "4" * 64,
        },
        "checkpoint": {
            "path": "standalone/checkpoint-final",
            "metadata_raw_sha256": "5" * 64,
            "metadata_canonical_sha256": "6" * 64,
        },
        "portable_cpu": {
            "warm_start_optimizer_fresh": True,
            "warm_start_rng_fresh": True,
            "evaluation_optimizer_absent": True,
            "cpu_driver_resets": 0,
            "cpu_driver_steps": 0,
        },
    }


def _successful_sparse_ledger():
    ledger = proof.EffectLedger(deadline=10.0, clock=lambda: 1.0)
    proof.GuardedDriver(_Driver(), ledger, "standalone").reset(
        seeds=(1,), entry_ids=("entry",), reset_mask=(True,),
    )
    proof.GuardedDriver(_Driver(), ledger, "standalone").step((0,))
    proof.GuardedPolicy(_Policy(), ledger, "standalone").act(
        (object(),), generator=None, deterministic=False,
    )
    ledger.reserve_diagnostic_pair("standalone")
    proof.GuardedPolicy(_Policy(), ledger, "standalone", diagnostic=True).act(
        (object(), object()), generator=None, deterministic=False,
    )
    proof.GuardedOptimizer(_Optimizer(), ledger, "standalone").step()
    return ledger


class LedgerProjectionTests(unittest.TestCase):
    def test_real_guarded_effects_project_into_the_strict_mode_protocol(self):
        ledger = proof.EffectLedger(deadline=10.0, clock=lambda: 1.0)
        driver = proof.GuardedDriver(_Driver(), ledger, "standalone")
        policy = proof.GuardedPolicy(_Policy(), ledger, "standalone")
        optimizer = proof.GuardedOptimizer(_Optimizer(), ledger, "standalone")

        driver.reset(seeds=(1,), entry_ids=("entry",), reset_mask=(True,))
        driver.step((0,))
        policy.act((object(),), generator=None, deterministic=False)
        ledger.reserve_diagnostic_pair("standalone")
        proof.GuardedPolicy(
            _Policy(), ledger, "standalone", diagnostic=True,
        ).act((object(), object()), generator=None, deterministic=False)
        optimizer.step()

        counters = proof.project_mode_counters(
            ledger, "standalone", faults=0, incomplete_environment_slots=0,
        )

        expected = {
            "steps": 1, "resets": 1, "actor_samples": 1,
            "diagnostic_samples": 2, "optimizer": 1,
            "steps_attempted": 1, "steps_delivered": 1, "steps_uncertain": 0,
            "resets_attempted": 1, "resets_delivered": 1, "resets_uncertain": 0,
            "actor_samples_attempted": 1, "actor_samples_delivered": 1,
            "actor_samples_uncertain": 0,
            "diagnostic_samples_attempted": 2, "diagnostic_samples_delivered": 2,
            "diagnostic_samples_uncertain": 0,
            "optimizer_attempted": 1, "optimizer_accepted": 1,
            "optimizer_uncertain": 0, "faults": 0,
            "incomplete_environment_slots": 0,
        }
        self.assertEqual(counters, expected)

        result = {
            "schema_id": proof.MODE_RESULT_SCHEMA,
            "status": "passed",
            "mode": "standalone",
            "test_id": "projection-test",
            "exit_code": 0,
            "skipped": False,
            "updates": {"accepted": 1, "finite": True, "weights_changed": True},
            "counters": counters,
            "checks": {
                "one_shared_deadline": True,
                "accepted_finite_update_and_changed_weights": True,
                "final_checkpoint_saved": True,
                "same_backend_weights_and_adam": True,
                "adam_scalar_steps_cpu_moments_cuda": True,
                "same_backend_torch_rng_and_python_rng": True,
                "same_backend_counters_version_and_recovery": True,
                "same_frame_deterministic_and_sampled_full_choice_parity": True,
                "portable_cpu_weights_and_tolerant_outputs": True,
                "warm_start_fresh_state_and_frozen_evaluation": True,
                "cpu_comparisons_used_inert_drivers_without_steps": True,
                "training_driver_closed_once": True,
                "actor_critic_and_shared_hashes_recorded": True,
                "zero_fault_uncertainty_and_incomplete_effects": True,
            },
            "weights": {
                "model_sha256": "1" * 64,
                "shared_body_sha256": "2" * 64,
                "actor_head_sha256": "3" * 64,
                "critic_head_sha256": "4" * 64,
            },
            "checkpoint": {
                "path": "standalone/checkpoint-final",
                "metadata_raw_sha256": "5" * 64,
                "metadata_canonical_sha256": "6" * 64,
            },
            "portable_cpu": {
                "warm_start_optimizer_fresh": True,
                "warm_start_rng_fresh": True,
                "evaluation_optimizer_absent": True,
                "cpu_driver_resets": 0,
                "cpu_driver_steps": 0,
            },
        }
        self.assertEqual(
            proof.validate_mode_result(result, mode="standalone", test_id="projection-test"),
            result,
        )

    def test_projection_rejects_unknown_and_invalid_internal_counters(self):
        ledger = _successful_sparse_ledger()
        ledger.modes["standalone"]["unlisted_internal"] = 0
        with self.assertRaisesRegex(ValueError, "unknown counters"):
            proof.project_mode_counters(ledger, "standalone", faults=0, incomplete_environment_slots=0)

        ledger = _successful_sparse_ledger()
        ledger.modes["standalone"]["steps"] = True
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            proof.project_mode_counters(ledger, "standalone", faults=0, incomplete_environment_slots=0)

    def test_projection_reconciles_optimizer_delivery_and_validator_rejects_unresolved(self):
        ledger = _successful_sparse_ledger()
        ledger.modes["standalone"]["optimizer_delivered"] += 1
        with self.assertRaisesRegex(ValueError, "optimizer delivery"):
            proof.project_mode_counters(ledger, "standalone", faults=0, incomplete_environment_slots=0)

        ledger = _successful_sparse_ledger()
        ledger.reserve("standalone", "steps", 1)
        counters = proof.project_mode_counters(
            ledger, "standalone", faults=0, incomplete_environment_slots=0,
        )
        with self.assertRaisesRegex(ValueError, "delivery is unresolved"):
            proof.validate_mode_result(
                _valid_result(counters), mode="standalone", test_id="projection-test",
            )


class EffectGuardContractTests(unittest.TestCase):
    def test_each_released_mode_cap_rejects_before_component_delegation(self):
        caps = {
            "steps": (32, "steps"),
            "resets": (36, "resets"),
            "actor_samples": (32, "actor_samples"),
            "diagnostic_samples": (2, "diagnostic_samples"),
            "optimizer": (4, "optimizer"),
        }
        for name, (maximum, kind) in caps.items():
            with self.subTest(kind=name):
                ledger = proof.EffectLedger(deadline=10.0, clock=lambda: 1.0)
                if kind == "diagnostic_samples":
                    ledger.reserve_diagnostic_pair("standalone")
                    ledger.claim_diagnostic_rows("standalone", maximum)
                    policy = _Policy()
                    guarded = proof.GuardedPolicy(
                        policy, ledger, "standalone", diagnostic=True,
                    )
                    call = lambda: guarded.act((object(),), generator=None)
                    delegated = lambda: policy.rows
                else:
                    ledger.reserve("standalone", kind, maximum)
                    if kind in {"steps", "resets"}:
                        driver = _Driver()
                        guarded = proof.GuardedDriver(driver, ledger, "standalone")
                        if kind == "steps":
                            call = lambda: guarded.step((0,))
                            delegated = lambda: getattr(driver, "steps", 0)
                        else:
                            call = lambda: guarded.reset(
                                seeds=(1,), entry_ids=("entry",), reset_mask=(True,),
                            )
                            delegated = lambda: getattr(driver, "resets", 0)
                    elif kind == "actor_samples":
                        policy = _Policy()
                        guarded = proof.GuardedPolicy(policy, ledger, "standalone")
                        call = lambda: guarded.act((object(),), generator=None)
                        delegated = lambda: policy.rows
                    else:
                        optimizer = _Optimizer()
                        optimizer.calls = 0
                        optimizer.step = lambda: setattr(optimizer, "calls", optimizer.calls + 1)
                        guarded = proof.GuardedOptimizer(optimizer, ledger, "standalone")
                        call = guarded.step
                        delegated = lambda: optimizer.calls
                with self.assertRaises(proof.EffectLimitError):
                    call()
                self.assertEqual(delegated(), 0)

    def test_diagnostic_pair_is_claimed_before_each_policy_draw(self):
        ledger = proof.EffectLedger(deadline=10.0, clock=lambda: 1.0)

        class ObservingPolicy(_Policy):
            def __init__(self):
                super().__init__()
                self.remaining_at_draw = []

            def act(self, frames, *, generator, deterministic=False):
                self.remaining_at_draw.append(ledger.diagnostic_reserved["standalone"])
                return super().act(frames, generator=generator, deterministic=deterministic)

        policy = ObservingPolicy()
        guarded = proof.GuardedPolicy(policy, ledger, "standalone", diagnostic=True)
        ledger.reserve_diagnostic_pair("standalone")
        guarded.act((object(),), generator=None)
        guarded.act((object(),), generator=None)
        self.assertEqual(policy.remaining_at_draw, [1, 0])


class ChoiceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.test_case = integration.NeuralGpuRuntimeProofTests("test_01_standalone_b4_h8")

    def test_full_choice_comparison_rejects_action_probability_value_and_version_changes(self):
        expected = (SimpleNamespace(
            original_action_index=2, log_probability=-0.25, value=0.5,
            behavior_policy_version=3,
        ),)
        for field, value in (
            ("original_action_index", 1),
            ("log_probability", -0.5),
            ("value", 0.75),
            ("behavior_policy_version", 4),
        ):
            actual = (SimpleNamespace(**{
                "original_action_index": 2,
                "log_probability": -0.25,
                "value": 0.5,
                "behavior_policy_version": 3,
                field: value,
            }),)
            with self.subTest(field=field), self.assertRaises(AssertionError):
                integration._assert_exact_choices(self.test_case, expected, actual)

    def test_portable_comparison_keeps_action_exact_and_numeric_outputs_within_one_e_minus_four(self):
        cpu = (SimpleNamespace(
            original_action_index=2, log_probability=-0.25, value=0.5,
        ),)
        within_tolerance = (SimpleNamespace(
            original_action_index=2, log_probability=-0.25009, value=0.50009,
        ),)
        integration._assert_portable_choices(self.test_case, cpu, within_tolerance)
        outside_tolerance = (SimpleNamespace(
            original_action_index=2, log_probability=-0.25011, value=0.5,
        ),)
        with self.assertRaises(AssertionError):
            integration._assert_portable_choices(self.test_case, cpu, outside_tolerance)
        changed_action = (SimpleNamespace(
            original_action_index=3, log_probability=-0.25, value=0.5,
        ),)
        with self.assertRaises(AssertionError):
            integration._assert_portable_choices(self.test_case, cpu, changed_action)

    def test_uncertainty_from_a_real_guarded_failure_is_rejected_by_the_strict_validator(self):
        class FailingStepDriver(_Driver):
            def step(self, actions):
                raise RuntimeError("accepted effect became uncertain")

        ledger = _successful_sparse_ledger()
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            proof.GuardedDriver(FailingStepDriver(), ledger, "standalone").step((0,))
        counters = proof.project_mode_counters(
            ledger, "standalone", faults=0, incomplete_environment_slots=0,
        )
        with self.assertRaisesRegex(ValueError, "uncertain"):
            proof.validate_mode_result(
                _valid_result(counters), mode="standalone", test_id="projection-test",
            )


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _Generator:
    def get_state(self):
        return b"state"


class _Adam:
    def __init__(self):
        self.state = {}
        self.param_groups = []

    def state_dict(self):
        return {"state": self.state, "param_groups": self.param_groups}


class _DriverWithClose:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _CollectorWithProductionRecoveryProperty:
    """Small state carrier using Collector.recovery_state and close methods."""

    from spire_learn.neural.collector import Collector as _Collector

    recovery_state = property(_Collector.recovery_state.fget)
    close = _Collector.close

    def __init__(self):
        self._failed = False
        self._closed = False
        self._phase = "idle"
        self._current_operation_kind = "none"
        self._pending_decisions = 0
        self._decision_ordinals = [3]
        self._seed_ordinal = 4
        self._episodes = [3]
        self.driver = _DriverWithClose()


class TimingBoundaryReproductionTests(unittest.TestCase):
    def test_saved_runtime_prefix_differs_from_live_elapsed_and_closed_recovery(self):
        from spire_learn.neural import runner
        from spire_learn.neural.collector import CollectorRecovery
        from spire_learn.neural.device import RuntimeDevicePlan
        from spire_learn.neural.contracts import NeuralIdentity

        clock = _Clock()

        class Budget:
            def elapsed(self):
                return clock.now - 99.0

        identity = NeuralIdentity(
            model_id="test-model",
            features_id="test-features",
            action_id="test-actions",
            objective_id="test-objective",
            accepted_raw_mode_mappings={"standalone": "test-mode"},
            config_hashes={"model": "a" * 64, "ppo": "b" * 64},
        )
        config = type("Config", (), {})()
        config.operation = "train"
        config.mode = "standalone"
        config.accepted_identities = (identity,)
        config.config_sha256 = "c" * 64
        config.manifest_sha256 = "d" * 64
        config.environment_revision = "fixture-gym-revision"
        collector = _CollectorWithProductionRecoveryProperty()
        optimizer = runner._CountingOptimizer(_Adam())
        policy = type("Policy", (), {"_behavior_policy_version": 0})()
        runtime = runner._NeuralTrainingRuntime(
            config=config,
            collector=collector,
            policy=policy,
            optimizer=optimizer,
            torch=object(),
            policy_generator=_Generator(),
            shuffle_generator=_Generator(),
            curriculum_rng=random.Random(1),
            controller_rng=random.Random(2),
            device_plan=RuntimeDevicePlan("cpu", "cpu", "cpu", "cpu", None),
            device_report={},
            budget=Budget(),
            recovery=CollectorRecovery(4, 3, (3,)),
        )

        writer_records = []

        def recording_writer(path, *, policy, optimizer, rng_state, metadata):
            del path, policy, optimizer, rng_state
            writer_records.append({
                "training_counters": dict(metadata["training_state"]["counters"]),
                "run_prefix": dict(metadata["run_metadata"]["cumulative_prefix"]),
                "recovery": collector.recovery_state,
            })

        with patch.object(runner, "save_checkpoint", recording_writer):
            runtime.save(Path("unused-checkpoint"), "final")

        saved = writer_records[0]
        self.assertEqual(saved["training_counters"], saved["run_prefix"])
        self.assertEqual(saved["recovery"], CollectorRecovery(4, 3, (4,)))

        # Simulated serialization and close consume time after the snapshot.
        clock.now += 0.25
        runtime.close()
        live = runtime._latest_counters()
        self.assertEqual(
            {name: value for name, value in live.items() if name != "elapsed_seconds"},
            {name: value for name, value in saved["training_counters"].items()
             if name != "elapsed_seconds"},
        )
        self.assertGreater(live["elapsed_seconds"], saved["training_counters"]["elapsed_seconds"])
        with self.assertRaises(AssertionError):
            self.assertEqual(live, saved["training_counters"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = collector.recovery_state
        self.assertEqual(collector.driver.close_calls, 1)


class ProjectedOrchestrationTests(unittest.TestCase):
    def _harness(self, root, *, mode_clock=True):
        from tests.unit.neural_gpu_proof_fakes import ProjectedHarness

        return ProjectedHarness(root, integration, mode_clock=mode_clock)

    def test_full_actual_run_mode_projects_both_modes_and_reconciles_host_evidence(self):
        output_parent = integration.ROOT
        with tempfile.TemporaryDirectory(
            prefix="neural-gpu-orchestration-", dir=output_parent,
        ) as temporary:
            harness = self._harness(Path(temporary))
            test = harness.setup_case("standalone")
            with harness.activate():
                standalone = test._run_mode(
                    "standalone",
                    "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                    (4, 8),
                )
                run = test._run_mode(
                    "run",
                    "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_02_run_b1_h32",
                    (1, 32),
                )

            self.assertEqual(standalone["counters"]["steps_attempted"], 32)
            self.assertEqual(standalone["counters"]["resets_attempted"], 36)
            self.assertEqual(standalone["counters"]["optimizer_attempted"], 4)
            self.assertEqual(run["counters"]["steps_attempted"], 32)
            self.assertEqual(run["counters"]["resets_attempted"], 33)
            self.assertEqual(run["counters"]["optimizer_attempted"], 4)
            self.assertEqual(len(harness.writer_calls), 2)
            self.assertEqual(len(harness.loaded_checkpoints), 2)
            loaded_by_path = {
                str(path): loaded for path, loaded in harness.loaded_checkpoints
            }
            from spire_learn.neural.checkpoint import LoadedCheckpoint

            for mode, result in (("standalone", standalone), ("run", run)):
                checkpoint_path = integration.ROOT / result["checkpoint"]["path"]
                loaded = loaded_by_path[str(checkpoint_path)]
                self.assertIs(type(loaded), LoadedCheckpoint)
                self.assertIsNotNone(loaded.optimizer_state)
                self.assertIsNotNone(loaded.rng_state)
                self.assertEqual(loaded.optimizer_step, loaded.metadata["training_state"]["optimizer_step"])
                self.assertEqual(loaded.inflight_record_count, 0)
                self.assertEqual(loaded.metadata["recovery"]["purpose"], "resume")
                frozen_recovery = loaded.metadata["run_metadata"]["collector_recovery"]
                self.assertIs(type(frozen_recovery), MappingProxyType)
                frozen_episodes = frozen_recovery["next_episode_ordinals"]
                self.assertIs(type(frozen_episodes), tuple)
                expected_recovery = {
                    "next_seed_ordinal": frozen_recovery["next_seed_ordinal"],
                    "next_decision_sequence": frozen_recovery["next_decision_sequence"],
                    "next_episode_ordinals": list(frozen_episodes),
                }

                result_recovery = result["checkpoint"]["collector_recovery"]
                self.assertIs(type(result_recovery), dict)
                self.assertIs(type(result_recovery["next_episode_ordinals"]), list)
                self.assertEqual(result_recovery, expected_recovery)
                published = proof.load_json(
                    Path(temporary) / mode / "gpu-proof-checks.json",
                )
                published_recovery = published["checkpoint"]["collector_recovery"]
                self.assertIs(type(published_recovery), dict)
                self.assertIs(type(published_recovery["next_episode_ordinals"]), list)
                self.assertEqual(published_recovery, expected_recovery)

                cached_recovery = harness.checkpoints[str(checkpoint_path)]["metadata"][
                    "run_metadata"
                ]["collector_recovery"]
                cached_before = {
                    "next_seed_ordinal": cached_recovery["next_seed_ordinal"],
                    "next_decision_sequence": cached_recovery["next_decision_sequence"],
                    "next_episode_ordinals": list(cached_recovery["next_episode_ordinals"]),
                }
                with self.assertRaises(TypeError):
                    frozen_recovery["next_seed_ordinal"] = -1
                with self.assertRaises(TypeError):
                    frozen_episodes[0] = -1
                self.assertEqual(cached_recovery, cached_before)

            self.assertEqual(
                proof.verify_mode_evidence(Path(temporary))["cumulative_counters"],
                {
                    "steps_attempted": 64, "steps_delivered": 64, "steps_uncertain": 0,
                    "resets_attempted": 69, "resets_delivered": 69, "resets_uncertain": 0,
                    "actor_samples_attempted": 64, "actor_samples_delivered": 64,
                    "actor_samples_uncertain": 0,
                    "diagnostic_samples_attempted": 4,
                    "diagnostic_samples_delivered": 4,
                    "diagnostic_samples_uncertain": 0,
                    "optimizer_attempted": 8, "optimizer_accepted": 8,
                    "optimizer_uncertain": 0,
                },
            )

    def test_closed_collector_recovery_is_not_read_after_runtime_cleanup(self):
        output_parent = integration.ROOT
        with tempfile.TemporaryDirectory(
            prefix="neural-gpu-closed-recovery-", dir=output_parent,
        ) as temporary:
            harness = self._harness(Path(temporary), mode_clock=False)
            test = harness.setup_case("standalone")
            with harness.activate():
                result = test._run_mode(
                    "standalone",
                    "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                    (4, 8),
                )
            self.assertEqual(result["status"], "passed")
            self.assertTrue(harness.runtimes[0]._closed)

    def test_earlier_shared_deadline_rejects_before_runtime_factory(self):
        with tempfile.TemporaryDirectory(
            prefix="neural-gpu-deadline-",
            dir=integration.ROOT,
        ) as temporary:
            harness = self._harness(Path(temporary))
            test = harness.setup_case("standalone")
            test.shared_deadline = harness.clock.now
            test.ledger.deadline = harness.clock.now
            with harness.activate(), self.assertRaises(proof.ProofDeadlineExceeded):
                test._run_mode(
                    "standalone",
                    "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                    (4, 8),
                )
            self.assertEqual(harness.create_calls, [])
            self.assertEqual(harness.drivers, [])

    def test_corrupt_saved_prefix_or_recovery_is_rejected_before_resume_creation(self):
        for corruption in ("corrupt_prefix", "corrupt_recovery"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory(
                prefix="neural-gpu-corrupt-snapshot-",
                dir=integration.ROOT,
            ) as temporary:
                harness = self._harness(Path(temporary), mode_clock=False)
                setattr(harness, corruption, True)
                test = harness.setup_case("standalone")
                with harness.activate(), self.assertRaises(AssertionError):
                    test._run_mode(
                        "standalone",
                        "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                        (4, 8),
                    )
                self.assertEqual(harness.create_calls, [("standalone", "train")])
                self.assertTrue(harness.runtimes[0]._closed)
                self.assertEqual(harness.drivers[0].close_calls, 1)
                self.assertEqual(test.ledger.mode_snapshot("standalone")["optimizer_attempted"], 4)

    def test_restored_recovery_or_policy_comparison_mismatches_close_inert_runtime(self):
        for mismatch in (
            "corrupt_restored_recovery", "mismatch_choice", "mismatch_rng",
            "mismatch_device", "mismatch_version",
        ):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory(
                prefix="neural-gpu-resume-mismatch-",
                dir=integration.ROOT,
            ) as temporary:
                harness = self._harness(Path(temporary), mode_clock=False)
                setattr(harness, mismatch, True)
                test = harness.setup_case("standalone")
                with harness.activate(), self.assertRaises(AssertionError):
                    test._run_mode(
                        "standalone",
                        "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                        (4, 8),
                    )
                self.assertEqual(len(harness.runtimes), 2)
                self.assertTrue(all(runtime._closed for runtime in harness.runtimes))
                inert_driver = harness.runtimes[1].collector.driver
                self.assertEqual(inert_driver.reset_calls, 0)
                self.assertEqual(inert_driver.step_calls, 0)
                self.assertEqual(inert_driver.close_calls, 1)
                self.assertEqual(test.ledger.mode_snapshot("standalone")["optimizer_attempted"], 4)

    def test_setup_and_save_failures_preserve_primary_and_close_once(self):
        with tempfile.TemporaryDirectory(
            prefix="neural-gpu-setup-cleanup-",
            dir=integration.ROOT,
        ) as temporary:
            harness = self._harness(Path(temporary), mode_clock=False)
            harness.close_error = RuntimeError("secondary close fault")
            test = harness.setup_case("standalone")

            def setup_assertion_failed(actual, expected):
                raise AssertionError("injected post-acquisition setup fault")

            test.assertIs = setup_assertion_failed
            with harness.activate(), self.assertRaisesRegex(
                AssertionError, "injected post-acquisition setup fault",
            ) as caught:
                test._run_mode(
                    "standalone",
                    "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                    (4, 8),
                )
            self.assertEqual(harness.train_calls, 0)
            self.assertEqual(harness.drivers[0].close_calls, 1)
            self.assertIn("secondary close fault", "\n".join(caught.exception.__notes__))

        with tempfile.TemporaryDirectory(
            prefix="neural-gpu-save-cleanup-",
            dir=integration.ROOT,
        ) as temporary:
            harness = self._harness(Path(temporary), mode_clock=False)
            harness.save_error = OSError("checkpoint writer failed")
            harness.close_error = RuntimeError("secondary runtime close fault")
            test = harness.setup_case("standalone")
            with harness.activate(), self.assertRaisesRegex(OSError, "checkpoint writer failed") as caught:
                test._run_mode(
                    "standalone",
                    "test_neural_gpu_runtime.NeuralGpuRuntimeProofTests.test_01_standalone_b4_h8",
                    (4, 8),
                )
            self.assertTrue(harness.runtimes[0]._closed)
            self.assertEqual(harness.drivers[0].close_calls, 1)
            self.assertEqual(len(harness.writer_calls), 1)
            self.assertIn("secondary runtime close fault", "\n".join(caught.exception.__notes__))

    def test_close_process_control_promotes_over_operational_failure_and_closes_all(self):
        class Resource:
            def __init__(self, error=None):
                self.error = error
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.error is not None:
                    raise self.error

        ordinary_primary = AssertionError("setup failed")
        single_control = Resource(KeyboardInterrupt("single control close"))
        with self.assertRaisesRegex(KeyboardInterrupt, "single control close") as single_caught:
            integration._close_owned_runtime(single_control, ordinary_primary)
        self.assertEqual(single_control.close_calls, 1)
        self.assertIn("setup failed", "\n".join(single_caught.exception.__notes__))

        ordinary_close = Resource(RuntimeError("first close failed"))
        first_control = Resource(KeyboardInterrupt("first control close"))
        later_control = Resource(SystemExit("later control close"))
        with self.assertRaisesRegex(KeyboardInterrupt, "first control close") as caught:
            integration._close_owned_runtimes(
                (ordinary_close, first_control, later_control), ordinary_primary,
            )
        self.assertEqual(
            (ordinary_close.close_calls, first_control.close_calls, later_control.close_calls),
            (1, 1, 1),
        )
        notes = "\n".join(caught.exception.__notes__)
        self.assertIn("setup failed", notes)
        self.assertIn("later control close", notes)

    def test_runner_train_admission_charges_saved_prefix_once_and_applies_shared_deadline(self):
        from spire_learn.neural import runner
        from spire_learn.neural.collector import CollectorRecovery
        from spire_learn.neural.config import load_neural_config

        base_config = load_neural_config(
            integration.ROOT / "configs/neural/standalone.toml",
        )
        now = 20.0

        class AdmissionRuntime:
            def __init__(self, config, deadline):
                self.recovered_progress = {"elapsed_seconds": 17.25}
                self.operation_limits = runner.RuntimeOperationLimits(
                    monotonic_deadline=deadline,
                )
                self.collector = SimpleNamespace(
                    recovery_state=CollectorRecovery(0, 0, (0,) * config.rollout_rows),
                )
                self.close_calls = 0

            def set_budget(self, budget):
                self.budget = budget

            def close(self):
                self.close_calls += 1

        for deadline, expected in ((now + 200.0, 102.75), (now + 5.5, 5.5)):
            with self.subTest(deadline=deadline):
                runtime = AdmissionRuntime(base_config, deadline)
                observed = {}

                def stop_at_writer_start(config, acquired_runtime, **kwargs):
                    del config, kwargs
                    observed["allowed_seconds"] = acquired_runtime.budget.allowed_seconds
                    raise OSError("stop at injected writer-start boundary")

                with patch.object(runner, "_writer_start", side_effect=stop_at_writer_start):
                    report = runner.train(base_config, runtime=runtime, clock=lambda: now)
                self.assertEqual(observed["allowed_seconds"], expected)
                self.assertEqual(report.exit_code, 1)
                self.assertEqual(runtime.close_calls, 1)

    def test_inert_factory_postconstruction_assertion_closes_owned_driver_once(self):
        from types import SimpleNamespace
        from spire_learn.neural import envs

        with tempfile.TemporaryDirectory(
            prefix="neural-gpu-inert-factory-",
            dir=integration.ROOT,
        ) as temporary:
            harness = self._harness(Path(temporary), mode_clock=False)
            test = harness.setup_case("standalone")
            test._trained_driver = SimpleNamespace(schema=object())
            acquired = {}

            class OwnedRuntime:
                def __init__(self, driver):
                    self._closed = False
                    self.close_calls = 0
                    self.collector = SimpleNamespace(driver=driver)

                def close(self):
                    self.close_calls += 1
                    self._closed = True
                    self.collector.driver.close()
                    raise OSError("inert close fault")

            def create_runtime(config, *, purpose, checkpoint, **kwargs):
                del purpose, checkpoint, kwargs
                acquired_driver = envs.make_driver(config, config.cases)
                runtime = OwnedRuntime(acquired_driver)
                acquired["runtime"] = runtime
                acquired["driver"] = acquired_driver
                return runtime

            test.create_runtime = create_runtime

            def postconstruction_assertion_failed(*args, **kwargs):
                raise AssertionError("inert factory count mismatch")

            test.assertEqual = postconstruction_assertion_failed
            with harness.activate(), self.assertRaisesRegex(
                AssertionError, "inert factory count mismatch",
            ) as caught:
                test._create_inert_runtime(
                    test.configs["standalone"], purpose="resume", checkpoint=Path("unused"),
                )
            self.assertEqual(acquired["runtime"].close_calls, 1)
            self.assertEqual(acquired["driver"].reset_calls, 0)
            self.assertEqual(acquired["driver"].step_calls, 0)
            self.assertEqual(acquired["driver"].close_calls, 1)
            self.assertIn("inert close fault", "\n".join(caught.exception.__notes__))

    def test_publication_failure_halts_run_before_another_runtime_factory(self):
        test_case_type = integration.NeuralGpuRuntimeProofTests
        test_case_type._first_mode_failure = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="neural-gpu-publication-halt-",
                dir=integration.ROOT,
            ) as temporary:
                harness = self._harness(Path(temporary), mode_clock=False)
                test = harness.setup_case("standalone")
                with harness.activate(), patch.object(
                    test, "_write_mode_evidence", side_effect=OSError("publication failed"),
                ), self.assertRaisesRegex(OSError, "publication failed"):
                    test.test_01_standalone_b4_h8()
                created_before_run = tuple(harness.create_calls)
                with self.assertRaisesRegex(RuntimeError, "halted without another workload"):
                    test.test_02_run_b1_h32()
                self.assertEqual(tuple(harness.create_calls), created_before_run)
                self.assertEqual(harness.drivers[0].close_calls, 1)
        finally:
            test_case_type._first_mode_failure = None


if __name__ == "__main__":
    unittest.main()
