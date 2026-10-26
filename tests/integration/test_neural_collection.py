"""Two bounded, opt-in native collection checks; import-free while disabled."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import time
import unittest

from scripts.neural.task5_collection_control import (
    PUBLIC_DIAGNOSTIC_ACTION_ID,
    PUBLIC_DIAGNOSTIC_KIND,
    PUBLIC_DIAGNOSTIC_MODEL_ID,
    PUBLIC_DIAGNOSTIC_SPEC,
    PUBLIC_DIAGNOSTIC_SPEC_SHA256,
    PUBLIC_DIAGNOSTIC_VERSION,
    SHARED_COHORT_BUDGET_ID,
    SHARED_COHORT_BUDGET_SECONDS,
)


_ENABLED = os.environ.get("SPIRE_LEARN_RUN_NEURAL_COLLECTION") == "1"
_EVIDENCE_PATH = os.environ.get("SPIRE_LEARN_COLLECTION_EVIDENCE")
_MAX_ACTION_SLOTS_PER_MODE = 2048
_RUN_COUNTER_FIELDS = (
    "attempted_environment_slots", "forwarded_environment_slots",
    "known_accepted_environment_slots", "delivered_environment_slots",
    "uncertain_environment_slots", "attempted_reset_slots", "forwarded_reset_slots",
    "known_accepted_reset_slots", "delivered_reset_slots", "uncertain_reset_slots",
    "reset_operations", "reset_slots", "next_seed_ordinal", "tasks_completed",
    "task_terminals",
)
_RESET_COUNTER_FIELDS = (
    "attempted_reset_slots", "forwarded_reset_slots", "known_accepted_reset_slots",
    "delivered_reset_slots", "uncertain_reset_slots", "reset_operations", "reset_slots",
    "next_seed_ordinal",
)


class PublicLowestLegalDiagnosticPolicy:
    """Frames-only public action-mask diagnostic policy, explicitly not a learner."""

    model_id = PUBLIC_DIAGNOSTIC_MODEL_ID
    features_id = "spire-learn-combat-features-v1"
    action_id = PUBLIC_DIAGNOSTIC_ACTION_ID
    objective_id = "spire-learn-combat-winloss-v1"

    def __init__(self, identity):
        self.identity = identity

    def act(self, frames, *, generator, deterministic=False):
        from spire_learn.neural.contracts import PolicyChoice

        if type(deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")
        del generator  # The point-mass diagnostic policy has no random draw.
        choices = []
        for frame in frames:
            mask = frame.tensors.get("action_mask")
            if (mask is None or mask.dtype != "int8" or len(mask.shape) != 1
                    or len(mask.data) != mask.shape[0]
                    or any(value not in (0, 1) for value in mask.data)):
                raise ValueError("diagnostic policy requires the public int8 action_mask")
            legal = [index for index, value in enumerate(mask.data) if value == 1]
            if not legal:
                raise ValueError("diagnostic policy received no public legal action")
            choices.append(PolicyChoice(min(legal), 0.0, 0.0, PUBLIC_DIAGNOSTIC_VERSION))
        return tuple(choices)

    def values(self, frames):
        return tuple(0.0 for _frame in frames)


def make_public_diagnostic_config(config):
    """Give the diagnostic policy a distinct, hash-bound configured identity."""
    from spire_learn.neural.contracts import NeuralIdentity

    base = config.accepted_identities[0]
    identity = NeuralIdentity(
        model_id=PUBLIC_DIAGNOSTIC_MODEL_ID,
        features_id="spire-learn-combat-features-v1",
        action_id=PUBLIC_DIAGNOSTIC_ACTION_ID,
        objective_id="spire-learn-combat-winloss-v1",
        accepted_raw_mode_mappings=base.accepted_raw_mode_mappings,
        config_hashes={"policy_spec": PUBLIC_DIAGNOSTIC_SPEC_SHA256},
    )
    return replace(config, rollout_steps=1, accepted_identities=(identity,)), identity


def _policy_metadata(identity) -> dict[str, object]:
    return {
        "kind": PUBLIC_DIAGNOSTIC_KIND,
        "behavior_policy_version": PUBLIC_DIAGNOSTIC_VERSION,
        "distribution": "point-mass-on-minimum-legal-action-mask-index",
        "value_estimator": "constant-zero-v1",
        "policy_spec": PUBLIC_DIAGNOSTIC_SPEC,
        "identity": {
            "model_id": identity.model_id,
            "features_id": identity.features_id,
            "action_id": identity.action_id,
            "objective_id": identity.objective_id,
            "policy_spec_sha256": identity.config_hashes["policy_spec"],
        },
    }


class NeuralCollectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _ENABLED:
            raise unittest.SkipTest("set SPIRE_LEARN_RUN_NEURAL_COLLECTION=1 in the reviewed CPU runtime")
        if not _EVIDENCE_PATH:
            raise RuntimeError("SPIRE_LEARN_COLLECTION_EVIDENCE must point into the fresh writable run child")

        from spire_learn.experiments.budget import Budget

        cls.shared_budget = Budget(SHARED_COHORT_BUDGET_SECONDS, time.monotonic)
        cls.shared_budget_id = SHARED_COHORT_BUDGET_ID

        from spire_learn.neural.collector import Collector
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.envs import make_driver
        from spire_learn.neural.model import build_policy

        cls.Collector = Collector
        cls.load_neural_config = staticmethod(load_neural_config)
        cls.make_driver = staticmethod(make_driver)
        cls.build_policy = staticmethod(build_policy)
        cls.root = Path(__file__).resolve().parents[2]

    @staticmethod
    def _frame_hash(frame) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(frame.tensors.items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.dtype.encode("ascii"))
            digest.update(repr(value.shape).encode("ascii"))
            digest.update(value.data)
        return digest.hexdigest()

    @staticmethod
    def _counter_snapshot(collection) -> dict[str, int]:
        return {name: getattr(collection, name) for name in _RUN_COUNTER_FIELDS}

    @staticmethod
    def _counter_before_action(previous, current) -> dict[str, int]:
        before = ({name: 0 for name in _RUN_COUNTER_FIELDS}
                  if previous is None else NeuralCollectionIntegrationTests._counter_snapshot(previous))
        after = NeuralCollectionIntegrationTests._counter_snapshot(current)
        for name in _RESET_COUNTER_FIELDS:
            before[name] = after[name]
        return before

    def _record(
        self, mode: str, config, rollout, *, run_continuation=None,
        manual_reset_boundary=None,
    ) -> None:
        collection = rollout.collection
        assert collection is not None
        record = {
            "test_id": (
                "tests.integration.test_neural_collection.NeuralCollectionIntegrationTests."
                + ("test_collect_standalone_manual_resets" if mode == "standalone"
                   else "test_collect_run_natural_task_end_without_run_reset")
            ),
            "mode": mode,
            "environment_revision": config.environment_revision,
            "common_view_fingerprint": config.common_view_fingerprint,
            "run_profile_id": config.profile_id,
            "run_observation_fingerprint": config.run_observation_fingerprint,
            "attempted_environment_slots": collection.attempted_environment_slots,
            "forwarded_environment_slots": collection.forwarded_environment_slots,
            "known_accepted_environment_slots": collection.known_accepted_environment_slots,
            "delivered_environment_slots": collection.delivered_environment_slots,
            "uncertain_environment_slots": collection.uncertain_environment_slots,
            "reset_operations": collection.reset_operations,
            "reset_slots": collection.reset_slots,
            "attempted_reset_slots": collection.attempted_reset_slots,
            "forwarded_reset_slots": collection.forwarded_reset_slots,
            "known_accepted_reset_slots": collection.known_accepted_reset_slots,
            "delivered_reset_slots": collection.delivered_reset_slots,
            "uncertain_reset_slots": collection.uncertain_reset_slots,
            "learned_decisions": collection.learned_decisions,
            "fixed_decisions": collection.fixed_decisions,
            "tasks_started": collection.tasks_started,
            "task_terminals": collection.task_terminals,
            "tasks_completed": collection.tasks_completed,
            "victories": collection.victories,
            "deaths": collection.deaths,
            "faults": collection.faults,
            "elapsed_seconds": collection.elapsed_seconds,
            "wall_overshoot_seconds": collection.wall_overshoot_seconds,
            "elapsed_seconds_is_fallback": collection.elapsed_seconds_is_fallback,
            "incomplete_decisions": collection.incomplete_decisions,
            "incomplete_environment_slots": collection.incomplete_environment_slots,
            "undispatched_actor_decisions": collection.undispatched_actor_decisions,
            "undispatched_fixed_decisions": collection.undispatched_fixed_decisions,
            "next_seed_ordinal": collection.next_seed_ordinal,
            "stop_reason": collection.stop_reason,
            "optimizer_updates": 0,
            "shared_budget_id": self.shared_budget_id,
            "wall_budget_limit_seconds": SHARED_COHORT_BUDGET_SECONDS,
            "run_continuation": run_continuation,
            "manual_reset_boundary": manual_reset_boundary,
            "seed_allocations": [
                {
                    "stream_id": seed.stream_id,
                    "slot": seed.slot,
                    "case_id": seed.case_id,
                    "entry_id": seed.entry_id,
                    "authored_reference_seed": seed.authored_reference_seed,
                    "actual_seed": seed.actual_seed,
                    "ordinal": seed.ordinal,
                    "allocation_policy_id": seed.allocation_policy_id,
                    "delivery": seed.delivery,
                }
                for seed in collection.seed_allocations
            ],
            "resource_checkpoints": [
                {
                    "stream_id": checkpoint.stream_id,
                    "task_id": checkpoint.task_id,
                    "visit": checkpoint.visit,
                    "outcome": checkpoint.outcome,
                    "encounter": checkpoint.encounter,
                    "hp_before_healing": checkpoint.hp_before_healing,
                    "hp_after_cleanup": checkpoint.hp_after_cleanup,
                    "public_frame_sha256": self._frame_hash(checkpoint.frame),
                }
                for checkpoint in collection.resource_checkpoints
            ],
        }
        target = Path(_EVIDENCE_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
        else:
            existing = {"schema": "spire-learn-neural-collection-evidence-v1", "tests": []}
        if existing.get("schema") != "spire-learn-neural-collection-evidence-v1":
            raise AssertionError("collection evidence has an unsupported schema")
        tests = [item for item in existing.get("tests", []) if item.get("mode") != mode]
        tests.append(record)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"schema": existing["schema"], "tests": tests},
                                        sort_keys=True, separators=(",", ":")) + "\n",
                             encoding="utf-8")
        temporary.replace(target)

    def _collect_until_boundary(self, mode: str):
        config_path = self.root / "configs" / "neural" / f"{mode}.toml"
        config = self.load_neural_config(config_path)
        diagnostic_identity = None
        if mode == "run":
            config, diagnostic_identity = make_public_diagnostic_config(config)
            policy = PublicLowestLegalDiagnosticPolicy(diagnostic_identity)
            behavior_version = PUBLIC_DIAGNOSTIC_VERSION
        else:
            config = replace(config, rollout_steps=1)
            behavior_version = 0

        driver = self.make_driver(config, config.cases)
        collector = None
        rollout = None
        run_trace = None
        manual_reset_boundary = None
        previous_collection = None
        sequence = 0
        stop_before_qualifying_run_path = False
        try:
            if mode == "standalone":
                policy = self.build_policy(
                    driver.schema,
                    config.model,
                    seed=config.seed_streams.policy,
                    device=config.device,
                )
            collector = self.Collector(
                driver,
                policy,
                config,
                controller_rng=random.Random(config.seed_streams.noncombat),
                curriculum_rng=random.Random(config.seed_streams.curriculum),
                budget=self.shared_budget,
                policy_generator=object() if mode == "run" else None,
            )
            for _ in range(_MAX_ACTION_SLOTS_PER_MODE + 1):
                previous_counters = (
                    {name: 0 for name in _RUN_COUNTER_FIELDS}
                    if previous_collection is None
                    else self._counter_snapshot(previous_collection)
                )
                rollout = collector.collect(
                    max_learned_decisions=_MAX_ACTION_SLOTS_PER_MODE,
                    max_env_actions=_MAX_ACTION_SLOTS_PER_MODE,
                    policy_version=behavior_version,
                )
                current = rollout.collection
                assert current is not None
                current_counters = self._counter_snapshot(current)
                action_before = self._counter_before_action(previous_collection, current)

                if (mode == "standalone" and manual_reset_boundary is None
                        and previous_collection is not None
                        and previous_collection.tasks_completed >= 1
                        and current.reset_operations > previous_collection.reset_operations):
                    seed_ordinals = [
                        seed.ordinal for seed in current.seed_allocations
                        if previous_collection.next_seed_ordinal <= seed.ordinal < current.next_seed_ordinal
                    ]
                    manual_reset_boundary = {
                        "reset_operations_before": previous_collection.reset_operations,
                        "reset_operations_after": current.reset_operations,
                        "tasks_completed_before": previous_collection.tasks_completed,
                        "reset_slots_before": previous_collection.reset_slots,
                        "reset_slots_after": current.reset_slots,
                        "delivered_reset_slots_before": previous_collection.delivered_reset_slots,
                        "delivered_reset_slots_after": current.delivered_reset_slots,
                        "next_seed_ordinal_before": previous_collection.next_seed_ordinal,
                        "next_seed_ordinal_after": current.next_seed_ordinal,
                        "seed_ordinals": seed_ordinals,
                    }

                for step in rollout.records:
                    sequence += 1
                    step_before = dict(action_before)
                    step_after = dict(current_counters)
                    if mode == "run":
                        if run_trace is not None:
                            if step.stream_id != run_trace["settlement"]["stream_id"]:
                                stop_before_qualifying_run_path = True
                                break
                            run_trace["continuation"] = {
                                "sequence": sequence,
                                "stream_id": step.stream_id,
                                "task_id": step.task_id,
                                "visit": int(step.task_id.rsplit("-", 1)[1]),
                                "decision_id": step.decision_id,
                                "continuation_kind": (
                                    "next_combat" if step.actor_training_mask else "fixed_noncombat"
                                ),
                                "actor_training_mask": step.actor_training_mask,
                                "native_terminated": step.terminated,
                                "native_truncated": step.truncated,
                                "source_frame_sha256": self._frame_hash(step.source),
                                "successor_frame_sha256": self._frame_hash(step.successor),
                                "log_probability": step.choice.log_probability,
                                "value": step.choice.value,
                                "behavior_policy_version": step.choice.behavior_policy_version,
                                "reset_operations_before": step_before["reset_operations"],
                                "reset_operations_after": step_after["reset_operations"],
                                "counters_before": step_before,
                                "counters_after": step_after,
                            }
                            break

                        if step.task_signal.task_terminal:
                            candidates = [checkpoint for checkpoint in current.resource_checkpoints
                                          if checkpoint.stream_id == step.stream_id
                                          and checkpoint.task_id == step.task_id]
                            checkpoint = candidates[-1] if candidates else None
                            if checkpoint is None:
                                stop_before_qualifying_run_path = True
                                break
                            outcome = checkpoint.outcome
                            if outcome != "victory" or step.terminated or step.truncated:
                                stop_before_qualifying_run_path = True
                                break
                            run_trace = {
                                "policy_metadata": _policy_metadata(diagnostic_identity),
                                "settlement": {
                                    "sequence": sequence,
                                    "stream_id": step.stream_id,
                                    "task_id": step.task_id,
                                    "visit": checkpoint.visit,
                                    "outcome": checkpoint.outcome,
                                    "encounter": checkpoint.encounter,
                                    "hp_before_healing": checkpoint.hp_before_healing,
                                    "hp_after_cleanup": checkpoint.hp_after_cleanup,
                                    "decision_id": step.decision_id,
                                    "task_terminal": step.task_signal.task_terminal,
                                    "native_terminated": step.terminated,
                                    "native_truncated": step.truncated,
                                    "actor_training_mask": step.actor_training_mask,
                                    "source_frame_sha256": self._frame_hash(step.source),
                                    "successor_frame_sha256": self._frame_hash(step.successor),
                                    "immediate_checkpoint_sha256": self._frame_hash(checkpoint.frame),
                                    "log_probability": step.choice.log_probability,
                                    "value": step.choice.value,
                                    "behavior_policy_version": step.choice.behavior_policy_version,
                                    "reset_operations_before": step_before["reset_operations"],
                                    "reset_operations_after": step_after["reset_operations"],
                                    "counters_before": step_before,
                                    "counters_after": step_after,
                                },
                                "continuation": None,
                            }
                        elif step.terminated or step.truncated:
                            stop_without_run_victory = True
                            break

                previous_collection = current
                if mode == "run":
                    if run_trace is not None and run_trace["continuation"] is not None:
                        break
                    if stop_before_qualifying_run_path:
                        break
                    if current.stop_reason != "horizon":
                        break
                elif manual_reset_boundary is not None:
                    break
                elif current.stop_reason != "horizon":
                    break

            self.assertIsNotNone(rollout)
            evidence = rollout.collection
            assert evidence is not None
            self.assertLessEqual(evidence.attempted_environment_slots,
                                 _MAX_ACTION_SLOTS_PER_MODE)
            self.assertEqual(evidence.faults, 0)
            if mode == "standalone":
                self.assertGreaterEqual(evidence.tasks_completed, 1)
                self.assertIsNotNone(manual_reset_boundary,
                                     "standalone must record a later delivered manual reset")
            else:
                self.assertEqual(evidence.reset_operations, 1,
                                 "the Run victory and continuation must stay in its initial Run")
                self.assertGreaterEqual(evidence.victories, 1,
                                        "Run collection must observe an actual natural victory")
                self.assertIsNotNone(run_trace)
                self.assertIsNotNone(run_trace["continuation"],
                                     "Run must deliver an action from the victory successor")
            self._record(
                mode, config, rollout, run_continuation=run_trace,
                manual_reset_boundary=manual_reset_boundary,
            )
            return rollout
        finally:
            if collector is not None:
                collector.close()
            else:
                driver.close()

    def test_collect_run_natural_task_end_without_run_reset(self):
        self._collect_until_boundary("run")

    def test_collect_standalone_manual_resets(self):
        self._collect_until_boundary("standalone")


if __name__ == "__main__":
    unittest.main()
