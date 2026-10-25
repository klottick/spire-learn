"""Bounded, opt-in conformance against the accepted Spire Gym runtime."""

from __future__ import annotations

from functools import wraps
import json
import os
import unittest


_RUN_NATIVE = os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1"
_EXPECTED_BUILD_INFO = {
    "extension": "spire-python",
    "revision": "4874fb398128c142bd56f10a12e6fca939a88f93",
    "rust_version": "1.98.0",
}
_MAX_STEP_INVOCATIONS = 128


def _stop_after_failure(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            type(self)._prior_test_failed = True
            raise

    return wrapped


@unittest.skipUnless(
    _RUN_NATIVE,
    "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the approved runtime",
)
class NativeEnvConformanceTests(unittest.TestCase):
    _prior_test_failed = False

    @classmethod
    def setUpClass(cls):
        cls._prior_test_failed = False

    def setUp(self):
        if self._prior_test_failed:
            self.skipTest("earlier conformance failure; no further native scenario")

    @_stop_after_failure
    def test_pack_actions_native_int64(self):
        import numpy as np

        from spire_learn.envs.native import pack_actions

        packed = pack_actions((5, 2, 0))
        self.assertEqual(packed.dtype, np.dtype("=i8"))
        self.assertEqual(packed.dtype.name, "int64")
        self.assertTrue(packed.dtype.isnative)
        self.assertEqual(packed.shape, (3,))
        self.assertEqual(packed.tolist(), [5, 2, 0])

    @_stop_after_failure
    def test_starter_cultist_public_contract(self):
        self._assert_scenario("starter-cultist-a0-v1")

    @_stop_after_failure
    def test_starter_jaw_worm_public_contract(self):
        self._assert_scenario("starter-jaw-worm-a0-v1")

    def _assert_scenario(self, scenario_id):
        from spire_learn.agents.baselines import RandomPolicy
        from spire_learn.contracts import CaseExecution, PublicDecision
        from spire_learn.envs.native import (
            NATIVE_CONTRACT,
            make_single,
            native_snapshot,
            normalize_public_info,
            public_build_info,
        )
        from spire_learn.schema import candidates_for, parse_descriptor

        case = CaseExecution(
            execution_id=f"native-{scenario_id}-selection-seed0",
            case_id=scenario_id,
            split="selection",
            partition="selection",
            scenario_id=scenario_id,
            environment_seed=0,
            policy_seed=0,
            expected_schema_id="spire-lab-combat-observation-v1",
        )
        env = make_single(case)
        summary = {
            "scenario_id": scenario_id,
            "partition": case.partition,
            "environment_seed": case.environment_seed,
            "policy_seed": case.policy_seed,
            "reset_invocations": 0,
            "close_invocations": 0,
            "max_step_invocations": _MAX_STEP_INVOCATIONS,
            "submitted_action_total": 0,
            "delivered_action_total": 0,
            "final_status": "not_started",
            "final_outcome": None,
            "reward_sum": 0.0,
            "schema_id": None,
            "content_id": None,
            "fingerprint": None,
            "native_build_info": None,
            "failure": None,
            "close_failure": None,
            "rng_001": (
                "Project-owned reproducible randomness is not seed-for-seed or "
                "bit-for-bit original-game RNG parity; no original-runtime transfer "
                "is established."
            ),
            "time_001": (
                "The full-run Secret Portal elapsed-time assumption is retained; "
                "these combat scenarios do not validate it."
            ),
        }
        primary_error = None
        close_error = None

        try:
            descriptor = env.observation_spec()
            schema = parse_descriptor(
                descriptor,
                allowed_schema_ids=NATIVE_CONTRACT.schema_ids,
            )
            summary["schema_id"] = schema.schema_id
            summary["content_id"] = schema.content_id
            summary["fingerprint"] = schema.fingerprint

            self.assertEqual(case.split, "selection")
            self.assertEqual(case.partition, "selection")
            self.assertEqual(schema.schema_id, case.expected_schema_id)
            self.assertIn(schema.schema_id, NATIVE_CONTRACT.schema_ids)
            self.assertIn(schema.content_id, NATIVE_CONTRACT.content_ids)
            self.assertEqual(schema.fingerprint, descriptor["fingerprint"])
            self.assertEqual(env.action_space.n, schema.action_count)
            self.assertEqual(env.action_space.start, 0)
            self.assertTrue(
                {"ongoing", "victory", "death"}.issubset(
                    schema.vocabularies["outcome"]
                )
            )

            build_info = public_build_info()
            self.assertIsNotNone(build_info)
            actual_build_info = dict(build_info)
            self.assertEqual(actual_build_info, _EXPECTED_BUILD_INFO)
            summary["native_build_info"] = actual_build_info
            summary["reward_profile"] = {
                "contract_id": NATIVE_CONTRACT.reward_contract_id,
                "evidence": NATIVE_CONTRACT.reward_evidence,
            }

            summary["reset_invocations"] = 1
            reset_observation, reset_info = env.reset(
                seed=0,
                options={"scenario_id": case.scenario_id},
            )
            self.assertTrue(
                env.observation_space.contains(reset_observation),
                "raw reset observation is outside the public observation space",
            )
            self.assertIs(type(reset_info.get("scenario_id")), str)
            self.assertEqual(reset_info["scenario_id"], case.scenario_id)
            self.assertIs(type(reset_info.get("episode_steps")), int)
            self.assertEqual(reset_info["episode_steps"], 0)
            self._check_optional_public_hp(reset_info)
            outcome = self._decode_outcome(reset_info, schema, normalize_public_info)
            self.assertEqual(outcome, "ongoing")
            summary["final_outcome"] = outcome

            current = native_snapshot(reset_observation, schema)
            policy = RandomPolicy(case.policy_seed)
            policy_state = policy.init_state()
            summary["final_status"] = "ongoing"

            while summary["submitted_action_total"] < _MAX_STEP_INVOCATIONS:
                mask, candidates = self._checked_candidates(
                    current,
                    schema,
                    candidates_for,
                    require_nonempty=True,
                )
                public = PublicDecision(
                    current,
                    candidates,
                    mask,
                    schema.fingerprint,
                    "spire-learn-native-current-index-v1",
                )
                choice = policy.act(public, policy_state)
                policy_state = choice.next_state
                action = choice.action

                self.assertIs(type(action), int)
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, schema.action_count)
                self.assertTrue(env.action_space.contains(action))
                self.assertTrue(mask[action], "policy selected an action masked as illegal")
                self.assertIn(action, {candidate.index for candidate in candidates})

                retained_source = tuple(
                    (key, value.dtype, value.shape, tuple(value.values))
                    for key, value in current.items()
                )
                summary["submitted_action_total"] += 1
                (
                    next_raw,
                    reward,
                    terminated,
                    truncated,
                    step_info,
                ) = env.step(action)
                summary["delivered_action_total"] += 1
                self.assertIs(type(reward), float)
                self.assertIs(type(terminated), bool)
                self.assertIs(type(truncated), bool)
                summary["reward_sum"] += reward

                self.assertEqual(
                    retained_source,
                    tuple(
                        (key, value.dtype, value.shape, tuple(value.values))
                        for key, value in current.items()
                    ),
                    "learner-owned source snapshot changed after stepping",
                )
                self.assertTrue(
                    env.observation_space.contains(next_raw),
                    "raw delivered observation is outside the public observation space",
                )
                next_snapshot = native_snapshot(next_raw, schema)

                self.assertIs(type(step_info.get("scenario_id")), str)
                self.assertEqual(step_info["scenario_id"], case.scenario_id)
                self.assertIs(type(step_info.get("episode_steps")), int)
                self.assertEqual(
                    step_info["episode_steps"],
                    summary["delivered_action_total"],
                )
                self._check_optional_public_hp(step_info)
                outcome = self._decode_outcome(
                    step_info,
                    schema,
                    normalize_public_info,
                )
                summary["final_outcome"] = outcome
                self._check_reward_outcome(reward, terminated, outcome)
                self.assertFalse(
                    truncated,
                    "built-in single-environment transitions document truncated=False",
                )

                self._checked_candidates(
                    next_snapshot,
                    schema,
                    candidates_for,
                    require_nonempty=not terminated,
                )
                current = next_snapshot
                if terminated:
                    summary["final_status"] = outcome
                    break
            else:
                # The limit bounds invocations; an ongoing state is not completion.
                summary["final_status"] = "ongoing_at_step_cap"
        except BaseException as error:
            primary_error = error
            summary["final_status"] = "error"
            summary["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            raise
        finally:
            summary["close_invocations"] = 1
            try:
                env.close()
            except BaseException as error:
                close_error = error
                summary["close_failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                if primary_error is None:
                    summary["final_status"] = "close_error"
                    summary["failure"] = summary["close_failure"]
                else:
                    primary_error.add_note(
                        f"environment close also failed: {type(error).__name__}: {error}"
                    )
            print("NATIVE_SCENARIO_EVIDENCE " + json.dumps(summary, sort_keys=True))
            if primary_error is None and close_error is not None:
                raise close_error

    def _checked_candidates(
        self,
        observation,
        schema,
        candidates_for,
        *,
        require_nonempty,
    ):
        raw_mask = observation["action_mask"].values
        self.assertTrue(
            all(type(value) is int and value in (0, 1) for value in raw_mask),
            "current legal mask must contain only integer zero/one values",
        )
        mask = tuple(value == 1 for value in raw_mask)
        candidates = candidates_for(observation, schema)
        self.assertEqual(
            {candidate.index for candidate in candidates},
            {index for index, legal in enumerate(mask) if legal},
            "schema-derived candidates disagree with the current legal mask",
        )
        if require_nonempty:
            self.assertTrue(any(mask), "ongoing nonterminal state has an empty legal mask")
        return mask, candidates

    def _decode_outcome(self, info, schema, normalize_public_info):
        normalized = normalize_public_info(info, schema)
        outcome = normalized.get("outcome")
        self.assertIn(outcome, schema.vocabularies["outcome"])
        return outcome

    def _check_optional_public_hp(self, info):
        for key in ("hp_before_healing", "hp_after_cleanup"):
            if key in info:
                self.assertIs(type(info[key]), int, f"public {key} must be an integer")

    def _check_reward_outcome(self, reward, terminated, outcome):
        if outcome == "victory":
            self.assertEqual(reward, 1.0)
            self.assertTrue(terminated)
        elif outcome == "death":
            self.assertEqual(reward, -1.0)
            self.assertTrue(terminated)
        else:
            self.assertEqual(outcome, "ongoing")
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
