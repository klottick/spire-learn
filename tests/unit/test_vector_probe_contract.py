"""Pure validation for bounded vector-probe inputs and public boundaries."""

import unittest

from spire_learn.contracts import CaseExecution, ContractError, Failure
from spire_learn.experiments.config import ProbeConfig


def _vector():
    from spire_learn.probes import vector

    return vector


def _config(*, workers=1, partition="selection", executions=None):
    if executions is None:
        executions = (
            CaseExecution("e0", "c0", "selection", "selection", "s0", 7, 101),
            CaseExecution("e1", "c1", "selection", "selection", "s1", 11, 103),
        )
    return ProbeConfig(2, workers, partition, executions, 128, 60.0)


class _DirectBaseException(BaseException):
    pass


def _current_info():
    return {
        "episode_id": ("episode-0", "episode-1"),
        "_episode_id": (True, True),
        "scenario_id": ("scenario-0", "scenario-1"),
        "_scenario_id": (True, True),
        "episode_steps": (0, 3),
        "_episode_steps": (True, True),
        "outcome": ("ongoing", "ongoing"),
        "_outcome": (True, True),
        "hp_before_healing": (12, 0),
        "_hp_before_healing": (True, False),
    }


def _final_info():
    return {
        "episode_id": ("episode-0", ""),
        "_episode_id": (True, False),
        "scenario_id": ("scenario-0", ""),
        "_scenario_id": (True, False),
        "episode_steps": (1, 0),
        "_episode_steps": (True, False),
        "outcome": ("death", ""),
        "_outcome": (True, False),
        "hp_after_cleanup": (0, 0),
        "_hp_after_cleanup": (True, False),
    }


class VectorProbeConfigTests(unittest.TestCase):
    def test_valid_direct_probe_config_is_accepted(self):
        _vector().validate_probe_config(_config())

    def test_direct_probe_config_rejects_invalid_bounds_and_case_binding(self):
        invalid = (
            _config(workers=True),
            _config(partition="train"),
            _config(executions=_config().executions[:1]),
            _config(
                executions=(
                    _config().executions[0],
                    CaseExecution("e1", "c1", "train", "selection", "s1", 11, 103),
                )
            ),
            _config(
                executions=(
                    _config().executions[0],
                    CaseExecution("e1", "c1", "selection", "other", "s1", 11, 103),
                )
            ),
            _config(
                executions=(
                    _config().executions[0],
                    CaseExecution("e1", "c1", "selection", "selection", "s1", True, 103),
                )
            ),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ContractError):
                _vector().validate_probe_config(config)

    def test_operational_failure_blocks_success_but_does_not_hide_required_failure(self):
        vector = _vector()
        passed = {name: "pass" for name in ("mixed_rows", "old_final", "replacement_reset", "presence_masks", "old_context")}
        self.assertEqual(vector.vector_probe_verdict(passed, operational_failure=True, runtime_available=True), "blocked")
        passed["old_final"] = "fail"
        self.assertEqual(vector.vector_probe_verdict(passed, operational_failure=True, runtime_available=True), "incompatible")


class VectorCloseHandlingTests(unittest.TestCase):
    def test_close_process_control_preserves_prior_structured_failures(self):
        attempt_close = _vector()._attempt_close
        prior_failure = Failure(
            "operational_error", "recorded probe failure", "step", "RuntimeError"
        )
        calls = []
        close_error = KeyboardInterrupt("close interrupted")

        def close():
            calls.append("close")
            raise close_error

        with self.assertRaises(KeyboardInterrupt) as raised:
            attempt_close(close, failures=(prior_failure,))
        self.assertIs(raised.exception, close_error)
        self.assertEqual(calls, ["close"])
        self.assertTrue(
            any("recorded probe failure" in note for note in close_error.__notes__)
        )

    def test_close_failure_is_attached_to_each_primary_process_control(self):
        attempt_close = _vector()._attempt_close
        primary_errors = (
            KeyboardInterrupt("interrupt"),
            SystemExit("exit"),
            _DirectBaseException("direct base exception"),
        )
        for primary in primary_errors:
            with self.subTest(primary=type(primary).__name__):
                calls = []
                close_error = RuntimeError("close failed")

                def close():
                    calls.append("close")
                    raise close_error

                with self.assertRaises(type(primary)) as raised:
                    attempt_close(close, primary=primary)
                self.assertIs(raised.exception, primary)
                self.assertEqual(calls, ["close"])
                self.assertTrue(any("RuntimeError: close failed" in note for note in primary.__notes__))

    def test_process_control_from_close_without_primary_is_reraised(self):
        attempt_close = _vector()._attempt_close
        calls = []
        close_error = _DirectBaseException("close interrupted")

        def close():
            calls.append("close")
            raise close_error

        with self.assertRaises(_DirectBaseException) as raised:
            attempt_close(close)
        self.assertIs(raised.exception, close_error)
        self.assertEqual(calls, ["close"])

    def test_ordinary_close_exception_without_primary_is_returned_for_reporting(self):
        attempt_close = _vector()._attempt_close
        calls = []
        close_error = RuntimeError("ordinary close failure")

        def close():
            calls.append("close")
            raise close_error

        self.assertIs(attempt_close(close), close_error)
        self.assertEqual(calls, ["close"])


class VectorPresenceTests(unittest.TestCase):
    def test_vector_tensor_shape_requires_exact_full_width_descriptor_batch(self):
        validate_vector_batch_shape = _vector().validate_vector_batch_shape
        validate_vector_batch_shape((2, 3, 4), (3, 4), width=2)
        for batch_shape in ((3, 3, 4), (1, 3, 4), (2, 3)):
            with self.subTest(batch_shape=batch_shape), self.assertRaises(ContractError):
                validate_vector_batch_shape(batch_shape, (3, 4), width=2)

    def test_current_info_requires_full_width_required_presence(self):
        validate_current_info_presence = _vector().validate_current_info_presence
        values = validate_current_info_presence(_current_info(), width=2)
        self.assertEqual(values["episode_steps"], (0, 3))

        missing = _current_info()
        missing["_episode_steps"] = (True, False)
        with self.assertRaises(ContractError):
            validate_current_info_presence(missing, width=2)

        no_optional_values = _current_info()
        no_optional_values["_hp_before_healing"] = (False, False)
        with self.assertRaises(ContractError):
            validate_current_info_presence(no_optional_values, width=2)

    def test_final_presence_accepts_mixed_rows_and_masked_placeholders(self):
        validate_final_batch_presence = _vector().validate_final_batch_presence
        info = {
            "final_obs": ({"public": "old"}, None),
            "_final_obs": (True, False),
            "final_info": _final_info(),
            "_final_info": (True, False),
        }
        observations, final_info = validate_final_batch_presence(info, (True, False))
        self.assertEqual(observations, ({"public": "old"}, None))
        self.assertEqual(final_info["episode_id"], ("episode-0", ""))

    def test_final_presence_accepts_no_final_keys_when_no_row_ended(self):
        validate_final_batch_presence = _vector().validate_final_batch_presence
        self.assertEqual(
            validate_final_batch_presence({}, (False, False)),
            (None, None),
        )

    def test_final_presence_rejects_partial_keys_masks_and_unsupported_rows(self):
        validate_final_batch_presence = _vector().validate_final_batch_presence
        valid = {
            "final_obs": ({"public": "old"}, None),
            "_final_obs": (True, False),
            "final_info": _final_info(),
            "_final_info": (True, False),
        }
        cases = []

        partial = dict(valid)
        partial.pop("_final_info")
        cases.append(partial)

        missing_nested_mask = dict(valid)
        missing_nested_mask["final_info"] = dict(_final_info())
        missing_nested_mask["final_info"].pop("_episode_id")
        cases.append(missing_nested_mask)

        unsupported_mask = dict(valid)
        unsupported_mask["final_info"] = dict(_final_info())
        unsupported_mask["final_info"]["_outcome"] = (True, True)
        cases.append(unsupported_mask)

        missing_ended_observation = dict(valid)
        missing_ended_observation["final_obs"] = (None, None)
        cases.append(missing_ended_observation)

        non_none_live_observation = dict(valid)
        non_none_live_observation["final_obs"] = ({"public": "old"}, {"public": "live"})
        cases.append(non_none_live_observation)

        for info in cases:
            with self.subTest(info=info), self.assertRaises(ContractError):
                validate_final_batch_presence(info, (True, False))


class VectorEpisodeIdentityTests(unittest.TestCase):
    def test_episode_identity_separates_old_final_from_replacement_reset(self):
        _vector().validate_episode_identity(
            source_ids=("old-0", "live-1"),
            source_steps=(4, 2),
            current_ids=("new-0", "live-1"),
            current_steps=(0, 3),
            ended=(True, False),
            final_info={"episode_id": ("old-0", ""), "episode_steps": (5, 0)},
        )

    def test_episode_identity_rejects_wrong_old_final_and_reset_identity(self):
        validate_episode_identity = _vector().validate_episode_identity
        arguments = {
            "source_ids": ("old-0", "live-1"),
            "source_steps": (4, 2),
            "current_ids": ("new-0", "live-1"),
            "current_steps": (0, 3),
            "ended": (True, False),
            "final_info": {"episode_id": ("wrong-old", ""), "episode_steps": (5, 0)},
        }
        with self.assertRaises(ContractError):
            validate_episode_identity(**arguments)

        arguments["final_info"] = {
            "episode_id": ("old-0", ""),
            "episode_steps": (5, 0),
        }
        arguments["current_steps"] = (1, 3)
        with self.assertRaises(ContractError):
            validate_episode_identity(**arguments)


if __name__ == "__main__":
    unittest.main()
