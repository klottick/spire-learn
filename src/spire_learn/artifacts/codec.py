"""Strict JSON conversion for the public, versioned learner records."""

from collections.abc import Mapping
import hashlib
import json
import math

from spire_learn.contracts import ArrayValue, Candidate, CaseExecution, CaseResult, Failure, Transition


def to_json_value(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, ArrayValue):
        return {"dtype": value.dtype, "shape": list(value.shape), "values": list(value.values)}
    if isinstance(value, Candidate):
        return {"index": value.index, "kind": value.kind, "source": value.source,
                "target": value.target, "features": to_json_value(value.features)}
    if isinstance(value, CaseExecution):
        return {name: to_json_value(getattr(value, name)) for name in (
            "execution_id", "case_id", "split", "partition", "scenario_id", "environment_seed",
            "policy_seed", "expected_schema_id", "expected_fingerprint")}
    if isinstance(value, Failure):
        return {name: to_json_value(getattr(value, name)) for name in ("kind", "reason", "phase", "exception_type")}
    if isinstance(value, CaseResult):
        return {name: to_json_value(getattr(value, name)) for name in (
            "execution", "status", "started", "outcome", "steps", "reward_sum", "stop_reason",
            "final_resources", "primary_failure", "secondary_errors", "observed_descriptor", "native_build_info")}
    if isinstance(value, Transition):
        return transition_to_json(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON mapping keys must be strings")
            result[key] = to_json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [to_json_value(item) for item in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(to_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def transition_to_json(transition: Transition) -> dict[str, object]:
    def observation(value):
        return to_json_value(value)
    return {
        "schema_id": "spire-learn-transition-v1",
        "source": observation(transition.source), "action": transition.action, "reward": transition.reward,
        "terminated": transition.terminated, "truncated": transition.truncated,
        "successor": observation(transition.successor),
        "replacement": None if transition.replacement is None else observation(transition.replacement),
        "route": {name: getattr(transition.route, name) for name in (
            "execution_id", "slot", "episode_id", "decision_id", "episode_start")},
    }
