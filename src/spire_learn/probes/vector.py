"""Dependency-free native vector probe report contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..artifacts.codec import to_json_value
from ..contracts import ContractError, Failure


NATIVE_CHECKS = frozenset(
    {"mixed_rows", "old_final", "replacement_reset", "presence_masks", "old_context"}
)
_VERDICTS = frozenset({"compatible", "adaptable", "incompatible", "blocked"})


@dataclass(frozen=True)
class ProbeReport:
    verdict: str
    checks: Mapping[str, str]
    versions: Mapping[str, str | None]
    batch_steps: int
    slot_decisions: int
    replacement_decisions: int
    failures: tuple[Failure, ...]

    def __post_init__(self) -> None:
        if type(self.verdict) is not str:
            raise ContractError("verdict must be a built-in string")
        if self.verdict not in _VERDICTS:
            raise ContractError("verdict must be compatible, adaptable, incompatible, or blocked")
        object.__setattr__(self, "checks", _copy_checks(self.checks))
        object.__setattr__(self, "versions", _copy_versions(self.versions))

        for name in ("batch_steps", "slot_decisions", "replacement_decisions"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"{name} must be a non-negative integer")
        if self.replacement_decisions > self.slot_decisions:
            raise ContractError("replacement_decisions cannot exceed slot_decisions")

        try:
            failures = tuple(self.failures)
        except TypeError as exc:
            raise ContractError("failures must contain Failure records") from exc
        owned_failures = []
        for failure in failures:
            if not isinstance(failure, Failure):
                raise ContractError("failures must contain Failure records")
            if (
                type(failure.kind) is not str
                or type(failure.reason) is not str
                or type(failure.phase) is not str
                or (failure.exception_type is not None and type(failure.exception_type) is not str)
            ):
                raise ContractError("Failure fields must be built-in strings or None for exception_type")
            owned_failures.append(
                Failure(failure.kind, failure.reason, failure.phase, failure.exception_type)
            )
        object.__setattr__(self, "failures", tuple(owned_failures))


def _copy_checks(values: object) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ContractError("checks must be a mapping")
    copied: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or type(value) is not str:
            raise ContractError("checks must map string keys to built-in string values")
        copied[name] = value
    return MappingProxyType(copied)


def _copy_versions(values: object) -> Mapping[str, str | None]:
    if not isinstance(values, Mapping):
        raise ContractError("versions must be a mapping")
    copied: dict[str, str | None] = {}
    for name, value in values.items():
        if not isinstance(name, str) or (value is not None and type(value) is not str):
            raise ContractError("versions must map string keys to built-in strings or None")
        copied[name] = value
    return MappingProxyType(copied)


def probe_verdict(
    checks: Mapping[str, str],
    *,
    required: frozenset[str],
    runtime_available: bool,
    adaptation_verified: bool = False,
) -> str:
    if not isinstance(checks, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in checks.items()
    ):
        raise ContractError("checks must map strings to strings")
    if not isinstance(required, frozenset) or any(
        not isinstance(name, str) for name in required
    ):
        raise ContractError("required must be a frozenset of strings")
    if not required:
        raise ContractError("required must not be empty")
    if type(runtime_available) is not bool:
        raise ContractError("runtime_available must be a boolean")
    if type(adaptation_verified) is not bool:
        raise ContractError("adaptation_verified must be a boolean")

    if not runtime_available:
        return "blocked"
    if any(checks.get(name) == "fail" for name in required):
        return "incompatible"
    if any(checks.get(name) != "pass" for name in required):
        return "blocked"
    return "adaptable" if adaptation_verified else "compatible"


def probe_report_to_json(report: ProbeReport) -> dict[str, object]:
    if not isinstance(report, ProbeReport):
        raise ContractError("report must be a ProbeReport")
    failures: list[dict[str, object]] = []
    for failure in report.failures:
        encoded = to_json_value(failure)
        if not isinstance(encoded, dict):
            raise ContractError("Failure did not encode as a JSON object")
        failures.append(encoded)
    return {
        "schema_id": "spire-learn-probe-v1",
        "verdict": report.verdict,
        "checks": dict(report.checks),
        "versions": dict(report.versions),
        "batch_steps": report.batch_steps,
        "slot_decisions": report.slot_decisions,
        "replacement_decisions": report.replacement_decisions,
        "failures": failures,
    }
