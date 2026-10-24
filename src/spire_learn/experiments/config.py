"""Strict, side-effect-free TOML experiment configuration."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import tomllib

from spire_learn.contracts import CaseExecution
from spire_learn.experiments.cases import CaseManifest


@dataclass(frozen=True)
class ProbeSettings:
    num_envs: int
    workers: int


@dataclass(frozen=True)
class ProbeConfig:
    num_envs: int
    workers: int
    partition: str
    executions: tuple[CaseExecution, ...]
    max_batch_steps: int
    max_wall_seconds: float


@dataclass(frozen=True)
class RunConfig:
    mode: str
    profile: str
    agent: str
    cases: Path
    output_root: Path
    per_case_decisions: int | None
    max_batch_steps: int | None
    max_wall_seconds: float
    traces: bool
    expected_partition: str | None = None
    expected_schema_id: str | None = None
    expected_fingerprint: str | None = None
    gym_checkout: Path | None = None
    split_guard_manifests: tuple[Path, ...] = ()
    probe: ProbeSettings | None = None


_ROOT_REQUIRED = {"mode", "profile", "agent", "cases", "output_root", "max_wall_seconds", "traces"}
_ROOT_OPTIONAL = {"per_case_decisions", "max_batch_steps", "expected_partition", "expected_schema_id",
                  "expected_fingerprint", "gym_checkout", "split_guard_manifests", "probe"}
_PROBE_MODES = {"probe-vector", "probe-torchrl"}
_PROFILES = {"smoke", "development", "overnight"}
_AGENTS = {"random", "play-first"}
_FINGERPRINT = re.compile(r"[0-9a-fA-F]{64}\Z")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path(value: object, name: str, parent: Path) -> Path:
    raw = _nonempty_string(value, name)
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def load_config(path: Path, *, expected_mode: str) -> RunConfig:
    """Parse a strict TOML configuration; perform no filesystem writes."""
    source = Path(path).resolve()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read config {source}: {exc}") from exc
    if not isinstance(raw, dict) or not _ROOT_REQUIRED <= raw.keys() or raw.keys() - (_ROOT_REQUIRED | _ROOT_OPTIONAL):
        raise ValueError("invalid configuration root fields")
    mode = raw["mode"]
    if not isinstance(mode, str) or mode not in {"evaluate", *_PROBE_MODES} or mode != expected_mode:
        raise ValueError("configuration mode does not match expected mode")
    profile, agent = raw["profile"], raw["agent"]
    if not isinstance(profile, str) or profile not in _PROFILES or not isinstance(agent, str) or agent not in _AGENTS:
        raise ValueError("invalid profile or agent")
    wall = raw["max_wall_seconds"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("max_wall_seconds must be positive and finite")
    traces = raw["traces"]
    if type(traces) is not bool:
        raise ValueError("traces must be a boolean")

    probe_raw = raw.get("probe")
    if mode == "evaluate":
        if probe_raw is not None or "max_batch_steps" in raw or "per_case_decisions" not in raw:
            raise ValueError("evaluate requires per_case_decisions and forbids probe/batch settings")
        decisions = _positive_int(raw["per_case_decisions"], "per_case_decisions")
        batch = None
        probe = None
    else:
        if "per_case_decisions" in raw or "max_batch_steps" not in raw or not isinstance(probe_raw, dict):
            raise ValueError("probe modes require batch settings and forbid per_case_decisions")
        if set(probe_raw) != {"num_envs", "workers"}:
            raise ValueError("probe requires only num_envs and workers")
        envs = _positive_int(probe_raw["num_envs"], "probe.num_envs")
        workers = _positive_int(probe_raw["workers"], "probe.workers")
        if envs < 2:
            raise ValueError("probe.num_envs must be at least two")
        batch = _positive_int(raw["max_batch_steps"], "max_batch_steps")
        decisions = None
        probe = ProbeSettings(envs, workers)

    optional_strings = {}
    for key in ("expected_partition", "expected_schema_id", "expected_fingerprint"):
        value = raw.get(key)
        if value is not None:
            optional_strings[key] = _nonempty_string(value, key)
    fingerprint = optional_strings.get("expected_fingerprint")
    if fingerprint is not None and _FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError("expected_fingerprint must be 64 hexadecimal characters")
    guards = raw.get("split_guard_manifests", [])
    if not isinstance(guards, list):
        raise ValueError("split_guard_manifests must be an array of paths")
    gym = _path(raw["gym_checkout"], "gym_checkout", source.parent) if "gym_checkout" in raw else None
    return RunConfig(
        mode, profile, agent, _path(raw["cases"], "cases", source.parent),
        _path(raw["output_root"], "output_root", source.parent), decisions, batch,
        float(wall), traces, optional_strings.get("expected_partition"),
        optional_strings.get("expected_schema_id"), fingerprint, gym,
        tuple(_path(item, "split_guard_manifests[]", source.parent) for item in guards), probe,
    )


def bind_case_expectations(config: RunConfig, execution: CaseExecution) -> CaseExecution:
    """Copy one execution while binding missing config-level schema assertions."""
    if config.expected_partition is not None and execution.partition != config.expected_partition:
        raise ValueError("execution partition conflicts with config expectation")
    for name in ("expected_schema_id", "expected_fingerprint"):
        expected = getattr(config, name)
        declared = getattr(execution, name)
        if expected is not None and declared is not None and expected != declared:
            raise ValueError(f"execution {name} conflicts with config expectation")
    return replace(
        execution,
        expected_schema_id=execution.expected_schema_id or config.expected_schema_id,
        expected_fingerprint=execution.expected_fingerprint or config.expected_fingerprint,
    )


def validate_expectations(config: RunConfig, manifest: CaseManifest) -> None:
    """Check config assertions against every execution before environment setup."""
    if config.expected_partition is not None and any(
        execution.partition != config.expected_partition for execution in manifest.executions
    ):
        raise ValueError("manifest partition conflicts with config expectation")
    for execution in manifest.executions:
        bind_case_expectations(config, execution)


def to_probe_config(config: RunConfig, manifest: CaseManifest) -> ProbeConfig:
    """Build a probe request using only explicitly declared manifest cases."""
    if config.mode not in _PROBE_MODES or config.probe is None or config.max_batch_steps is None:
        raise ValueError("probe configuration required")
    if manifest.split != "selection":
        raise ValueError("probes require a selection manifest")
    if len(manifest.executions) != config.probe.num_envs:
        raise ValueError("manifest execution count must equal probe.num_envs")
    partitions = {execution.partition for execution in manifest.executions}
    if len(partitions) != 1:
        raise ValueError("probe executions must have one partition")
    executions = tuple(bind_case_expectations(config, execution) for execution in manifest.executions)
    return ProbeConfig(config.probe.num_envs, config.probe.workers, next(iter(partitions)),
                       executions, config.max_batch_steps, config.max_wall_seconds)
