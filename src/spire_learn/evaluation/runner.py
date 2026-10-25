"""Single-environment fixed-case evaluation with bounded artifact handling."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import platform
import time
from typing import Any
import uuid

from spire_learn.agents.baselines import PlayFirstPolicy, RandomPolicy
from spire_learn.artifacts.provenance import collect_provenance
from spire_learn.artifacts.writer import RunWriter
from spire_learn.collection.single import run_case
from spire_learn.contracts import CaseExecution, CaseResult, Failure
from spire_learn.envs.native import NATIVE_CONTRACT, make_single
from spire_learn.experiments.budget import Budget
from spire_learn.experiments.cases import CaseManifest, load_cases, validate_splits
from spire_learn.experiments.config import RunConfig, bind_case_expectations, validate_expectations
from spire_learn.schema import EnvironmentContract
from .report import summarize

_RNG_001 = (
    "Project-owned reproducible randomness is not seed-for-seed or bit-for-bit "
    "original-game RNG parity; no original-runtime transfer is established."
)
_TIME_001 = (
    "The full-run Secret Portal elapsed-time assumption is retained; this "
    "evaluation does not validate it."
)
_NATURAL_STATUSES = frozenset({"terminal", "env_truncated"})


@dataclass(frozen=True)
class EvaluationReport:
    summary: Mapping[str, object]
    exit_code: int
    run_dir: Path | None
    fatal_error: Failure | None = None


def evaluate(
    config: RunConfig,
    *,
    allow_final: bool = False,
    env_factory: Callable[[CaseExecution], object] = make_single,
    clock: Callable[[], float] = time.monotonic,
    writer_factory: Callable[[Path, str], RunWriter] = RunWriter.create,
    run_id: str | None = None,
    contract: EnvironmentContract = NATIVE_CONTRACT,
) -> EvaluationReport:
    """Evaluate only the explicitly scheduled cases, using one env per case."""
    manifest: CaseManifest | None = None
    budget: Budget | None = None
    try:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if config.mode != "evaluate":
            raise ValueError("evaluation requires an evaluate configuration")
        if type(allow_final) is not bool:
            raise ValueError("allow_final must be a boolean")
        if not callable(env_factory) or not callable(clock) or not callable(writer_factory):
            raise ValueError("factories and clock must be callable")
        budget = Budget(config.max_wall_seconds, clock)
        manifest = load_cases(config.cases)
        guards = tuple(load_cases(path) for path in config.split_guard_manifests)
        validate_splits((manifest, *guards))
        validate_expectations(config, manifest)
        if manifest.split == "final" and not allow_final:
            raise _BlockedEvaluation("final split requires explicit allow_final opt-in")
        executions = tuple(bind_case_expectations(config, item) for item in manifest.executions)
    except Exception as error:
        failure = _failure(error, phase="preflight", kind="blocked" if isinstance(error, _BlockedEvaluation) else "setup_error")
        results = _preflight_results(manifest.executions if manifest is not None else (), failure)
        return EvaluationReport(summarize(results), 2, None, failure)

    assert budget is not None
    if run_id is None:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex

    try:
        provenance = collect_provenance(
            learner_root=Path(__file__).resolve().parents[3],
            gym_checkout=config.gym_checkout,
        )
        writer = writer_factory(config.output_root, run_id)
    except Exception as error:
        failure = _failure(error, phase="artifact_create", kind="artifact_error")
        results = _not_started_results(executions, "artifact_failure")
        return EvaluationReport(summarize(results), 1, None, failure)

    return _evaluate_with_writer(
        writer,
        config,
        manifest,
        executions,
        budget,
        contract,
        provenance,
        allow_final,
        env_factory,
    )


def _evaluate_with_writer(
    writer: RunWriter,
    config: RunConfig,
    manifest: CaseManifest,
    executions: Sequence[CaseExecution],
    budget: Budget,
    contract: EnvironmentContract,
    provenance: Mapping[str, object],
    allow_final: bool,
    env_factory: Callable[[CaseExecution], object],
) -> EvaluationReport:
    writer_closed = False

    def close_writer_once() -> BaseException | None:
        nonlocal writer_closed
        if writer_closed:
            return None
        writer_closed = True
        try:
            writer.close()
        except BaseException as error:
            return error
        return None

    try:
        run_dir = _writer_run_dir(writer)
        config_record = _config_record(config, contract, provenance, allow_final)
        manifest_record = {
            "path": str(manifest.path),
            "split": manifest.split,
            "executions": executions,
        }
        try:
            writer.write_start(config_record, manifest_record)
        except Exception as error:
            failure = _failure(error, phase="artifact_start", kind="artifact_error")
            close_failure = close_writer_once()
            if close_failure is not None:
                if not isinstance(close_failure, Exception):
                    _note_prior_failure(close_failure, failure, ())
                    raise close_failure
                failure = _with_secondary_context(failure, close_failure)
            results = _not_started_results(executions, "artifact_failure")
            return EvaluationReport(summarize(results), 1, run_dir, failure)

        results: list[CaseResult] = []
        fatal_error: Failure | None = None

        def persist_not_started(pending: Sequence[CaseResult]) -> bool:
            nonlocal fatal_error
            first_result = len(results)
            results.extend(pending)
            for offset, result in enumerate(pending):
                try:
                    writer.write_case(result)
                except Exception as error:
                    failure = _failure(error, phase="case_write", kind="artifact_error")
                    results[first_result + offset] = replace(
                        result,
                        secondary_errors=(*result.secondary_errors, failure),
                    )
                    fatal_error = fatal_error or failure
                    return False
            return True

        for index, execution in enumerate(executions):
            try:
                expired = budget.exhausted()
            except Exception as error:
                failure = _failure(error, phase="budget", kind="exec_error")
                fatal_error = fatal_error or failure
                pending = [_not_started_result(execution, "budget_error", failure)]
                pending.extend(
                    _not_started_result(future, "budget_error", failure)
                    for future in executions[index + 1:]
                )
                persist_not_started(pending)
                break

            if expired:
                persist_not_started(_not_started_results(executions[index:], "wall_time_budget"))
                break

            policy = _policy(config.agent, execution.policy_seed)
            trace_error: Exception | None = None

            def write_transition(transition) -> None:
                nonlocal trace_error
                if trace_error is not None:
                    raise trace_error
                try:
                    writer.write_transition(transition)
                except Exception as error:
                    trace_error = error
                    raise

            result = run_case(
                execution,
                policy,
                env_factory=env_factory,
                budget=budget,
                max_decisions=config.per_case_decisions,
                on_transition=write_transition if config.traces else None,
                contract=contract,
            )

            case_artifact_failures: list[Failure] = []
            if trace_error is not None:
                failure = _failure(trace_error, phase="trace_write", kind="artifact_error")
                case_artifact_failures.append(failure)
                result = _reconcile_trace_failure(result, failure)

            if result.observed_descriptor is not None or result.native_build_info is not None:
                discovery = {
                    "event": "environment_discovered",
                    "execution_id": execution.execution_id,
                    "observed_descriptor": result.observed_descriptor,
                    "native_build_info": result.native_build_info,
                }
                try:
                    writer.update_manifest(discovery)
                except Exception as error:
                    failure = _failure(error, phase="manifest_update", kind="artifact_error")
                    case_artifact_failures.append(failure)
                    result = replace(result, secondary_errors=(*result.secondary_errors, failure))

            try:
                writer.write_case(result)
            except Exception as error:
                failure = _failure(error, phase="case_write", kind="artifact_error")
                case_artifact_failures.append(failure)
                result = replace(result, secondary_errors=(*result.secondary_errors, failure))

            results.append(result)
            if case_artifact_failures:
                fatal_error = fatal_error or case_artifact_failures[0]
                results.extend(_not_started_results(executions[index + 1:], "artifact_failure"))
                break

            if result.status == "not_started" and result.stop_reason == "wall_time_budget":
                persist_not_started(
                    _not_started_results(executions[index + 1:], "wall_time_budget")
                )
                break

        close_failure = close_writer_once()
        if close_failure is not None:
            if not isinstance(close_failure, Exception):
                _note_prior_failure(close_failure, fatal_error, results)
                raise close_failure
            failure = _failure(close_failure, phase="artifact_close", kind="artifact_error")
            fatal_error = fatal_error or failure
            _attach_secondary(results, failure)

        report_summary = summarize(results)
        try:
            writer.finalize(report_summary)
        except Exception as error:
            failure = _failure(error, phase="summary_write", kind="artifact_error")
            fatal_error = fatal_error or failure
            _attach_secondary(results, failure)
            report_summary = summarize(results)

        return EvaluationReport(report_summary, _exit_code(results, fatal_error), run_dir, fatal_error)
    except BaseException as primary:
        close_failure = close_writer_once()
        if close_failure is not None:
            _note_close_failure(primary, close_failure)
        raise


class _BlockedEvaluation(ValueError):
    pass


def _policy(agent: str, policy_seed: int):
    if agent == "random":
        return RandomPolicy(policy_seed)
    if agent == "play-first":
        return PlayFirstPolicy()
    raise ValueError(f"unsupported agent: {agent}")


def _preflight_results(executions: Sequence[CaseExecution], failure: Failure) -> list[CaseResult]:
    return [
        _not_started_result(execution, "preflight_blocked", failure, status="setup_error")
        for execution in executions
    ]


def _not_started_results(executions: Sequence[CaseExecution], reason: str) -> list[CaseResult]:
    return [_not_started_result(execution, reason) for execution in executions]


def _not_started_result(
    execution: CaseExecution,
    reason: str,
    failure: Failure | None = None,
    *,
    status: str = "not_started",
) -> CaseResult:
    return CaseResult(
        execution=execution,
        status=status,
        started=False,
        outcome=None,
        steps=0,
        reward_sum=0.0,
        stop_reason=reason,
        final_resources={},
        primary_failure=failure,
    )


def _config_record(
    config: RunConfig,
    contract: EnvironmentContract,
    provenance: Mapping[str, object],
    allow_final: bool,
) -> dict[str, object]:
    return {
        "mode": config.mode,
        "profile": config.profile,
        "agent": config.agent,
        "cases": str(config.cases),
        "output_root": str(config.output_root),
        "per_case_decisions": config.per_case_decisions,
        "max_batch_steps": config.max_batch_steps,
        "max_wall_seconds": config.max_wall_seconds,
        "traces": config.traces,
        "expected_partition": config.expected_partition,
        "expected_schema_id": config.expected_schema_id,
        "expected_fingerprint": config.expected_fingerprint,
        "gym_checkout": str(config.gym_checkout) if config.gym_checkout is not None else None,
        "split_guard_manifests": [str(path) for path in config.split_guard_manifests],
        "allow_final": allow_final,
        "provenance": provenance,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "environment_contract": {
            "schema_ids": sorted(contract.schema_ids),
            "content_ids": sorted(contract.content_ids),
            "reward_contract_id": contract.reward_contract_id,
            "reward_evidence": contract.reward_evidence,
        },
        "rng_001": _RNG_001,
        "time_001": _TIME_001,
    }


def _writer_run_dir(writer: Any) -> Path | None:
    value = getattr(writer, "run_dir", None)
    return Path(value) if value is not None else None


def _attach_secondary(results: list[CaseResult], failure: Failure) -> None:
    if results:
        results[-1] = replace(
            results[-1],
            secondary_errors=(*results[-1].secondary_errors, failure),
        )


def _reconcile_trace_failure(result: CaseResult, failure: Failure) -> CaseResult:
    if result.primary_failure is not None and result.primary_failure.phase == "trace":
        return replace(result, primary_failure=failure)
    secondary = list(result.secondary_errors)
    for index, prior in enumerate(secondary):
        if prior.phase == "trace":
            secondary[index] = failure
            return replace(result, secondary_errors=tuple(secondary))
    return replace(result, secondary_errors=(*result.secondary_errors, failure))


def _failure(error: Exception, *, phase: str, kind: str) -> Failure:
    try:
        reason = str(error)
    except Exception:
        reason = "exception message unavailable"
    return Failure(kind, reason, phase, type(error).__name__)


def _with_secondary_context(primary: Failure, secondary: BaseException) -> Failure:
    return Failure(
        primary.kind,
        f"{primary.reason}; close also failed: {type(secondary).__name__}: {_exception_message(secondary)}",
        primary.phase,
        primary.exception_type,
    )


def _note_close_failure(primary: BaseException, close_failure: BaseException) -> None:
    try:
        primary.add_note(
            "run writer close also failed: "
            f"{type(close_failure).__name__}: {_exception_message(close_failure)}"
        )
    except BaseException:
        pass


def _note_prior_failure(
    primary: BaseException,
    failure: Failure | None,
    results: Sequence[CaseResult],
) -> None:
    notes = []
    if failure is not None:
        notes.append(
            "prior run failure: "
            f"{failure.kind} during {failure.phase}: {failure.reason}"
        )
    if results:
        result = results[-1]
        notes.append(
            "accepted evaluation context: "
            f"execution_id={result.execution.execution_id}, status={result.status}, "
            f"outcome={result.outcome}, steps={result.steps}"
        )
    for note in notes:
        try:
            primary.add_note(note)
        except BaseException:
            break


def _exception_message(error: BaseException) -> str:
    try:
        return str(error)
    except BaseException:
        return "exception message unavailable"


def _exit_code(results: Sequence[CaseResult], fatal_error: Failure | None) -> int:
    if fatal_error is not None and fatal_error.kind not in {"setup_error", "blocked"}:
        return 1
    if any(result.secondary_errors for result in results):
        return 1
    if any(result.started and result.status not in _NATURAL_STATUSES for result in results):
        return 1
    if any(
        result.status == "budget_stopped"
        or (result.status == "not_started" and result.stop_reason == "wall_time_budget")
        for result in results
    ):
        return 1
    if fatal_error is not None:
        return 2
    if any(result.status == "setup_error" for result in results):
        return 2
    if results and all(result.started and result.status in _NATURAL_STATUSES for result in results):
        return 0
    return 2
