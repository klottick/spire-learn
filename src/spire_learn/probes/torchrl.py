"""Optional TorchRL compatibility probe with dependency-free contracts."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

from ..artifacts.writer import RunWriter
from ..contracts import ContractError, SetupBlocked
from ..experiments.config import ProbeConfig
from . import vector
from .vector import NATIVE_CHECKS


TORCHRL_CHECKS = NATIVE_CHECKS | frozenset(
    {"observation_mapping", "mask_sampling", "mask_log_prob"}
)


class _FacadeCallGuard:
    """Gate facade calls until the controller has explicitly permitted them."""

    def __init__(self, *, batch_size: int, require_native_actions: bool = False):
        if type(batch_size) is not int or batch_size <= 0:
            raise ContractError("batch_size must be a positive built-in integer")
        if type(require_native_actions) is not bool:
            raise ContractError("require_native_actions must be a boolean")
        self.batch_size = batch_size
        self.require_native_actions = require_native_actions
        self.failed = False
        self.constructed = False
        self.reset_delivered = False
        self._reset_permit: tuple[tuple[int, ...], tuple[str, ...]] | None = None
        self._step_permit: tuple[int, ...] | None = None
        self._last_step_error: BaseException | None = None
        self._close_attempted = False
        self._counts = {
            "reset_attempts": 0,
            "reset_forwarded": 0,
            "reset_delivered": 0,
            "reset_captured": 0,
            "step_attempts": 0,
            "step_forwarded": 0,
            "step_delivered": 0,
            "step_captured": 0,
            "denied_calls": 0,
            "close_attempts": 0,
            "close_forwarded": 0,
        }

    @property
    def counts(self) -> Mapping[str, int]:
        return dict(self._counts)

    def _deny(self, message: str) -> None:
        self.failed = True
        self._counts["denied_calls"] += 1
        raise SetupBlocked(message)

    def construction_complete(self) -> None:
        if self.failed or self.constructed or self._close_attempted:
            self._deny("facade construction cannot be completed in this guard state")
        self.constructed = True

    def permit_initial_reset(self, *, seed: object, options: object) -> None:
        if (
            self.failed
            or self._close_attempted
            or not self.constructed
            or self.reset_delivered
            or self._reset_permit is not None
        ):
            self._deny("the single initial reset is not available in this guard state")
        if type(seed) not in (tuple, list) or len(seed) != self.batch_size or any(
            type(value) is not int or value < 0 for value in seed
        ):
            self._deny("initial reset requires one declared non-negative seed per row")
        if not isinstance(options, Mapping):
            self._deny("initial reset requires declared scenario options")
        if set(options) != {"scenario_ids"}:
            self._deny("initial reset options must contain only scenario_ids")
        scenarios = options.get("scenario_ids")
        if type(scenarios) not in (tuple, list) or len(scenarios) != self.batch_size or any(
            type(value) is not str or not value for value in scenarios
        ):
            self._deny("initial reset requires one declared scenario per row")
        self._reset_permit = (tuple(seed), tuple(scenarios))

    def forward_reset(
        self,
        delegate: Callable[..., object],
        *,
        seed: object = None,
        options: object = None,
        capture: Callable[[object], object] | None = None,
    ) -> object:
        self._counts["reset_attempts"] += 1
        if (
            self.failed
            or self._close_attempted
            or not self.constructed
            or self.reset_delivered
            or self._reset_permit is None
        ):
            self._deny("reset was not explicitly permitted")
        expected_seed, expected_scenarios = self._reset_permit
        if (
            type(seed) not in (tuple, list)
            or len(seed) != self.batch_size
            or any(type(value) is not int or value < 0 for value in seed)
            or tuple(seed) != expected_seed
        ):
            self._deny("reset seeds differ from the declared full-batch reset")
        if (
            not isinstance(options, Mapping)
            or set(options) != {"scenario_ids"}
            or type(options["scenario_ids"]) not in (tuple, list)
            or len(options["scenario_ids"]) != self.batch_size
            or any(type(value) is not str or not value for value in options["scenario_ids"])
            or tuple(options["scenario_ids"]) != expected_scenarios
        ):
            self._deny("reset scenarios differ from the declared full-batch reset")

        # Consume and count the permit before dispatch so an accepted effect is
        # never repeated after a delegate or capture failure.
        self._reset_permit = None
        self.reset_delivered = True
        self._counts["reset_forwarded"] += 1
        try:
            result = delegate(seed=seed, options=options)
            self._counts["reset_delivered"] += 1
            if capture is not None:
                capture(result)
                self._counts["reset_captured"] += 1
        except BaseException:
            self.failed = True
            raise
        return result

    def permit_step(self, actions: Sequence[int]) -> None:
        if self.failed or self._close_attempted or not self.reset_delivered or self._step_permit is not None:
            self._deny("a full-batch step is not available in this guard state")
        try:
            expected = tuple(actions)
        except TypeError:
            self._deny("step permit requires a full-batch action sequence")
            return
        if len(expected) != self.batch_size or any(
            type(action) is not int or action < 0 for action in expected
        ):
            self._deny("step permit requires one non-negative built-in action per row")
        self._last_step_error = None
        self._step_permit = expected

    def forward_step(
        self,
        delegate: Callable[[object], object],
        actions: object,
        *,
        capture: Callable[[object], object] | None = None,
    ) -> object:
        self._counts["step_attempts"] += 1
        if self.failed or self._close_attempted or not self.reset_delivered or self._step_permit is None:
            self._deny("step was not explicitly permitted")
        try:
            actual = _full_batch_actions(
                actions,
                width=self.batch_size,
                require_native_array=self.require_native_actions,
            )
        except SetupBlocked as error:
            self._deny(str(error))
        if actual != self._step_permit:
            self._deny("step actions differ from the controller permit")
        self._step_permit = None
        self._counts["step_forwarded"] += 1
        try:
            result = delegate(actions)
            self._counts["step_delivered"] += 1
            if capture is not None:
                capture(result)
                self._counts["step_captured"] += 1
        except BaseException as error:
            self._last_step_error = error
            self.failed = True
            raise
        return result

    def close_once(self, delegate: Callable[[], object]) -> object | None:
        self._counts["close_attempts"] += 1
        if self._close_attempted:
            return None
        # Mark before forwarding; destructor/cleanup paths cannot retry close.
        self._close_attempted = True
        self._counts["close_forwarded"] += 1
        try:
            return delegate()
        except BaseException:
            self.failed = True
            raise


def _full_batch_actions(
    value: object, *, width: int, require_native_array: bool = False
) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None and tuple(shape) != (width,):
        raise SetupBlocked("facade step requires exactly one full-width action batch")
    dtype = getattr(value, "dtype", None)
    if require_native_array and (
        shape is None
        or dtype is None
        or getattr(dtype, "name", None) != "int64"
        or getattr(dtype, "isnative", None) is not True
    ):
        raise SetupBlocked("TorchRL actions must be native-endian int64[B]")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        values = tolist()
    else:
        values = value
    if not isinstance(values, (tuple, list)) or len(values) != width or any(
        type(action) is not int or action < 0 for action in values
    ):
        raise SetupBlocked("facade step actions must be a full-width integer batch")
    return tuple(values)


def _raise_step_error_or_denial(
    guard: _FacadeCallGuard,
    *,
    denied_calls_before: int,
    wrapper_error: BaseException | None,
) -> None:
    """Preserve real step errors; synthesize a block only for a denied call."""
    if wrapper_error is not None:
        raise wrapper_error
    if guard._last_step_error is not None:
        raise guard._last_step_error
    if guard.counts["denied_calls"] > denied_calls_before:
        raise SetupBlocked("TorchRL step returned after a guarded environment call was denied")


def _snapshot_final_rows(
    raw_rows: object, ended: Sequence[bool], schema: object
) -> tuple[Mapping[str, object] | None, ...]:
    """Own each present public final row before TorchRL can mutate placeholders."""
    ended_rows = tuple(ended)
    if not ended_rows or any(type(value) is not bool for value in ended_rows):
        raise ContractError("ended must be a non-empty boolean sequence")
    rows = vector._sequence_values(
        raw_rows,
        width=len(ended_rows),
        label="final_obs",
        object_vector=True,
    )
    snapshots: list[Mapping[str, object] | None] = []
    for index, (row, did_end) in enumerate(zip(rows, ended_rows, strict=True)):
        if did_end:
            if not isinstance(row, Mapping):
                raise vector._VectorContractViolation(
                    "presence_masks", f"ended row {index} has no public final observation"
                )
            snapshots.append(vector._snapshot_one(row, schema, check="old_final"))
        else:
            if row is not None:
                raise vector._VectorContractViolation(
                    "presence_masks", f"live row {index} contains a final observation"
                )
            snapshots.append(None)
    return tuple(snapshots)


class _DiagnosticSampleBudget:
    """Reserve bounded diagnostic samples before calling an optional sampler."""

    def __init__(self, *, max_samples: int):
        if type(max_samples) is not int or max_samples <= 0:
            raise ContractError("max_samples must be a positive built-in integer")
        self.max_samples = max_samples
        self.samples_reserved = 0

    def sample(self, *, rows: int, sampler: Callable[[], object]) -> object:
        if type(rows) is not int or rows <= 0:
            raise ContractError("rows must be a positive built-in integer")
        if self.samples_reserved + rows > self.max_samples:
            raise ContractError("diagnostic sample budget is exhausted")
        self.samples_reserved += rows
        return sampler()


def _validate_mask_diagnostic(
    masks: Sequence[Sequence[bool]],
    sampled_actions: Sequence[int],
    log_probabilities: Sequence[float],
    *,
    width: int,
    action_count: int,
) -> None:
    if (
        type(width) is not int
        or width <= 0
        or type(action_count) is not int
        or action_count <= 0
        or type(masks) not in (tuple, list)
        or len(masks) != width
    ):
        raise vector._VectorContractViolation(
            "mask_sampling", "diagnostic mask must have exact boolean shape [B,A]"
        )
    normalized_masks = []
    for row in masks:
        if (
            type(row) not in (tuple, list)
            or len(row) != action_count
            or any(type(value) is not bool for value in row)
            or not any(row)
        ):
            raise vector._VectorContractViolation(
                "mask_sampling", "diagnostic masks must be non-empty boolean rows [B,A]"
            )
        normalized_masks.append(tuple(row))
    if (
        type(sampled_actions) not in (tuple, list)
        or len(sampled_actions) != width
        or any(type(action) is not int or not 0 <= action < action_count for action in sampled_actions)
        or any(not normalized_masks[row][action] for row, action in enumerate(sampled_actions))
    ):
        raise vector._VectorContractViolation(
            "mask_sampling", "MaskedCategorical sampled an illegal or malformed action"
        )
    if (
        type(log_probabilities) not in (tuple, list)
        or len(log_probabilities) != width
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in log_probabilities
        )
    ):
        raise vector._VectorContractViolation(
            "mask_log_prob", "legal sampled actions require finite per-row log probabilities"
        )


@dataclass(frozen=True)
class _CapturedReset:
    rows: tuple[Mapping[str, object], ...]
    current_info: Mapping[str, tuple[object, ...]]


@dataclass(frozen=True)
class _CapturedStep:
    main_rows: tuple[Mapping[str, object], ...]
    final_rows: tuple[Mapping[str, object] | None, ...]
    current_info: Mapping[str, tuple[object, ...]]
    final_info: Mapping[str, tuple[object, ...]] | None
    final_presence: tuple[bool, ...] | None
    final_info_presence: tuple[bool, ...] | None
    final_info_masks: Mapping[str, tuple[bool, ...]] | None
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    ended: tuple[bool, ...]


def _owned_current_info(
    info: Mapping[str, object], *, width: int
) -> Mapping[str, tuple[object, ...]]:
    values = vector.validate_current_info_presence(info, width=width)
    copied: dict[str, tuple[object, ...]] = {key: tuple(column) for key, column in values.items()}
    for key in (*values.keys(),):
        copied["_" + key] = vector._sequence_values(
            info["_" + key], width=width, label="_" + key, boolean=True
        )
    return MappingProxyType(copied)


def _capture_reset_result(
    result: object, schema: object, *, width: int
) -> _CapturedReset:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise vector._VectorContractViolation(
            "presence_masks", "initial reset must return observations and public info"
        )
    observation, info = result
    rows = vector._snapshot_batch(
        observation, schema, width=width, check="observation_mapping"
    )
    if not isinstance(info, Mapping):
        raise vector._VectorContractViolation("presence_masks", "reset info must be a mapping")
    return _CapturedReset(rows, _owned_current_info(info, width=width))


def _capture_step_result(
    result: object, schema: object, *, width: int
) -> _CapturedStep:
    if not isinstance(result, (tuple, list)) or len(result) != 5:
        raise vector._VectorContractViolation(
            "observation_mapping", "step must return the five public Gymnasium values"
        )
    observation, raw_rewards, raw_terminated, raw_truncated, info = result
    if not isinstance(info, Mapping):
        raise vector._VectorContractViolation("presence_masks", "step info must be a mapping")
    main_rows = vector._snapshot_batch(
        observation, schema, width=width, check="observation_mapping"
    )
    rewards = vector._read_rewards(raw_rewards, width)
    terminated = vector._sequence_values(
        raw_terminated, width=width, label="terminated", boolean=True
    )
    truncated = vector._sequence_values(
        raw_truncated, width=width, label="truncated", boolean=True
    )
    ended = tuple(term or trunc for term, trunc in zip(terminated, truncated, strict=True))
    _, final_info = vector.validate_final_batch_presence(info, ended)
    final_rows = (
        _snapshot_final_rows(info["final_obs"], ended, schema)
        if "final_obs" in info
        else (None,) * width
    )
    final_presence = (
        vector._sequence_values(
            info["_final_obs"], width=width, label="_final_obs", boolean=True
        )
        if "_final_obs" in info
        else None
    )
    final_info_presence = (
        vector._sequence_values(
            info["_final_info"], width=width, label="_final_info", boolean=True
        )
        if "_final_info" in info
        else None
    )
    final_info_masks = None
    if final_info is not None:
        nested = info["final_info"]
        final_info_masks = MappingProxyType(
            {
                key: vector._sequence_values(
                    nested["_" + key],
                    width=width,
                    label="final_info._" + key,
                    boolean=True,
                )
                for key in final_info
            }
        )
        final_info = MappingProxyType(
            {key: tuple(column) for key, column in final_info.items()}
        )
    return _CapturedStep(
        main_rows=main_rows,
        final_rows=final_rows,
        current_info=_owned_current_info(info, width=width),
        final_info=final_info,
        final_presence=final_presence,
        final_info_presence=final_info_presence,
        final_info_masks=final_info_masks,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        ended=ended,
    )


def _mapped_observation_rows(
    tensordict: object, schema: object, *, width: int
) -> tuple[Mapping[str, object], ...]:
    """Convert GymWrapper's public CPU TensorDict leaves to owned core rows."""
    keys = getattr(tensordict, "keys", None)
    if not callable(keys):
        raise vector._VectorContractViolation(
            "observation_mapping", "GymWrapper output does not expose TensorDict keys"
        )
    try:
        mapped_keys = tuple(keys())
    except Exception as error:
        raise vector._VectorContractViolation(
            "observation_mapping", "GymWrapper TensorDict keys are unreadable"
        ) from error
    specs = tuple(schema.tensors)  # type: ignore[union-attr]
    expected_keys = tuple(spec.key for spec in specs)
    if len(set(mapped_keys)) != len(mapped_keys) or not set(expected_keys) <= set(mapped_keys):
        raise vector._VectorContractViolation(
            "observation_mapping",
            "GymWrapper TensorDict must contain every public descriptor tensor",
        )

    arrays: dict[str, object] = {}
    for spec in specs:
        key = spec.key
        try:
            tensor = tensordict[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as error:
            raise vector._VectorContractViolation(
                "observation_mapping", f"GymWrapper TensorDict is missing {key}"
            ) from error
        device = getattr(tensor, "device", None)
        if device is None or getattr(device, "type", None) != "cpu":
            raise vector._VectorContractViolation(
                "observation_mapping", f"GymWrapper tensor {key} is not on CPU"
            )
        detach = getattr(tensor, "detach", None)
        numpy = getattr(detach() if callable(detach) else tensor, "numpy", None)
        if not callable(numpy):
            raise vector._VectorContractViolation(
                "observation_mapping", f"GymWrapper tensor {key} cannot expose a CPU numeric array"
            )
        try:
            arrays[key] = numpy()
        except Exception as error:
            raise vector._VectorContractViolation(
                "observation_mapping", f"GymWrapper tensor {key} cannot be read as a CPU array"
            ) from error
    try:
        return vector._snapshot_batch(arrays, schema, width=width, check="observation_mapping")
    except vector._VectorContractViolation as error:
        if error.check == "observation_mapping":
            raise
        raise vector._VectorContractViolation("observation_mapping", str(error)) from error


class _GuardedVectorFacade:
    """Minimal public Gymnasium facade with pre-forward permits and raw capture."""

    _FORWARDED_PUBLIC_ATTRIBUTES = frozenset(
        {
            "action_space",
            "observation_space",
            "single_action_space",
            "single_observation_space",
            "num_envs",
            "render_mode",
            "reward_range",
            "spec",
        }
    )

    def __init__(self, env: object, guard: _FacadeCallGuard, schema: object):
        self._env = env
        self._guard = guard
        self._schema = schema
        self._width = guard.batch_size
        self.last_reset: _CapturedReset | None = None
        self.last_step: _CapturedStep | None = None
        self.last_reset_raw_result: object | None = None
        self.last_step_raw_result: object | None = None

    def __getattr__(self, name: str) -> object:
        if name not in self._FORWARDED_PUBLIC_ATTRIBUTES:
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def metadata(self) -> Mapping[str, object]:
        value = getattr(self._env, "metadata", None)
        if not isinstance(value, Mapping):
            raise SetupBlocked("public vector facade metadata is not a mapping")
        return value

    @property
    def autoreset_mode(self) -> object:
        try:
            return self.metadata["autoreset_mode"]
        except KeyError as error:
            raise SetupBlocked(
                "public vector metadata does not declare autoreset_mode"
            ) from error

    @property
    def unwrapped(self) -> object:
        return getattr(self._env, "unwrapped", self._env)

    def reset(self, *, seed: object = None, options: object = None) -> object:
        def capture(result: object) -> None:
            self.last_reset_raw_result = result
            self.last_reset = _capture_reset_result(
                result, self._schema, width=self._width
            )

        return self._guard.forward_reset(
            self._env.reset,
            seed=seed,
            options=options,
            capture=capture,
        )

    def step(self, actions: object) -> object:
        def capture(result: object) -> None:
            self.last_step_raw_result = result
            self.last_step = _capture_step_result(
                result, self._schema, width=self._width
            )

        return self._guard.forward_step(
            self._env.step,
            actions,
            capture=capture,
        )

    def close(self) -> object | None:
        return self._guard.close_once(self._env.close)


def _spec_entries(spec: object) -> Mapping[object, object]:
    if isinstance(spec, Mapping):
        return spec
    keys = getattr(spec, "keys", None)
    if not callable(keys):
        raise SetupBlocked("TorchRL observation/info spec does not expose public keys")
    try:
        names = tuple(keys())
    except TypeError as error:
        raise SetupBlocked("TorchRL observation/info spec keys are unreadable") from error
    try:
        return {name: spec[name] for name in names}  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise SetupBlocked("TorchRL observation/info spec entries are unreadable") from error


def _spec_signature(spec: object) -> tuple[object, tuple[int, ...]]:
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        shape = tuple(spec[1])
        return spec[0], shape
    dtype = getattr(spec, "dtype", None)
    shape = getattr(spec, "shape", None)
    if dtype is None or shape is None:
        raise SetupBlocked("TorchRL bridge cannot verify a final observation leaf spec")
    try:
        return dtype, tuple(shape)
    except TypeError as error:
        raise SetupBlocked("TorchRL bridge found an invalid final observation leaf spec") from error


def _passive_register_info_reader(wrapper: object, reader: object) -> object:
    """Register TorchRL's supplied reader without its rollout-based discovery."""
    if reader is None or not callable(reader):
        raise SetupBlocked("TorchRL did not supply a callable SAME_STEP info reader")
    registered = getattr(wrapper, "info_dict_reader", None)
    if registered:
        raise SetupBlocked("TorchRL wrapper already has an info reader")
    observation_spec = getattr(wrapper, "observation_spec", None)
    zero_factory = getattr(observation_spec, "zero", None)
    if not callable(zero_factory):
        raise SetupBlocked("TorchRL wrapper has no known public observation-spec zero")
    registration_tensordict = zero_factory()

    # terminal_obs_reader uses the empty-info pass to establish its final spec.
    # This call is passive: it receives only authored/spec zero data, never env IO.
    reader({}, registration_tensordict)
    info_spec = getattr(reader, "info_spec", None)
    if info_spec is None:
        raise SetupBlocked("TorchRL reader did not initialize a public info spec")
    try:
        final_spec = info_spec["final"]
    except (KeyError, TypeError) as error:
        raise SetupBlocked("TorchRL reader info spec lacks final observations") from error
    expected_entries = _spec_entries(observation_spec)
    actual_entries = _spec_entries(final_spec)
    if set(expected_entries) != set(actual_entries):
        raise SetupBlocked("TorchRL final spec keys differ from the public observation spec")
    for key, expected in expected_entries.items():
        if _spec_signature(expected) != _spec_signature(actual_entries[key]):
            raise SetupBlocked(f"TorchRL final spec for {key!r} differs from public observations")

    setter = getattr(wrapper, "set_info_dict_reader", None)
    if not callable(setter):
        raise SetupBlocked("TorchRL wrapper does not expose set_info_dict_reader")
    result = setter(info_dict_reader=reader)
    after = getattr(wrapper, "info_dict_reader", None)
    if not isinstance(after, Sequence) or tuple(after).count(reader) != 1:
        raise SetupBlocked("TorchRL wrapper did not register the passive info reader exactly once")
    return result


def _passive_gym_wrapper_class(gym_wrapper_class: type) -> type:
    """Make the minimal public-method override required to avoid auto-rollout."""
    class PassiveGymWrapper(gym_wrapper_class):
        def auto_register_info_dict(self, ignore_private=True, *, info_dict_reader=None):
            del ignore_private
            return _passive_register_info_reader(self, info_dict_reader)

    return PassiveGymWrapper


def _torchrl_probe_verdict(
    checks: Mapping[str, str],
    *,
    runtime_available: bool,
    operational_failure: bool,
    adaptation_verified: bool = False,
) -> str:
    if type(operational_failure) is not bool:
        raise ContractError("operational_failure must be a boolean")
    verdict = vector.probe_verdict(
        checks,
        required=TORCHRL_CHECKS,
        runtime_available=runtime_available,
        adaptation_verified=adaptation_verified,
    )
    if operational_failure and not any(checks.get(name) == "fail" for name in TORCHRL_CHECKS):
        return "blocked"
    return verdict


class _TorchRLRuntimeUnavailable(Exception):
    pass


class _TorchRLSetupFailure(Exception):
    pass


def _load_torchrl_runtime() -> Mapping[str, object]:
    """Import optional framework APIs only when the probe is explicitly run."""
    try:
        import gymnasium
        import tensordict
        import torch
        from torchrl.envs import GymWrapper
        from torchrl.envs.utils import step_mdp
        from torchrl.modules import MaskedCategorical
    except Exception as error:
        raise _TorchRLRuntimeUnavailable(
            f"TorchRL CPU runtime imports are incomplete: {type(error).__name__}: {error}"
        ) from error
    return MappingProxyType(
        {
            "gymnasium": gymnasium,
            "tensordict": tensordict,
            "torch": torch,
            "GymWrapper": GymWrapper,
            "step_mdp": step_mdp,
            "MaskedCategorical": MaskedCategorical,
        }
    )


def _torchrl_versions() -> dict[str, str | None]:
    import platform

    return {
        "python": platform.python_version(),
        "spire_lab": vector._distribution_version("spire-lab"),
        "gymnasium": vector._distribution_version("gymnasium"),
        "numpy": vector._distribution_version("numpy"),
        "torch": vector._distribution_version("torch"),
        "torchrl": vector._distribution_version("torchrl"),
        "tensordict": vector._distribution_version("tensordict"),
    }


def _torchrl_row_values(value: object, *, width: int, label: str) -> tuple[object, ...]:
    """Read one bounded CPU TorchRL vector without coercing its shape or device."""
    device = getattr(value, "device", None)
    shape = getattr(value, "shape", None)
    if device is None or getattr(device, "type", None) != "cpu" or shape is None:
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL {label} must be a public CPU tensor"
        )
    try:
        shape = tuple(shape)
    except TypeError as error:
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL {label} has an invalid shape"
        ) from error
    if shape not in {(width,), (width, 1)}:
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL {label} must contain exactly one value per row"
        )
    detach = getattr(value, "detach", None)
    tensor = detach() if callable(detach) else value
    numpy = getattr(tensor, "numpy", None)
    if not callable(numpy):
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL {label} cannot be read as a CPU array"
        )
    try:
        array = numpy()
        dtype = array.dtype
        if getattr(dtype, "isnative", None) is not True:
            raise ValueError("non-native array dtype")
        values = array.tolist()
    except Exception as error:
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL {label} cannot be read as a native CPU vector"
        ) from error
    if shape == (width, 1):
        if not isinstance(values, list) or any(
            not isinstance(row, list) or len(row) != 1 for row in values
        ):
            raise vector._VectorContractViolation(
                "presence_masks", f"TorchRL {label} did not preserve a column vector"
            )
        values = [row[0] for row in values]
    if not isinstance(values, list) or len(values) != width:
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL {label} did not preserve batch width"
        )
    return tuple(values)


def _torchrl_column(tensordict: object, key: str, *, width: int) -> tuple[object, ...]:
    getter = getattr(tensordict, "get", None)
    if not callable(getter):
        raise vector._VectorContractViolation(
            "presence_masks", "TorchRL next TensorDict does not expose public get"
        )
    try:
        value = getter(key)
    except Exception as error:
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL next TensorDict cannot read {key}"
        ) from error
    if value is None:
        raise vector._VectorContractViolation(
            "presence_masks", f"TorchRL next TensorDict is missing {key}"
        )
    return _torchrl_row_values(value, width=width, label=key)


def _diagnostic_masked_sample(
    runtime: Mapping[str, object],
    budget: _DiagnosticSampleBudget,
    masks: Sequence[Sequence[bool]],
    *,
    width: int,
    action_count: int,
) -> tuple[int, ...]:
    torch = runtime["torch"]
    masked_categorical = runtime["MaskedCategorical"]

    def sample_once() -> tuple[int, ...]:
        try:
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device="cpu")  # type: ignore[union-attr]
            logits = torch.zeros((width, action_count), dtype=torch.float32, device="cpu")  # type: ignore[union-attr]
        except Exception as error:
            raise vector._VectorContractViolation(
                "mask_sampling", f"could not construct CPU [B,A] diagnostic tensors: {error}"
            ) from error
        if (
            tuple(mask_tensor.shape) != (width, action_count)
            or mask_tensor.dtype != torch.bool  # type: ignore[union-attr]
            or mask_tensor.device.type != "cpu"
            or tuple(logits.shape) != (width, action_count)
            or logits.device.type != "cpu"
        ):
            raise vector._VectorContractViolation(
                "mask_sampling", "diagnostic logits/mask must be CPU tensors with shape [B,A]"
            )
        try:
            mask_values = mask_tensor.numpy().tolist()
        except Exception as error:
            raise vector._VectorContractViolation(
                "mask_sampling", f"diagnostic mask cannot be inspected: {error}"
            ) from error
        if tuple(tuple(row) for row in mask_values) != tuple(tuple(row) for row in masks):
            raise vector._VectorContractViolation(
                "mask_sampling", "diagnostic boolean mask differs from public action-mask data"
            )
        try:
            with torch.random.fork_rng(devices=[]), torch.no_grad():  # type: ignore[union-attr]
                distribution = masked_categorical(logits=logits, mask=mask_tensor)  # type: ignore[operator]
                sampled = distribution.sample()
        except Exception as error:
            raise vector._VectorContractViolation(
                "mask_sampling", f"MaskedCategorical sampling failed: {type(error).__name__}: {error}"
            ) from error
        try:
            sampled_values = _torchrl_row_values(
                sampled, width=width, label="MaskedCategorical sample"
            )
            if any(type(value) is not int for value in sampled_values):
                raise ValueError("sampled actions are not built-in integer values")
            # Evaluate log_prob on the exact saved batch sample, not a new draw.
            with torch.no_grad():  # type: ignore[union-attr]
                log_prob = distribution.log_prob(sampled)
            log_prob_values = _torchrl_row_values(
                log_prob, width=width, label="MaskedCategorical log_prob"
            )
        except vector._VectorContractViolation as error:
            check = "mask_log_prob" if "log_prob" in str(error) else "mask_sampling"
            raise vector._VectorContractViolation(check, str(error)) from error
        except Exception as error:
            raise vector._VectorContractViolation(
                "mask_log_prob", f"MaskedCategorical log_prob failed: {type(error).__name__}: {error}"
            ) from error
        if any(type(value) not in (int, float) for value in log_prob_values):
            raise vector._VectorContractViolation(
                "mask_log_prob", "MaskedCategorical returned non-numeric log probabilities"
            )
        _validate_mask_diagnostic(
            masks,
            sampled_values,  # type: ignore[arg-type]
            log_prob_values,  # type: ignore[arg-type]
            width=width,
            action_count=action_count,
        )
        return tuple(sampled_values)  # type: ignore[return-value]

    return budget.sample(rows=width, sampler=sample_once)


def _same_observation_rows(
    expected: Sequence[Mapping[str, object]], actual: Sequence[Mapping[str, object]]
) -> bool:
    return len(expected) == len(actual) and all(
        dict(expected_row) == dict(actual_row)
        for expected_row, actual_row in zip(expected, actual, strict=True)
    )


def _captured_flags_and_rewards_match(
    tensordict: object, captured: _CapturedStep, *, width: int
) -> None:
    rewards = _torchrl_column(tensordict, "reward", width=width)
    terminated = _torchrl_column(tensordict, "terminated", width=width)
    truncated = _torchrl_column(tensordict, "truncated", width=width)
    done = _torchrl_column(tensordict, "done", width=width)
    for label, values in (
        ("terminated", terminated),
        ("truncated", truncated),
        ("done", done),
    ):
        if any(type(value) is not bool for value in values):
            raise vector._VectorContractViolation(
                "presence_masks", f"TorchRL {label} values must be built-in booleans"
            )
    if any(
        type(value) not in (int, float) for value in rewards
    ):
        raise vector._VectorContractViolation(
            "presence_masks", "TorchRL rewards must be finite built-in numeric values"
        )
    try:
        finite_rewards = all(math.isfinite(value) for value in rewards)
    except (OverflowError, TypeError, ValueError):
        finite_rewards = False
    if not finite_rewards:
        raise vector._VectorContractViolation(
            "presence_masks", "TorchRL rewards must be finite built-in numeric values"
        )
    for label, actual, expected in (
        ("reward", rewards, captured.rewards),
        ("terminated", terminated, captured.terminated),
        ("truncated", truncated, captured.truncated),
        ("done", done, captured.ended),
    ):
        if tuple(actual) != tuple(expected):
            raise vector._VectorContractViolation(
                "presence_masks", f"TorchRL {label} differs from the same-call public Gymnasium result"
            )


@dataclass(frozen=True)
class _RoutedCapturedStep:
    boundary_results: tuple[object, ...]
    next_states: tuple[object | None, ...]
    next_ids: tuple[object, ...]
    next_scenarios: tuple[object, ...]
    next_steps: tuple[object, ...]


def _route_captured_step(
    captured: _CapturedStep,
    *,
    schema: object,
    executions: Sequence[object],
    current_rows: Sequence[Mapping[str, object]],
    current_ids: Sequence[object],
    current_steps: Sequence[object],
    routes: Sequence[object],
    actions: Sequence[int],
    states_after_source: Sequence[object | None],
    policies: Sequence[object],
) -> _RoutedCapturedStep:
    """Route owned raw data before asking TorchRL to finish its step processing."""
    next_info = captured.current_info
    next_ids = next_info["episode_id"]
    next_scenarios = next_info["scenario_id"]
    next_steps = next_info["episode_steps"]
    for index, did_end in enumerate(captured.ended):
        vector._active_mask_and_candidates(
            captured.main_rows[index],
            schema,
            check="replacement_reset" if did_end else "old_final",
            slot=index,
        )
    vector.validate_episode_identity(
        source_ids=current_ids,
        source_steps=current_steps,
        current_ids=next_ids,
        current_steps=next_steps,
        ended=captured.ended,
        final_info=captured.final_info,
    )
    for index, execution in enumerate(executions):
        if next_scenarios[index] != execution.scenario_id:
            check = "replacement_reset" if captured.ended[index] else "old_final"
            raise vector._VectorContractViolation(
                check, f"slot {index} changed its public scenario_id"
            )
        if vector._decode_outcome(next_info["outcome"][index], schema) != "ongoing":
            check = "replacement_reset" if captured.ended[index] else "old_final"
            raise vector._VectorContractViolation(
                check, f"slot {index} main row is not an ongoing episode"
            )
        if captured.ended[index]:
            final_info = captured.final_info
            if final_info is None or captured.final_rows[index] is None:
                raise vector._VectorContractViolation(
                    "presence_masks", f"ended slot {index} lacks same-call old-final data"
                )
            if final_info["scenario_id"][index] != execution.scenario_id:
                raise vector._VectorContractViolation(
                    "old_final", f"old final scenario does not match slot {index}"
                )
            vector._decode_outcome(final_info["outcome"][index], schema)

    boundaries = vector.route_vector_step(
        routes=routes,
        sources=current_rows,
        actions=actions,
        rewards=captured.rewards,
        terminated=captured.terminated,
        truncated=captured.truncated,
        main_observations=captured.main_rows,
        final_observations=captured.final_rows,
        final_presence=captured.ended,
        final_info_presence=captured.ended,
        states_after_source=states_after_source,
        reset_state=lambda index: policies[index].init_state(),
    )
    next_states = []
    for index, (boundary, did_end) in enumerate(
        zip(boundaries, captured.ended, strict=True)
    ):
        expected_successor = (
            captured.final_rows[index] if did_end else captured.main_rows[index]
        )
        if boundary.transition.successor != expected_successor:
            raise vector._VectorContractViolation(
                "old_final", f"routed successor does not preserve slot {index} delivered data"
            )
        expected_replacement = captured.main_rows[index] if did_end else None
        if boundary.transition.replacement != expected_replacement:
            raise vector._VectorContractViolation(
                "replacement_reset", f"routed replacement differs from slot {index} main row"
            )
        if boundary.transition.state_after_source is not states_after_source[index]:
            raise vector._VectorContractViolation(
                "old_context", f"transition did not retain slot {index} source state"
            )
        if did_end:
            active_state = boundary.active_state
            old_state = states_after_source[index]
            if (
                not isinstance(active_state, vector._SentinelState)
                or active_state is old_state
                or active_state.identity is getattr(old_state, "identity", None)
            ):
                raise vector._VectorContractViolation(
                    "old_context", f"ended slot {index} did not receive fresh active state"
                )
        elif boundary.active_state is not states_after_source[index]:
            raise vector._VectorContractViolation(
                "old_context", f"continuing slot {index} did not retain source state"
            )
        next_states.append(boundary.active_state)
    return _RoutedCapturedStep(
        tuple(boundaries), tuple(next_states), tuple(next_ids),
        tuple(next_scenarios), tuple(next_steps),
    )


def _close_guarded_resources(
    wrapper_close: Callable[[], object] | None,
    facade_close: Callable[[], object],
    guard: _FacadeCallGuard,
) -> tuple[BaseException, ...]:
    """Attempt wrapper cleanup, then fallback only if the guarded close did not run."""
    errors: list[BaseException] = []
    if wrapper_close is not None:
        try:
            wrapper_close()
        except BaseException as error:
            errors.append(error)
    if not guard._close_attempted:
        try:
            facade_close()
        except BaseException as error:
            errors.append(error)
    return tuple(errors)


def _record_close_errors(
    errors: Sequence[BaseException],
    failures: list[object],
    *,
    primary: BaseException | None = None,
) -> tuple[BaseException | None, bool]:
    """Retain every ordinary close error and preserve primary/control failures."""
    if not errors:
        return None, False
    if primary is not None:
        control_errors = [error for error in errors if not isinstance(error, Exception)]
        ordinary_errors = [error for error in errors if isinstance(error, Exception)]
        if isinstance(primary, Exception) and control_errors:
            control = control_errors[0]
            try:
                control.add_note(
                    f"primary probe/writer failure also occurred: {type(primary).__name__}: {primary}"
                )
            except AttributeError:
                pass
            for error in errors:
                if error is control:
                    continue
                try:
                    control.add_note(
                        f"another environment close attempt failed: {type(error).__name__}: {error}"
                    )
                except AttributeError:
                    pass
            return control, bool(ordinary_errors)
        for error in errors:
            try:
                primary.add_note(
                    f"environment close also failed: {type(error).__name__}: {error}"
                )
            except AttributeError:
                pass
        return None, False

    ordinary_errors = [error for error in errors if isinstance(error, Exception)]
    control_errors = [error for error in errors if not isinstance(error, Exception)]
    for error in ordinary_errors:
        failures.append(
            vector.Failure(
                "operational_error", str(error), "close", type(error).__name__
            )
        )
    if not control_errors:
        return None, bool(ordinary_errors)

    control = control_errors[0]
    for error in errors:
        if error is control:
            continue
        try:
            control.add_note(
                f"another environment close attempt failed: {type(error).__name__}: {error}"
            )
        except AttributeError:
            pass
    return control, bool(ordinary_errors)


def run_torchrl_probe(
    config: ProbeConfig,
    writer: RunWriter,
    *,
    clock: Callable[[], float],
) -> vector.ProbeReport:
    """Collect bounded CPU TorchRL adapter evidence without closing the writer.

    Optional TorchRL imports, environment construction, and sampling occur only
    when this function is explicitly invoked. The writer remains caller-owned.
    """
    if not isinstance(config, ProbeConfig):
        raise ContractError("config must be a ProbeConfig")
    budget_clock = vector.Budget(config.max_wall_seconds, clock)
    vector.validate_probe_config(config)

    checks: dict[str, str] = {name: "blocked" for name in TORCHRL_CHECKS}
    checks.update(
        {
            "runtime": "blocked",
            "setup": "blocked",
            "guard": "blocked",
            "passive_registration": "blocked",
            "torchrl_wrapper": "blocked",
            "operational": "pass",
        }
    )
    failures: list[object] = []
    versions = _torchrl_versions()
    descriptor_values: dict[str, object] = {}
    build_info: Mapping[str, object] | None = None
    native_env = None
    facade = None
    torchrl_env = None
    guard = None
    guard_construction_counts = None
    runtime: Mapping[str, object] | None = None
    schema = None
    action_batch_contract = None
    diagnostic_budget = _DiagnosticSampleBudget(
        max_samples=config.num_envs * config.max_batch_steps  # type: ignore[attr-defined]
    )
    attempted_batch_steps = 0
    attempted_slot_decisions = 0
    attempted_replacement_decisions = 0
    wrapper_delivered_batch_steps = 0
    batch_steps = 0
    slot_decisions = 0
    replacement_decisions = 0
    operational_failure = False
    runtime_available = False
    adaptation_verified = False
    stop_reason = "not_started"
    phase = "runtime"
    writer_failure: BaseException | None = None
    writer_traceback = None
    process_control_failure: BaseException | None = None
    process_control_traceback = None
    elapsed_wall_seconds = None
    remaining_wall_seconds = None

    try:
        missing_distributions = tuple(
            name for name in ("spire_lab", "gymnasium", "numpy", "torch", "torchrl", "tensordict")
            if versions.get(name) is None
        )
        if missing_distributions:
            raise _TorchRLRuntimeUnavailable(
                "runtime distributions are missing: " + ", ".join(missing_distributions)
            )
        runtime = _load_torchrl_runtime()
        runtime_available = True
        checks["runtime"] = "pass"
        if budget_clock.exhausted():
            stop_reason = "wall_budget_before_construction"
        else:
            phase = "construction"
            native_env = vector.make_vector(
                config.num_envs, config.partition, config.workers  # type: ignore[attr-defined]
            )

            phase = "descriptor"
            descriptor = native_env.observation_spec()
            schema = vector.parse_descriptor(
                descriptor,
                allowed_schema_ids=vector.NATIVE_CONTRACT.schema_ids,
            )
            if schema.content_id not in vector.NATIVE_CONTRACT.content_ids:
                raise _TorchRLSetupFailure(f"unsupported native content_id: {schema.content_id}")
            descriptor_values = {
                "schema_id": schema.schema_id,
                "content_id": schema.content_id,
                "fingerprint": schema.fingerprint,
            }
            versions.update(descriptor_values)
            for execution in config.executions:  # type: ignore[attr-defined]
                if execution.expected_schema_id is not None and execution.expected_schema_id != schema.schema_id:
                    raise _TorchRLSetupFailure(
                        f"{execution.execution_id} expected schema {execution.expected_schema_id}, observed {schema.schema_id}"
                    )
                if execution.expected_fingerprint is not None and execution.expected_fingerprint != schema.fingerprint:
                    raise _TorchRLSetupFailure(
                        f"{execution.execution_id} expected fingerprint {execution.expected_fingerprint}, observed {schema.fingerprint}"
                    )
            checks["schema_expectations"] = "pass"

            action_space = native_env.single_action_space
            if getattr(action_space, "n", None) != schema.action_count or getattr(action_space, "start", None) != 0:
                raise _TorchRLSetupFailure(
                    "vector action space disagrees with the public observation descriptor"
                )
            build_info = vector.public_build_info()
            if build_info is None:
                raise _TorchRLSetupFailure(
                    "the already-loaded native binding did not provide public build_info"
                )
            checks["native_build"] = "pass"
            vector._writer_call(
                writer,  # type: ignore[arg-type]
                "update_manifest",
                {
                    "event": "torchrl_probe_discovery",
                    "schema": descriptor_values,
                    "reward": {
                        "contract_id": vector.NATIVE_CONTRACT.reward_contract_id,
                        "evidence": vector.NATIVE_CONTRACT.reward_evidence,
                    },
                    "python_version": versions["python"],
                    "distribution_versions": dict(versions),
                    "native_build_info": build_info,
                    "adapter": {
                        "GymWrapper": "public Gymnasium wrapper",
                        "categorical_action_encoding": True,
                        "frame_skip": 1,
                        "device": "cpu",
                        "passive_info_registration": True,
                    },
                },
            )

            phase = "wrapper_construction"
            guard = _FacadeCallGuard(
                batch_size=config.num_envs, require_native_actions=True  # type: ignore[attr-defined]
            )
            facade = _GuardedVectorFacade(native_env, guard, schema)
            gymnasium = runtime["gymnasium"]
            expected_autoreset = gymnasium.vector.AutoresetMode.SAME_STEP  # type: ignore[union-attr]
            if facade.autoreset_mode != expected_autoreset:
                raise _TorchRLSetupFailure(
                    "public vector metadata does not declare Gymnasium SAME_STEP autoreset"
                )
            wrapper_class = _passive_gym_wrapper_class(runtime["GymWrapper"])  # type: ignore[arg-type]
            torchrl_env = wrapper_class(
                facade,
                categorical_action_encoding=True,
                frame_skip=1,
                device=runtime["torch"].device("cpu"),  # type: ignore[union-attr]
                convert_actions_to_numpy=True,
            )
            counts_during_construction = guard.counts
            guard_construction_counts = dict(counts_during_construction)
            if guard.failed or counts_during_construction["reset_forwarded"] or counts_during_construction["step_forwarded"]:
                raise _TorchRLSetupFailure(
                    "TorchRL construction attempted guarded environment interaction"
                )
            guard.construction_complete()
            checks["guard"] = "pass"
            checks["passive_registration"] = "pass"
            if tuple(torchrl_env.batch_size) != (config.num_envs,):  # type: ignore[attr-defined]
                raise _TorchRLSetupFailure("GymWrapper batch size differs from the native vector width")
            checks["torchrl_wrapper"] = "pass"
            checks["setup"] = "pass"

            if budget_clock.exhausted():
                stop_reason = "wall_budget_before_reset"
            else:
                phase = "reset"
                seeds = [execution.environment_seed for execution in config.executions]  # type: ignore[attr-defined]
                scenarios = [execution.scenario_id for execution in config.executions]  # type: ignore[attr-defined]
                reset_options = {"scenario_ids": scenarios}
                guard.permit_initial_reset(seed=seeds, options=reset_options)
                current_tensordict = torchrl_env.reset(seed=seeds, options=reset_options)
                if guard.failed:
                    raise SetupBlocked(
                        "TorchRL reset returned after a guarded environment call was denied"
                    )
                if guard.counts["reset_delivered"] != 1 or guard.counts["reset_captured"] != 1:
                    raise _TorchRLSetupFailure(
                        "the controller reset did not produce exactly one captured native reset result"
                    )
                captured_reset = facade.last_reset
                if captured_reset is None:
                    raise vector._VectorContractViolation(
                        "observation_mapping", "guard did not capture the same-call reset result"
                    )
                mapped_rows = _mapped_observation_rows(
                    current_tensordict, schema, width=config.num_envs  # type: ignore[attr-defined]
                )
                if not _same_observation_rows(captured_reset.rows, mapped_rows):
                    raise vector._VectorContractViolation(
                        "observation_mapping", "GymWrapper reset observations differ from same-call public raw arrays"
                    )
                checks["observation_mapping"] = "pass"
                current_rows = captured_reset.rows
                current_info = captured_reset.current_info
                current_ids = current_info["episode_id"]
                current_scenarios = current_info["scenario_id"]
                current_steps = current_info["episode_steps"]
                for index, execution in enumerate(config.executions):  # type: ignore[attr-defined]
                    if current_scenarios[index] != execution.scenario_id or current_steps[index] != 0:
                        raise _TorchRLSetupFailure(
                            f"initial full reset did not start declared scenario in slot {index}"
                        )
                    if vector._decode_outcome(current_info["outcome"][index], schema) != "ongoing":
                        raise _TorchRLSetupFailure(
                            f"initial full reset did not return an ongoing slot {index}"
                        )
                    vector._active_mask_and_candidates(
                        current_rows[index], schema, check="presence_masks", slot=index
                    )

                policies = tuple(
                    vector._SentinelPolicy(execution.policy_seed)  # type: ignore[attr-defined]
                    for execution in config.executions  # type: ignore[attr-defined]
                )
                states: list[object | None] = [policy.init_state() for policy in policies]
                generations = [0] * config.num_envs  # type: ignore[attr-defined]
                decision_counts = [0] * config.num_envs  # type: ignore[attr-defined]
                fresh_state_pending: list[object | None] = [None] * config.num_envs  # type: ignore[attr-defined]
                ended_state_identity: list[object | None] = [None] * config.num_envs  # type: ignore[attr-defined]
                stop_reason = "batch_step_cap"

                while attempted_batch_steps < config.max_batch_steps:  # type: ignore[attr-defined]
                    if budget_clock.exhausted():
                        stop_reason = "wall_budget_before_step"
                        break
                    mapped_current_rows = _mapped_observation_rows(
                        current_tensordict, schema, width=config.num_envs  # type: ignore[attr-defined]
                    )
                    if not _same_observation_rows(current_rows, mapped_current_rows):
                        raise vector._VectorContractViolation(
                            "observation_mapping", "current GymWrapper data changed from the verified public replacement rows"
                        )

                    routes = []
                    actions = []
                    states_after_source = []
                    source_states = list(states)
                    source_generations = list(generations)
                    fresh_source = [False] * config.num_envs  # type: ignore[attr-defined]
                    for index, execution in enumerate(config.executions):  # type: ignore[attr-defined]
                        if fresh_state_pending[index] is not None:
                            expected_state = fresh_state_pending[index]
                            if source_states[index] is not expected_state:
                                raise vector._VectorContractViolation(
                                    "old_context", f"replacement slot {index} did not receive its fresh state"
                                )
                            if getattr(source_states[index], "identity", None) is ended_state_identity[index]:
                                raise vector._VectorContractViolation(
                                    "old_context", f"replacement slot {index} reused old episode state"
                                )
                            fresh_source[index] = True
                        mask, candidates = vector._active_mask_and_candidates(
                            current_rows[index], schema, check="old_final", slot=index
                        )
                        public = vector.PublicDecision(
                            current_rows[index], candidates, mask, schema.fingerprint,
                            vector._ACTION_MAPPING_ID,
                        )
                        route = vector.Route(
                            execution.execution_id,
                            index,
                            vector._route_episode_id(current_ids[index]),
                            f"{execution.execution_id}:{index}:{decision_counts[index]}",
                            current_steps[index] == 0,
                        )
                        result = vector.resolve_decision(
                            vector.DecisionRequest(public, route), policies[index], source_states[index]
                        )
                        if type(result.choice.action) is not int:
                            raise vector._VectorContractViolation(
                                "presence_masks", f"policy returned non-built-in action in slot {index}"
                            )
                        routes.append(route)
                        actions.append(result.choice.action)
                        states_after_source.append(result.choice.next_state)
                        decision_counts[index] += 1

                    masks = tuple(
                        tuple(value == 1 for value in row["action_mask"].values)
                        for row in mapped_current_rows
                    )
                    _diagnostic_masked_sample(
                        runtime,
                        diagnostic_budget,
                        masks,
                        width=config.num_envs,  # type: ignore[attr-defined]
                        action_count=schema.action_count,
                    )
                    checks["mask_sampling"] = "pass"
                    checks["mask_log_prob"] = "pass"
                    if budget_clock.exhausted():
                        stop_reason = "wall_budget_before_step"
                        break
                    action_tensor = runtime["torch"].as_tensor(  # type: ignore[union-attr]
                        actions, dtype=runtime["torch"].int64, device="cpu"  # type: ignore[union-attr]
                    )
                    current_tensordict.set("action", action_tensor)
                    guard.permit_step(actions)
                    attempted_batch_steps += 1
                    attempted_slot_decisions += config.num_envs  # type: ignore[attr-defined]
                    attempted_replacement_decisions += sum(
                        generation > 0 for generation in source_generations
                    )
                    phase = "step"
                    previous_native_deliveries = guard.counts["step_delivered"]
                    previous_denied_calls = guard.counts["denied_calls"]
                    facade.last_step = None
                    facade.last_step_raw_result = None
                    step_output = None
                    wrapper_step_error = None
                    try:
                        step_output = torchrl_env.step(current_tensordict)
                    except BaseException as error:
                        wrapper_step_error = error
                    native_deliveries = (
                        guard.counts["step_delivered"] - previous_native_deliveries
                    )
                    if native_deliveries not in (0, 1):
                        raise _TorchRLSetupFailure(
                            "one controller permit produced more than one native step return"
                        )
                    if native_deliveries == 0:
                        _raise_step_error_or_denial(
                            guard,
                            denied_calls_before=previous_denied_calls,
                            wrapper_error=wrapper_step_error,
                        )
                        raise _TorchRLSetupFailure(
                            "GymWrapper returned without one accepted native step result"
                        )

                    # A native return is a delivered batch even if TorchRL's
                    # subsequent conversion/reader/transform raises. Preserve
                    # its counts and owned transitions before examining that
                    # wrapper result or propagating its error.
                    batch_steps += 1
                    slot_decisions += config.num_envs  # type: ignore[attr-defined]
                    replacement_delivered = sum(generation > 0 for generation in source_generations)
                    replacement_decisions += replacement_delivered
                    action_batch_contract = {
                        "dtype": "int64",
                        "native_endian": True,
                        "shape": [config.num_envs],  # type: ignore[attr-defined]
                    }
                    captured_step = facade.last_step
                    if captured_step is None:
                        _raise_step_error_or_denial(
                            guard,
                            denied_calls_before=previous_denied_calls,
                            wrapper_error=wrapper_step_error,
                        )
                        raise vector._VectorContractViolation(
                            "observation_mapping", "guard did not capture the same-call raw step result"
                        )
                    try:
                        routed = _route_captured_step(
                            captured_step,
                            schema=schema,
                            executions=config.executions,
                            current_rows=current_rows,
                            current_ids=current_ids,
                            current_steps=current_steps,
                            routes=routes,
                            actions=actions,
                            states_after_source=states_after_source,
                            policies=policies,
                        )
                    except BaseException as route_error:
                        if wrapper_step_error is not None:
                            try:
                                route_error.add_note(
                                    "TorchRL wrapper processing also failed after the native step returned: "
                                    f"{type(wrapper_step_error).__name__}: {wrapper_step_error}"
                                )
                            except AttributeError:
                                pass
                        raise
                    terminated = captured_step.terminated
                    truncated = captured_step.truncated
                    ended = captured_step.ended
                    rewards = captured_step.rewards
                    next_info = captured_step.current_info
                    next_ids = routed.next_ids
                    next_scenarios = routed.next_scenarios
                    next_steps = routed.next_steps
                    main_rows = captured_step.main_rows
                    final_rows = captured_step.final_rows
                    boundary_results = routed.boundary_results
                    if any(ended) and not all(ended):
                        checks["mixed_rows"] = "pass"
                    if any(ended):
                        checks["presence_masks"] = "pass"
                        checks["old_final"] = "pass"
                    if any(fresh_source) and replacement_delivered > 0:
                        checks["old_context"] = "pass"
                    try:
                        for boundary in boundary_results:
                            vector._writer_call(
                                writer, "write_transition", boundary.transition
                            )  # type: ignore[arg-type]
                    except vector._WriterFailure as error:
                        if wrapper_step_error is not None:
                            try:
                                error.original.add_note(
                                    "TorchRL wrapper processing also failed after the native step returned: "
                                    f"{type(wrapper_step_error).__name__}: {wrapper_step_error}"
                                )
                            except AttributeError:
                                pass
                        raise

                    states = list(routed.next_states)
                    for index, did_end in enumerate(ended):
                        if did_end:
                            generations[index] += 1
                            fresh_state_pending[index] = boundary_results[index].active_state
                            ended_state_identity[index] = getattr(
                                states_after_source[index], "identity", None
                            )
                        elif fresh_source[index]:
                            fresh_state_pending[index] = None
                            ended_state_identity[index] = None
                    _raise_step_error_or_denial(
                        guard,
                        denied_calls_before=previous_denied_calls,
                        wrapper_error=wrapper_step_error,
                    )
                    wrapper_delivered_batch_steps += 1
                    try:
                        next_tensordict = step_output.get("next")
                    except Exception as error:
                        raise vector._VectorContractViolation(
                            "observation_mapping", f"GymWrapper step result has no public next TensorDict: {error}"
                        ) from error
                    if next_tensordict is None:
                        raise vector._VectorContractViolation(
                            "observation_mapping", "GymWrapper step result has no public next TensorDict"
                        )
                    mapped_next_rows = _mapped_observation_rows(
                        next_tensordict, schema, width=config.num_envs  # type: ignore[attr-defined]
                    )
                    _captured_flags_and_rewards_match(
                        next_tensordict, captured_step, width=config.num_envs  # type: ignore[attr-defined]
                    )
                    expected_next_rows = tuple(
                        captured_step.final_rows[index]
                        if captured_step.ended[index]
                        else captured_step.main_rows[index]
                        for index in range(config.num_envs)  # type: ignore[attr-defined]
                    )
                    if any(row is None for row in expected_next_rows):
                        raise vector._VectorContractViolation(
                            "presence_masks", "ended row has no independently captured same-call final observation"
                        )
                    if not _same_observation_rows(expected_next_rows, mapped_next_rows):  # type: ignore[arg-type]
                        checks["observation_mapping"] = "fail"
                        check = "old_final" if any(captured_step.ended) else "observation_mapping"
                        raise vector._VectorContractViolation(
                            check,
                            "GymWrapper next observations differ from same-call old-final/continuing raw arrays",
                        )
                    checks["observation_mapping"] = "pass"

                    if not any(ended):
                        current_tensordict = runtime["step_mdp"](step_output)  # type: ignore[operator]
                        current_rows = main_rows
                        current_info = next_info
                        current_ids = next_ids
                        current_scenarios = next_scenarios
                        current_steps = next_steps
                        if (
                            all(checks.get(name) == "pass" for name in TORCHRL_CHECKS)
                            and replacement_decisions > 0
                        ):
                            stop_reason = "required_evidence_complete"
                            break
                        continue

                    if all(ended):
                        if (
                            all(checks.get(name) == "pass" for name in TORCHRL_CHECKS)
                            and replacement_decisions > 0
                        ):
                            stop_reason = "required_evidence_complete"
                            break
                        stop_reason = "all_rows_ended_before_cached_replacement_check"
                        break

                    # step_mdp makes the real step output the root state;
                    # maybe_reset then asks TorchRL's SAME_STEP transform for
                    # its cached replacement rows. The facade guard allows no
                    # second native reset.
                    root_after_step = runtime["step_mdp"](step_output)  # type: ignore[operator]
                    current_tensordict = torchrl_env.maybe_reset(root_after_step)
                    if guard.failed or guard.counts["reset_forwarded"] != 1:
                        raise _TorchRLSetupFailure(
                            "TorchRL cached replacement reset was denied or forwarded to the native vector"
                        )
                    mapped_replacements = _mapped_observation_rows(
                        current_tensordict, schema, width=config.num_envs  # type: ignore[attr-defined]
                    )
                    if not _same_observation_rows(main_rows, mapped_replacements):
                        checks["observation_mapping"] = "fail"
                        raise vector._VectorContractViolation(
                            "replacement_reset",
                            "TorchRL cached reset did not return the same-call public replacement rows",
                        )
                    if guard.counts["reset_attempts"] != 1 or guard.counts["step_forwarded"] != batch_steps:
                        raise _TorchRLSetupFailure(
                            "TorchRL cached replacement path changed guarded native call counts"
                        )
                    checks["replacement_reset"] = "pass"
                    adaptation_verified = True
                    current_rows = main_rows
                    current_info = next_info
                    current_ids = next_ids
                    current_scenarios = next_scenarios
                    current_steps = next_steps

                    if (
                        all(checks.get(name) == "pass" for name in TORCHRL_CHECKS)
                        and replacement_decisions > 0
                    ):
                        stop_reason = "required_evidence_complete"
                        break
    except _TorchRLRuntimeUnavailable as error:
        checks["runtime"] = "blocked"
        failures.append(vector.Failure("setup_error", str(error), phase, type(error).__name__))
        stop_reason = "runtime_unavailable"
    except vector._WriterFailure as error:
        writer_failure = error.original
        writer_traceback = error.original.__traceback__
        stop_reason = "writer_io_error"
    except vector._VectorContractViolation as error:
        checks[error.check] = "fail"
        failures.append(
            vector.Failure("contract_violation", str(error), phase, type(error).__name__)
        )
        if error.check not in TORCHRL_CHECKS:
            operational_failure = True
            checks["operational"] = "fail"
        stop_reason = "contract_violation"
    except _TorchRLSetupFailure as error:
        operational_failure = True
        checks["setup"] = "fail"
        checks["operational"] = "fail"
        failures.append(vector.Failure("setup_error", str(error), phase, type(error).__name__))
        stop_reason = "setup_error"
    except SetupBlocked as error:
        operational_failure = True
        checks["guard"] = "fail"
        checks["setup"] = "fail"
        checks["operational"] = "fail"
        failures.append(vector.Failure("setup_error", str(error), phase, type(error).__name__))
        stop_reason = "guard_blocked"
    except Exception as error:
        failure = vector._exception_failure(error, phase=phase)
        failures.append(failure)
        checks["operational"] = "fail"
        operational_failure = True
        if failure.kind in {"invalid_action", "environment_state", "setup_error", "simulator_fault"}:
            checks["native_failure"] = "fail"
        stop_reason = "operational_error"
    except BaseException as error:
        process_control_failure = error
        process_control_traceback = error.__traceback__
    finally:
        if native_env is not None and guard is not None and facade is not None:
            primary_failure = process_control_failure or writer_failure
            close_errors = _close_guarded_resources(
                (lambda: torchrl_env.close()) if torchrl_env is not None else None,
                facade.close,
                guard,
            )
            control_failure, ordinary_close_failure = _record_close_errors(
                close_errors, failures, primary=primary_failure
            )
            if control_failure is not None:
                process_control_failure = control_failure
                process_control_traceback = control_failure.__traceback__
            if ordinary_close_failure:
                operational_failure = True
                checks["operational"] = "fail"
                if stop_reason == "required_evidence_complete":
                    stop_reason = "close_error_after_evidence"
        elif native_env is not None:
            close_errors = []
            try:
                native_env.close()
            except BaseException as error:
                close_errors.append(error)
            control_failure, ordinary_close_failure = _record_close_errors(
                close_errors,
                failures,
                primary=process_control_failure or writer_failure,
            )
            if control_failure is not None:
                process_control_failure = control_failure
                process_control_traceback = control_failure.__traceback__
            if ordinary_close_failure:
                operational_failure = True
                checks["operational"] = "fail"
                if stop_reason == "required_evidence_complete":
                    stop_reason = "close_error_after_evidence"

    if process_control_failure is not None:
        vector._note_prior_failures(process_control_failure, failures)
        raise process_control_failure.with_traceback(process_control_traceback)
    if writer_failure is not None:
        vector._note_prior_failures(writer_failure, failures)
        raise writer_failure.with_traceback(writer_traceback)

    try:
        elapsed_wall_seconds = budget_clock.elapsed()
        remaining_wall_seconds = max(
            0.0, float(config.max_wall_seconds) - elapsed_wall_seconds  # type: ignore[attr-defined]
        )
    except Exception as error:
        operational_failure = True
        checks["operational"] = "fail"
        failures.append(
            vector.Failure("operational_error", str(error), "budget", type(error).__name__)
        )
        if stop_reason == "not_started":
            stop_reason = "budget_error"

    verdict = _torchrl_probe_verdict(
        checks,
        runtime_available=runtime_available,
        operational_failure=operational_failure,
        adaptation_verified=adaptation_verified,
    )
    report = vector.ProbeReport(
        verdict=verdict,
        checks=checks,
        versions=versions,
        batch_steps=batch_steps,
        slot_decisions=slot_decisions,
        replacement_decisions=replacement_decisions,
        failures=tuple(failures),
    )
    encoded_report = vector.probe_report_to_json(report)
    guard_counts = dict(guard.counts) if guard is not None else {
        "reset_attempts": 0,
        "reset_forwarded": 0,
        "reset_delivered": 0,
        "reset_captured": 0,
        "step_attempts": 0,
        "step_forwarded": 0,
        "step_delivered": 0,
        "step_captured": 0,
        "denied_calls": 0,
        "close_attempts": 0,
        "close_forwarded": 0,
    }
    summary = {
        "schema_id": "spire-learn-torchrl-probe-summary-v1",
        "report": encoded_report,
        "stop_reason": stop_reason,
        "budgets": {
            "max_batch_steps": config.max_batch_steps,  # type: ignore[attr-defined]
            "max_wall_seconds": float(config.max_wall_seconds),  # type: ignore[attr-defined]
            "remaining_wall_seconds": remaining_wall_seconds,
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "overshoot_wall_seconds": (
                None if elapsed_wall_seconds is None
                else max(0.0, elapsed_wall_seconds - float(config.max_wall_seconds))  # type: ignore[attr-defined]
            ),
            "diagnostic_sample_limit": diagnostic_budget.max_samples,
            "diagnostic_samples_reserved": diagnostic_budget.samples_reserved,
        },
        "attempted": {
            "batch_steps": attempted_batch_steps,
            "slot_decisions": attempted_slot_decisions,
            "replacement_decisions": attempted_replacement_decisions,
        },
        "delivered": {
            "batch_steps": batch_steps,
            "slot_decisions": slot_decisions,
            "replacement_decisions": replacement_decisions,
            "wrapper_batch_steps": wrapper_delivered_batch_steps,
        },
        "call_counts": {
            "controller_attempted_steps": attempted_batch_steps,
            "native_forwarded_steps": guard_counts["step_forwarded"],
            "native_delivered_steps": guard_counts["step_delivered"],
            "native_captured_steps": guard_counts["step_captured"],
            "wrapper_delivered_steps": wrapper_delivered_batch_steps,
        },
        "schema": descriptor_values,
        "distribution_versions": dict(versions),
        "native_build_info": build_info,
        "action_batch_contract": action_batch_contract,
        "adapter": {
            "library": "torchrl.envs.GymWrapper",
            "categorical_action_encoding": True,
            "frame_skip": 1,
            "device": "cpu",
            "passive_info_registration": checks["passive_registration"] == "pass",
            "adaptation_verified": adaptation_verified,
            "guard_counts": guard_counts,
            "construction_guard_counts": guard_construction_counts,
            "last_step_raw_result_retained": (
                facade is not None and facade.last_step_raw_result is not None
            ),
            "last_step_snapshot_retained": (
                facade is not None and facade.last_step is not None
            ),
            "diagnostic_rng": "Torch CPU RNG state forked/restored; samples are not actor actions",
        },
        "reward": {
            "contract_id": vector.NATIVE_CONTRACT.reward_contract_id,
            "evidence": vector.NATIVE_CONTRACT.reward_evidence,
        },
        "rng_001": vector._RNG_001,
        "time_001": vector._TIME_001,
    }
    try:
        vector._writer_call(
            writer,  # type: ignore[arg-type]
            "update_manifest",
            {
                "event": "torchrl_probe_finished",
                "stop_reason": stop_reason,
                "attempted": summary["attempted"],
                "delivered": summary["delivered"],
                "guard_counts": guard_counts,
                "diagnostic_samples_reserved": diagnostic_budget.samples_reserved,
            },
        )
        vector._writer_call(writer, "write_metric", encoded_report)  # type: ignore[arg-type]
        vector._writer_call(writer, "finalize", summary)  # type: ignore[arg-type]
    except vector._WriterFailure as error:
        vector._note_prior_failures(error.original, failures)
        raise error.original.with_traceback(error.original.__traceback__)
    return report
