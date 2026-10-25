"""Lazy, public-protocol reader for Dict-valued terminal observations."""

from collections.abc import Mapping, Sequence

from ..contracts import ArrayValue, ContractError
from ..envs.native import native_snapshot


DICT_FINAL_ADAPTER_ID = "spire-learn-dict-final-reader-v1"


def _array_values(value: object, *, width: int, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ContractError(f"{label} must be a full-width vector")
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            if tuple(shape) != (width,):
                raise ContractError(f"{label} must have shape ({width},)")
        except TypeError as error:
            raise ContractError(f"{label} has an invalid shape") from error
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            values = tolist()
        except Exception as error:
            raise ContractError(f"{label} cannot be read as a vector") from error
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ContractError(f"{label} must be a sized vector")
    if not isinstance(values, (list, tuple)) or len(values) != width:
        raise ContractError(f"{label} must contain exactly {width} rows")
    return tuple(values)


def _bool_vector(value: object, *, width: int, label: str) -> tuple[bool, ...]:
    dtype = getattr(value, "dtype", None)
    if dtype is not None and (
        getattr(dtype, "name", None) != "bool"
        or getattr(dtype, "isnative", None) is not True
    ):
        raise ContractError(f"{label} must use native bool dtype")
    values = _array_values(value, width=width, label=label)
    if any(type(item) is not bool for item in values):
        raise ContractError(f"{label} must contain only booleans")
    return values  # type: ignore[return-value]


def prepare_final_rows(
    info_dict: object, ended: Sequence[bool], schema: object
) -> tuple[Mapping[str, ArrayValue] | None, ...]:
    """Validate and own all ended Dict leaves before a TensorDict is changed."""
    if not isinstance(info_dict, Mapping):
        raise ContractError("info_dict must be a mapping")
    try:
        ended_rows = tuple(ended)
    except TypeError as error:
        raise ContractError("done must be a full-width boolean vector") from error
    if not ended_rows or any(type(value) is not bool for value in ended_rows):
        raise ContractError("done must be a non-empty boolean vector")
    specs = tuple(getattr(schema, "tensors", ()))
    if not specs:
        raise ContractError("schema must declare public observation tensors")
    expected_keys = tuple(getattr(spec, "key", None) for spec in specs)
    if any(not isinstance(key, str) for key in expected_keys) or len(set(expected_keys)) != len(expected_keys):
        raise ContractError("schema observation tensor keys must be unique strings")

    has_rows = "final_obs" in info_dict
    has_mask = "_final_obs" in info_dict
    if has_rows != has_mask:
        raise ContractError("final_obs and _final_obs must be present together")
    if not has_rows:
        if any(ended_rows):
            raise ContractError("ended rows are missing final_obs and _final_obs")
        return (None,) * len(ended_rows)

    presence = _bool_vector(
        info_dict["_final_obs"], width=len(ended_rows), label="_final_obs"
    )
    if presence != ended_rows:
        raise ContractError("final_obs presence mask must exactly match TorchRL done")
    rows = _array_values(
        info_dict["final_obs"], width=len(ended_rows), label="final_obs"
    )
    row_snapshots: list[Mapping[str, ArrayValue] | None] = []
    for index, (row, did_end) in enumerate(zip(rows, ended_rows, strict=True)):
        if not did_end:
            if row is not None:
                raise ContractError(f"live row {index} must not contain final_obs data")
            row_snapshots.append(None)
            continue
        if not isinstance(row, Mapping):
            raise ContractError(f"ended row {index} has no public final observation")
        if set(row) != set(expected_keys):
            raise ContractError(f"ended row {index} keys differ from the public descriptor")
        row_snapshots.append(native_snapshot(row, schema))

    return tuple(row_snapshots)  # type: ignore[return-value]


def _done_values(tensordict: object, torch: object) -> tuple[bool, ...] | None:
    getter = getattr(tensordict, "get", None)
    if not callable(getter):
        raise ContractError("TensorDict must expose its public get method")
    try:
        done = getter("done")
    except Exception as error:
        raise ContractError("TensorDict done mask cannot be read") from error
    if done is None:
        return None
    shape = getattr(done, "shape", None)
    try:
        shape = tuple(shape)
    except TypeError as error:
        raise ContractError("TensorDict done must be a CPU boolean tensor") from error
    batch_size = getattr(tensordict, "batch_size", None)
    try:
        batch_shape = tuple(batch_size)
    except TypeError as error:
        raise ContractError("TensorDict has no public batch size") from error
    if len(batch_shape) != 1 or type(batch_shape[0]) is not int or batch_shape[0] <= 0:
        raise ContractError("Dict-final reader requires a one-dimensional vector batch")
    width = batch_shape[0]
    if shape not in {(width,), (width, 1)}:
        raise ContractError("TensorDict done must contain exactly one value per row")
    device = getattr(done, "device", None)
    if device is None or getattr(device, "type", None) != "cpu":
        raise ContractError("TensorDict done must be a public CPU tensor")
    detach = getattr(done, "detach", None)
    value = detach() if callable(detach) else done
    cpu = getattr(value, "cpu", None)
    value = cpu() if callable(cpu) else value
    numpy = getattr(value, "numpy", None)
    if not callable(numpy):
        raise ContractError("TensorDict done cannot be read as a CPU boolean tensor")
    try:
        array = numpy()
    except Exception as error:
        raise ContractError("TensorDict done cannot be read as a CPU boolean tensor") from error
    dtype = getattr(array, "dtype", None)
    if (
        getattr(dtype, "name", None) != "bool"
        or getattr(dtype, "isnative", None) is not True
    ):
        raise ContractError("TensorDict done must use native bool dtype")
    if shape == (width, 1):
        raw_values = getattr(array, "tolist", lambda: None)()
        if not isinstance(raw_values, list) or any(
            not isinstance(row, list) or len(row) != 1 for row in raw_values
        ):
            raise ContractError("TensorDict done did not preserve its column shape")
        raw_values = [row[0] for row in raw_values]
        values = tuple(raw_values)
    else:
        values = _array_values(array, width=width, label="done")
    if any(type(item) is not bool for item in values):
        raise ContractError("TensorDict done must contain only booleans")
    del torch  # its public API is supplied to keep the module optional-import free
    return values  # type: ignore[return-value]


def _zero_value(shape: tuple[int, ...]) -> object:
    if not shape:
        return 0
    return tuple(_zero_value(shape[1:]) for _ in range(shape[0]))


def _reshape_values(values: tuple[int | float, ...], shape: tuple[int, ...]) -> object:
    iterator = iter(values)

    def build(dimensions: tuple[int, ...]) -> object:
        if not dimensions:
            return next(iterator)
        return tuple(build(dimensions[1:]) for _ in range(dimensions[0]))

    return build(shape)


def _validate_tensor(tensor: object, *, dtype: object, shape: tuple[int, ...], key: str) -> None:
    if getattr(tensor, "dtype", None) != dtype:
        raise ContractError(f"prepared final tensor {key} has an unexpected dtype")
    device = getattr(tensor, "device", None)
    if device is None or getattr(device, "type", None) != "cpu":
        raise ContractError(f"prepared final tensor {key} is not on CPU")
    try:
        actual_shape = tuple(getattr(tensor, "shape", ()))
    except TypeError as error:
        raise ContractError(f"prepared final tensor {key} has an invalid shape") from error
    if actual_shape != shape:
        raise ContractError(f"prepared final tensor {key} has an unexpected shape")


def make_dict_final_reader(
    *,
    observation_spec: object,
    schema: object,
    torch: object,
    info_reader_type: type,
    composite_type: type,
) -> object:
    """Create an initialized reader without importing TorchRL or Torch here."""
    if not isinstance(info_reader_type, type) or not isinstance(composite_type, type):
        raise ContractError("public TorchRL reader and Composite types are required")
    keys = getattr(observation_spec, "keys", None)
    if not callable(keys):
        raise ContractError("known observation spec must expose public keys")
    try:
        observation_keys = tuple(keys())
    except Exception as error:
        raise ContractError("known observation-spec keys cannot be read") from error
    schema_keys = tuple(spec.key for spec in getattr(schema, "tensors", ()))
    if set(observation_keys) != set(schema_keys):
        raise ContractError("known observation spec keys differ from the public schema")
    try:
        batch_shape = tuple(observation_spec.shape)
    except (AttributeError, TypeError) as error:
        raise ContractError("known observation spec has no public batch shape") from error
    if len(batch_shape) != 1 or type(batch_shape[0]) is not int or batch_shape[0] <= 0:
        raise ContractError("known observation spec must use one positive vector batch axis")
    bound_batch_shape = batch_shape
    for spec in schema.tensors:
        try:
            actual = observation_spec[spec.key]
            actual_shape = tuple(actual.shape)
        except (KeyError, AttributeError, TypeError) as error:
            raise ContractError(f"known observation spec leaf {spec.key} is unreadable") from error
        actual_dtype = getattr(actual, "dtype", None)
        dtype_name = getattr(actual_dtype, "name", None)
        if dtype_name is None:
            dtype_name = str(actual_dtype).rsplit(".", 1)[-1]
        expected_shape = tuple(spec.shape)
        accepted_shapes = {expected_shape, (batch_shape[0], *expected_shape)}
        if dtype_name != spec.dtype or actual_shape not in accepted_shapes:
            raise ContractError(
                f"known observation spec leaf {spec.key} dtype/shape differs from the public schema"
            )
        device = getattr(actual, "device", None)
        if device is not None and getattr(device, "type", None) != "cpu":
            raise ContractError(f"known observation spec leaf {spec.key} is not on CPU")
    clone = getattr(observation_spec, "clone", None)
    final_observation_spec = clone() if callable(clone) else observation_spec
    info_spec = composite_type({"final": final_observation_spec})

    class DictFinalInfoReader(info_reader_type):
        adapter_id = DICT_FINAL_ADAPTER_ID

        def __init__(self):
            self._info_spec = info_spec

        @property
        def info_spec(self):
            return self._info_spec

        def __call__(self, info_dict, tensordict):
            try:
                call_batch_shape = tuple(tensordict.batch_size)
            except (AttributeError, TypeError) as error:
                raise ContractError("TensorDict has no public batch shape") from error
            if call_batch_shape != bound_batch_shape:
                raise ContractError(
                    "TensorDict batch shape differs from the bound observation spec"
                )
            ended = _done_values(tensordict, torch)
            if ended is None:
                if "final_obs" in info_dict or "_final_obs" in info_dict:
                    raise ContractError("final observations require an actual TorchRL done mask")
                return tensordict

            prepared = prepare_final_rows(info_dict, ended, schema)
            if not any(row is not None for row in prepared):
                return tensordict

            tensor_factory = getattr(torch, "tensor", None)
            if not callable(tensor_factory):
                raise ContractError("Torch must expose its public tensor constructor")
            output: dict[str, object] = {}
            for spec in schema.tensors:
                dtype = getattr(torch, spec.dtype, None)
                if dtype is None:
                    raise ContractError(f"Torch does not expose dtype {spec.dtype}")
                rows = tuple(
                    _zero_value(tuple(spec.shape))
                    if row is None
                    else _reshape_values(row[spec.key].values, tuple(spec.shape))
                    for row in prepared
                )
                try:
                    tensor = tensor_factory(rows, dtype=dtype, device="cpu")
                except Exception as error:
                    raise ContractError(
                        f"could not materialize validated final observation {spec.key}"
                    ) from error
                _validate_tensor(
                    tensor,
                    dtype=dtype,
                    shape=(len(ended), *tuple(spec.shape)),
                    key=spec.key,
                )
                output[spec.key] = tensor

            setter = getattr(tensordict, "set", None)
            if not callable(setter):
                raise ContractError("TensorDict must expose its public set method")
            for key, tensor in output.items():
                setter(("final", key), tensor)
            return tensordict

    return DictFinalInfoReader()
