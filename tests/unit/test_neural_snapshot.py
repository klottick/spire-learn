"""Owned runtime snapshot tests with small learner-authored schemas."""

from __future__ import annotations

import unittest
from types import SimpleNamespace


def _api(test: unittest.TestCase) -> SimpleNamespace:
    try:
        from spire_learn.neural import contracts as neural_contracts
        from spire_learn.neural import schema
        from spire_learn.neural.contracts import BufferArray, CombatFrame
        from spire_learn.neural.snapshot import snapshot_common
    except ImportError as error:
        test.fail(f"neural snapshot API is unavailable: {error}")
    from tests.unit.test_neural_schema import _frame, _synthetic_schema

    return SimpleNamespace(
        schema=schema,
        neural_contracts=neural_contracts,
        BufferArray=BufferArray,
        CombatFrame=CombatFrame,
        snapshot_common=snapshot_common,
        frame=_frame,
        synthetic_schema=_synthetic_schema,
    )


class _DType:
    def __init__(self, dtype: str, *, native: bool = True):
        self.name = dtype
        self.kind = "i"
        self.itemsize = 1 if dtype == "int8" else 8
        self.isnative = native


class _Flags:
    def __init__(self, c_contiguous: bool = True):
        self.c_contiguous = c_contiguous


class _Array:
    def __init__(self, dtype: str, shape: tuple[int, ...], data: bytearray,
                 *, native: bool = True, c_contiguous: bool = True):
        self.dtype = _DType(dtype, native=native)
        self.shape = shape
        self.ndim = len(shape)
        self.flags = _Flags(c_contiguous)
        self._data = data

    def tobytes(self, order="C"):
        if order != "C":
            raise ValueError("only C order is supported")
        return bytes(self._data)


def _arrays(frame):
    output = {}
    for name, item in frame.tensors.items():
        output[name] = _Array(item.dtype, item.shape, bytearray(item.data))
    return output


class NeuralSnapshotTests(unittest.TestCase):
    def test_mutated_source_cannot_change_snapshot(self):
        api = _api(self)
        with api.synthetic_schema(api) as schema:
            fixture = api.frame(api, schema)
            raw = _arrays(fixture)
            owned = api.snapshot_common(raw, schema)
            before = owned.tensors["player"].data
            raw["player"]._data[0] ^= 0x01
            self.assertEqual(owned.tensors["player"].data, before)
            self.assertIs(type(owned.tensors["player"].data), bytes)

    def test_metadata_is_not_actor_input(self):
        api = _api(self)
        with api.synthetic_schema(api) as schema:
            fixture = api.frame(api, schema)
            raw = _arrays(fixture)
            owned = api.snapshot_common(raw, schema)
            self.assertNotIn("token", owned.tensors)
            self.assertNotIn("active_visit", owned.tensors)
            self.assertNotIn("effective_seed", owned.tensors)
            raw["effective_seed"] = 17
            with self.assertRaises(api.schema.ContractError):
                api.snapshot_common(raw, schema)

    def test_snapshot_rejects_wrong_shape_dtype_endian_and_layout(self):
        api = _api(self)
        with api.synthetic_schema(api) as schema:
            fixture = api.frame(api, schema)
            cases = (
                ("player", _Array("int64", (2,), bytearray(16))),
                ("player", _Array("int8", (1,), bytearray(1))),
                ("player", _Array("int64", (1,), bytearray(8), native=False)),
                ("player", _Array("int64", (1,), bytearray(8), c_contiguous=False)),
            )
            for name, invalid in cases:
                raw = _arrays(fixture)
                raw[name] = invalid
                with self.subTest(name=name, dtype=invalid.dtype.name), self.assertRaises(
                    api.schema.ContractError
                ):
                    api.snapshot_common(raw, schema)


if __name__ == "__main__":
    unittest.main()
