"""Bridge a dense Standard combat view to the compact preset layout (numpy; runtime container only).

The Run's combat projection (``RunEnv.combat_view()``) is the dense Standard combat view; Gym
rejects compact selection there. The Gym documents the two layouts as exact: casting a compact
observation's values to ``int64`` and zero-padding its rows to the dense capacities gives the dense
observation of the same boundary, byte for byte. ``CompactBridge`` runs that the other way. It
trims each row group to the compact descriptor's capacity and casts each array to the compact dtype.
It never drops a public fact: a trimmed row must be pure padding (value 0, known 0, occupied 0),
so an occupied row past a capacity raises, and so does a known empty ``potions`` or ``orbs`` slot
past one. Every kept value must lie inside the compact column's ``low``/``high``, and every legal
action must fit the compact action width. A violation raises ``BridgeError``.

The bridge returns both forms: the numpy dict that ``search_view.View`` and the teacher read, and
a validated ``CombatFrame`` for ``CheckpointPlayer``. numpy is imported lazily.
"""

from __future__ import annotations

from collections.abc import Mapping


class BridgeError(ValueError):
    """A dense combat view does not fit the compact layout (a row past capacity or a value out of range)."""


class CompactBridge:
    """Dense Standard combat view to compact preset arrays, checked against both descriptors."""

    def __init__(self, dense_descriptor: Mapping, compact_descriptor: Mapping, schema=None):
        import numpy as np

        self._np = np
        dense = {tensor["key"]: tensor for tensor in dense_descriptor["tensors"]}
        compact = {tensor["key"]: tensor for tensor in compact_descriptor["tensors"]}
        if set(dense) != set(compact):
            raise BridgeError("dense and compact combat views name different tensors")
        if dense_descriptor["vocabularies"] != compact_descriptor["vocabularies"]:
            raise BridgeError("dense and compact combat views have different vocabularies")
        self.schema = schema
        self.compact_fingerprint = compact_descriptor["fingerprint"]
        self.dense_fingerprint = dense_descriptor["fingerprint"]
        self._plan = {}
        for key, item in compact.items():
            wide = dense[key]
            shape, dense_shape = tuple(item["shape"]), tuple(wide["shape"])
            if len(shape) != len(dense_shape) or shape[1:] != dense_shape[1:] or (shape and shape[0] > dense_shape[0]):
                raise BridgeError(f"{key} compact shape {list(shape)} does not trim dense shape {list(dense_shape)}")
            if wide.get("fields") != item.get("fields"):
                raise BridgeError(f"{key} dense and compact columns differ")
            low = np.asarray(item["low"], dtype=np.int64)
            high = np.asarray(item["high"], dtype=np.int64)
            self._plan[key] = (np.dtype(item["dtype"]), shape, dense_shape, low, high)

    def arrays(self, dense: Mapping) -> dict:
        """The compact numpy dict of one dense combat view; raises ``BridgeError`` on any loss."""
        np = self._np
        if set(dense) != set(self._plan):
            raise BridgeError("dense observation tensors differ from the descriptor")
        result = {}
        for key, (dtype, shape, dense_shape, low, high) in self._plan.items():
            array = np.asarray(dense[key])
            if array.shape != dense_shape:
                raise BridgeError(f"{key} has shape {list(array.shape)}, expected {list(dense_shape)}")
            if shape and shape[0] < dense_shape[0]:
                rest = array[shape[0]:]
                if rest.any():
                    rows = sorted({int(row) for row in np.nonzero(rest)[0]})
                    raise BridgeError(f"{key} holds data past its compact capacity {shape[0]} "
                                      f"(dense rows {[shape[0] + row for row in rows[:4]]})")
                array = array[:shape[0]]
            if array.size and (np.any(array < low) or np.any(array > high)):
                raise BridgeError(f"{key} holds a value outside its compact range")
            result[key] = np.ascontiguousarray(array.astype(dtype, copy=True))
        return result

    def __call__(self, dense: Mapping):
        """``(compact numpy dict, CombatFrame)``; the frame needs the bridge's compact ``schema``."""
        if self.schema is None:
            raise BridgeError("the bridge has no compact schema to build a CombatFrame")
        from spire_learn.neural.snapshot import snapshot_common

        arrays = self.arrays(dense)
        return arrays, snapshot_common(arrays, self.schema)


def widen(compact: Mapping, dense_descriptor: Mapping) -> dict:
    """The dense observation of a compact one, as the Gym documents it: int64 values, zero-padded rows."""
    import numpy as np

    result = {}
    for tensor in dense_descriptor["tensors"]:
        key = tensor["key"]
        array = np.asarray(compact[key])
        out = np.zeros(tuple(tensor["shape"]), dtype=np.dtype(tensor["dtype"]))
        out[tuple(slice(0, n) for n in array.shape)] = array
        result[key] = out
    return result


__all__ = ["BridgeError", "CompactBridge", "widen"]
