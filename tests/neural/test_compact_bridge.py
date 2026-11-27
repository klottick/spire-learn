"""The dense-to-compact combat bridge (``--compact-bridge`` route; runtime container only).

(a) Compact standalone frames from the pinned Gym, widened to dense as the Gym documents (int64 values,
zero-padded rows), bridge back to the same bytes; the widened frames equal a dense twin session's.
(b) Overflow: an occupied row or a known empty potion/orb slot past a capacity, a value outside the
compact range and a legal action past the compact width each raise ``BridgeError``.
numpy and the Gym are imported lazily, never at discovery.
"""

from __future__ import annotations

import copy
import random
import unittest

ENTRY = {
    "character": "ironclad", "ascension": 20, "gold": 99, "relic_counters": {},
    "deck": [{"card": card, "upgrades": 0, "misc": 0}
             for card in ["bash", "shrug_it_off", "pommel_strike", "inflame"] + ["defend"] * 4 + ["strike"] * 5],
}
ENCOUNTERS = (("cultist", ["burning_blood"], ["fire_potion", None]),
              ("gremlin_nob", ["burning_blood", "anchor"], ["block_potion", "weak_potion"]),
              ("jaw_worm", ["burning_blood", "potion_belt"], ["fire_potion", None, "swift_potion", None]))


def _modules():
    try:
        import numpy
    except ImportError:
        raise unittest.SkipTest("numpy is only importable in the runtime container")
    from spire_learn.neural import compact_bridge
    return numpy, compact_bridge


def _catalog():
    entries = []
    for encounter, relics, potions in ENCOUNTERS:
        entries.append({**copy.deepcopy(ENTRY), "id": encounter, "encounter": encounter, "hp": 60, "max_hp": 80,
                        "relics": relics, "potions": potions})
    return {"catalog_schema_id": "spire-gym-standard-combat-catalog-v1", "profile_id": "standard-combat-v1",
            "entries": entries}


class CompactBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np, cls.bridge_module = _modules()
        from spire_lab import _native
        from spire_lab.gymnasium import StandardCombatEnv
        from spire_learn.neural.gym_contracts import STANDARD_COMPACT_PRESET
        from spire_learn.neural.schema import parse_combat_descriptor

        catalog = _catalog()
        cls.compact_env = StandardCombatEnv(catalog, partition="selection", observation="compact",
                                            capacity_preset=STANDARD_COMPACT_PRESET)
        cls.dense_env = StandardCombatEnv(catalog, partition="selection")
        cls.compact_descriptor = cls.compact_env.observation_spec()
        cls.dense_descriptor = cls.dense_env.observation_spec()
        cls.schema = parse_combat_descriptor(cls.compact_descriptor, revision=_native.build_info()["revision"])
        cls.bridge = cls.bridge_module.CompactBridge(cls.dense_descriptor, cls.compact_descriptor, cls.schema)

    @classmethod
    def tearDownClass(cls):
        cls.compact_env.close()
        cls.dense_env.close()

    def frames(self, steps=12):
        """(compact, dense) observation pairs along a few seeded fights played identically in both sessions."""
        np = self.np
        for index, (encounter, _, _) in enumerate(ENCOUNTERS):
            rng = random.Random(index)
            compact, _ = self.compact_env.reset(seed=100 + index, options={"entry_id": encounter})
            dense, _ = self.dense_env.reset(seed=100 + index, options={"entry_id": encounter})
            for _ in range(steps):
                yield compact, dense
                legal = [int(i) for i in np.flatnonzero(compact["action_mask"])]
                action = rng.choice(legal)
                compact, _, done, _, _ = self.compact_env.step(action)
                dense, _, dense_done, _, _ = self.dense_env.step(action)
                self.assertEqual(done, dense_done)
                if done:
                    break

    def test_bridge_equals_gym_compact(self):
        from spire_learn.neural.snapshot import snapshot_common

        np, widen = self.np, self.bridge_module.widen
        checked = 0
        for compact, dense in self.frames():
            widened = widen(compact, self.dense_descriptor)
            for key in dense:  # the Gym's documented equivalence, checked on its own twin sessions
                self.assertEqual(widened[key].dtype, dense[key].dtype, key)
                self.assertTrue(np.array_equal(widened[key], dense[key]), key)
            arrays, frame = self.bridge(widened)
            self.assertEqual(set(arrays), set(compact))
            for key, array in compact.items():
                self.assertEqual((arrays[key].dtype, arrays[key].shape), (array.dtype, array.shape), key)
                self.assertEqual(arrays[key].tobytes(), np.ascontiguousarray(array).tobytes(), key)
            expected = snapshot_common(compact, self.schema)
            self.assertEqual({k: bytes(v.data) for k, v in frame.tensors.items()},
                             {k: bytes(v.data) for k, v in expected.tensors.items()})
            checked += 1
        self.assertGreaterEqual(checked, 12)

    def test_bridge_overflow_raises(self):
        np, error = self.np, self.bridge_module.BridgeError
        compact, _ = next(iter(self.frames(steps=1)))
        base = self.bridge_module.widen(compact, self.dense_descriptor)
        capacity = {tensor["key"]: tensor["shape"] for tensor in self.compact_descriptor["tensors"]}

        def broken(change):
            dense = {key: array.copy() for key, array in base.items()}
            change(dense)
            return dense

        powers = capacity["powers"][0]
        potions = capacity["potions"][0]
        cases = {
            "occupied row past capacity": lambda d: d["powers_occupied"].__setitem__(powers, 1),
            "row values past capacity": lambda d: d["powers"].__setitem__((powers + 3, 2), 1),
            "known empty potion slot past capacity": lambda d: d["potions_known"].__setitem__((potions, 0), 1),
            "potion value outside int16": lambda d: d["potions"].__setitem__((0, 0), 40_000),
            "gold outside int32": lambda d: d["player"].__setitem__(2, 2 ** 31),
            "legal action past the compact width": lambda d: d["action_mask"].__setitem__(capacity["actions"][0], 1),
        }
        for name, change in cases.items():
            with self.subTest(name), self.assertRaises(error):
                self.bridge.arrays(broken(change))
        # The same rule guards orbs: a known empty orb slot past a narrower orb capacity raises.
        narrow = copy.deepcopy(self.compact_descriptor)
        orbs = next(t for t in narrow["tensors"] if t["key"] == "orbs")
        for tensor in narrow["tensors"]:
            if tensor["key"] in ("orbs", "orbs_known", "orbs_occupied"):
                tensor["shape"][0] = 2
        del orbs
        bridge = self.bridge_module.CompactBridge(self.dense_descriptor, narrow)
        dense = broken(lambda d: d["orbs_known"].__setitem__((3, 0), 1))
        with self.assertRaisesRegex(error, "orbs_known"):
            bridge.arrays(dense)
        self.assertEqual(bridge.arrays(base)["orbs"].shape[0], 2)
        self.assertTrue(np.array_equal(self.bridge.arrays(base)["action_mask"], compact["action_mask"]))


if __name__ == "__main__":
    unittest.main()
