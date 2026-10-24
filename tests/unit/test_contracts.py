import dataclasses
import unittest

from spire_learn.contracts import ArrayValue, ContractError, frozen_mapping


class ArrayValueTests(unittest.TestCase):
    def test_validates_shape_count_and_finite_numeric_values(self):
        self.assertEqual(ArrayValue("int64", (2,), (1, 2)).values, (1, 2))
        for shape, values in [((2,), (1,)), ((-1,), ()), ((1,), (True,)), ((1,), (float("inf"),)), ((1,), (object(),))]:
            with self.subTest(shape=shape, values=values), self.assertRaises(ContractError):
                ArrayValue("float32", shape, values)

    def test_frozen_mapping_owns_data_and_cannot_be_mutated(self):
        source = {"hp": ArrayValue("int64", (), (10,))}
        result = frozen_mapping(source)
        source["new"] = ArrayValue("int64", (), (0,))
        self.assertEqual(set(result), {"hp"})
        with self.assertRaises(TypeError):
            result["hp"] = ArrayValue("int64", (), (0,))

    def test_records_are_frozen(self):
        record = ArrayValue("int8", (), (1,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.dtype = "int64"
