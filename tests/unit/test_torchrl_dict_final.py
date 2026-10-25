"""Pure protocol tests for the learner-owned Dict final reader."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest

from spire_learn.contracts import ArrayValue, ContractError, SetupBlocked
from spire_learn.schema import parse_descriptor
from tests.support import FakeArray, tiny_descriptor


def _module(test: unittest.TestCase):
    try:
        return importlib.import_module("spire_learn.probes.torchrl_dict_final")
    except ModuleNotFoundError:
        test.fail("the dictionary-aware final-observation reader is not implemented")


def _schema(*, outcome_leaf: bool = False, float_leaf: bool = False):
    descriptor = tiny_descriptor()
    if outcome_leaf:
        descriptor["tensors"] = [
            *descriptor["tensors"],
            {
                "key": "outcome",
                "dtype": "int32",
                "shape": [],
                "fields": ["outcome"],
                "axes": [],
                "category": None,
                "mask": None,
            },
        ]
    if float_leaf:
        descriptor["tensors"] = [
            *descriptor["tensors"],
            {
                "key": "score",
                "dtype": "float32",
                "shape": [],
                "fields": ["score"],
                "axes": [],
                "category": None,
                "mask": None,
            },
        ]
    return parse_descriptor(
        descriptor,
        allowed_schema_ids=frozenset({"spire-learn-fixture-v1"}),
    )


def _final_observation(hp: int, *, outcome: int | None = None):
    row = {
        "action_mask": FakeArray("int8", (3,), (1, 1, 0)),
        "action_kind": FakeArray("int32", (3,), (1, 2, 0)),
        "action_source": FakeArray("int32", (3,), (-1, 0, -1)),
        "action_target": FakeArray("int32", (3,), (-1, 0, -1)),
        "action_features": FakeArray(
            "int32", (3, 4), (0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0)
        ),
        "player": FakeArray("int64", (), (hp,)),
    }
    if outcome is not None:
        row["outcome"] = FakeArray("int32", (), (outcome,))
    return row


def _final_info(ended=(True, False)):
    return {
        "episode_id": ("old-0" if ended[0] else "", "old-1" if ended[1] else ""),
        "_episode_id": tuple(ended),
        "scenario_id": ("cultist" if ended[0] else "", "jaw-worm" if ended[1] else ""),
        "_scenario_id": tuple(ended),
        "episode_steps": (5 if ended[0] else 0, 7 if ended[1] else 0),
        "_episode_steps": tuple(ended),
        "outcome": ("death" if ended[0] else "", "victory" if ended[1] else ""),
        "_outcome": tuple(ended),
    }


class _Device:
    type = "cpu"


class _LeafSpec:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.device = _Device()

    def clone(self):
        return type(self)(self.dtype, self.shape)


class _Composite(dict):
    def __init__(self, values=None, *, shape=(), device=None):
        super().__init__(values or {})
        self.shape = tuple(shape)
        self.device = device or _Device()

    def clone(self):
        return type(self)(
            {key: value.clone() if callable(getattr(value, "clone", None)) else value
             for key, value in self.items()},
            shape=self.shape,
            device=self.device,
        )

    def zero(self):
        width = self.shape[0]
        return _FakeTensorDict(width, (False,) * width)


class _BaseInfoDictReader:
    pass


class _FakeTorch:
    int8 = "int8"
    int32 = "int32"
    int64 = "int64"
    float32 = "float32"

    @classmethod
    def tensor(cls, values, *, dtype, device):
        shape = [len(values)]
        value = values[0] if values else ()
        while isinstance(value, (list, tuple)):
            shape.append(len(value))
            value = value[0] if value else None
        return _MaterializedTensor(dtype, tuple(shape), device)


class _MaterializedTensor:
    def __init__(self, dtype, shape, device):
        self.dtype = dtype
        self.shape = shape
        self.device = _Device()


class _FakeTensorDict:
    def __init__(self, width, done):
        self.batch_size = (width,)
        self.values = {"done": _CpuTensor(FakeArray("bool", (width,), done))}
        self.set_calls = []

    def get(self, key):
        return self.values[key]

    def set(self, key, value):
        self.set_calls.append((key, value))
        self.values[key] = value
        return self


class _BatchedArray:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.dtype = self.rows[0].dtype
        self.shape = (len(self.rows), *self.rows[0].shape)

    def __getitem__(self, index):
        return self.rows[index]

    def tolist(self):
        return [row.tolist() for row in self.rows]


class _CpuTensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.device = _Device()

    def detach(self):
        return self

    def numpy(self):
        return self.array


def _observation_spec(schema, *, width=2):
    return _Composite(
        {
            leaf.key: _LeafSpec(leaf.dtype, (width, *leaf.shape))
            for leaf in schema.tensors
        },
        shape=(width,),
    )


class _WrapperOwner:
    def __init__(self, observation_spec):
        self.observation_spec = observation_spec
        self.info_dict_reader = []
        self.registered = []
        self.reset_calls = 0
        self.step_calls = 0

    def set_info_dict_reader(self, *, info_dict_reader):
        self.registered.append(info_dict_reader)
        self.info_dict_reader.append(info_dict_reader)
        return self

    def reset(self, *args, **kwargs):
        self.reset_calls += 1
        raise AssertionError("reader registration must not reset an environment")

    def step(self, *args, **kwargs):
        self.step_calls += 1
        raise AssertionError("reader registration must not step an environment")


class DictFinalPreparationTests(unittest.TestCase):
    def test_reader_rejects_tensor_dict_width_different_from_bound_spec_before_set(self):
        module = _module(self)
        schema = _schema()
        reader = module.make_dict_final_reader(
            observation_spec=_observation_spec(schema, width=2),
            schema=schema,
            torch=_FakeTorch,
            info_reader_type=_BaseInfoDictReader,
            composite_type=_Composite,
        )
        tensordict = _FakeTensorDict(3, (True, True, True))
        info = {
            "final_obs": [_final_observation(31), _final_observation(32), _final_observation(33)],
            "_final_obs": (True, True, True),
        }

        with self.assertRaisesRegex(ContractError, "batch.*bound|bound.*batch"):
            reader(info, tensordict)

        self.assertEqual(tensordict.set_calls, [])

    def test_reader_rejects_unrepresentable_final_values_before_any_set(self):
        module = _module(self)
        cases = (
            (_schema(), _final_observation(37), "int8"),
            (_schema(float_leaf=True), _final_observation(37), "float32"),
        )
        cases[0][1]["action_mask"] = FakeArray("int8", (3,), (256, 1, 0))
        cases[1][1]["score"] = FakeArray("float32", (), (1.1,))

        for schema, row, dtype in cases:
            with self.subTest(dtype=dtype):
                reader = module.make_dict_final_reader(
                    observation_spec=_observation_spec(schema, width=1),
                    schema=schema,
                    torch=_FakeTorch,
                    info_reader_type=_BaseInfoDictReader,
                    composite_type=_Composite,
                )
                tensordict = _FakeTensorDict(1, (True,))
                info = {"final_obs": [row], "_final_obs": (True,)}

                with self.assertRaises(ContractError):
                    reader(info, tensordict)

                self.assertEqual(tensordict.set_calls, [])

    def test_owned_final_rows_preserve_values_and_leave_inputs_and_none_untouched(self):
        prepare = _module(self).prepare_final_rows
        ended = (True, False)
        row = _final_observation(37)
        rows = [row, None]
        info = {
            "final_obs": rows,
            "_final_obs": ended,
            "final_info": _final_info(ended),
            "_final_info": ended,
        }

        original_info = info["final_info"]
        original_rows = info["final_obs"]
        prepared = prepare(info, ended, _schema())
        rows[0]["player"] = FakeArray("int64", (), (99,))

        self.assertEqual(prepared[0]["player"], ArrayValue("int64", (), (37,)))
        self.assertEqual(prepared[0]["action_mask"].values, (1, 1, 0))
        self.assertIsNone(prepared[1])
        self.assertIs(info["final_obs"], original_rows)
        self.assertIs(info["final_info"], original_info)
        self.assertIsNone(rows[1])
        self.assertEqual(info["_final_obs"], ended)

    def test_registration_and_live_batch_without_final_fields_returns_absent_rows(self):
        prepare = _module(self).prepare_final_rows

        self.assertEqual(prepare({}, (False, False), _schema()), (None, None))

    def test_missing_final_leaf_is_rejected_without_a_partial_result(self):
        prepare = _module(self).prepare_final_rows
        row = _final_observation(37)
        del row["player"]
        info = {
            "final_obs": [row, None],
            "_final_obs": (True, False),
            "final_info": _final_info(),
            "_final_info": (True, False),
        }

        with self.assertRaisesRegex(ContractError, "keys|schema"):
            prepare(info, (True, False), _schema())

    def test_final_masks_must_match_the_public_done_rows(self):
        prepare = _module(self).prepare_final_rows
        info = {
            "final_obs": [_final_observation(37), None],
            "_final_obs": (False, False),
            "final_info": _final_info(),
            "_final_info": (True, False),
        }

        with self.assertRaisesRegex(ContractError, "presence|done"):
            prepare(info, (True, False), _schema())

    def test_leaf_dtype_and_shape_are_validated_before_preparation(self):
        prepare = _module(self).prepare_final_rows
        wrong_dtype = _final_observation(37)
        wrong_dtype["player"] = FakeArray("int32", (), (37,))
        wrong_shape = _final_observation(37)
        wrong_shape["action_mask"] = FakeArray("int8", (2,), (1, 1))

        for row, expected in ((wrong_dtype, "dtype"), (wrong_shape, "shape")):
            info = {
                "final_obs": [row, None],
                "_final_obs": (True, False),
                "final_info": _final_info(),
                "_final_info": (True, False),
            }
            with self.subTest(expected=expected), self.assertRaisesRegex(ContractError, expected):
                prepare(info, (True, False), _schema())

    def test_observation_outcome_leaf_is_never_sourced_from_info_outcome(self):
        prepare = _module(self).prepare_final_rows
        ended = (True, False)
        final = _final_observation(37, outcome=2)
        info = {
            "final_obs": [final, None],
            "_final_obs": ended,
            "final_info": _final_info(ended),
            "_final_info": ended,
            "outcome": ("victory", "ongoing"),
        }

        prepared = prepare(info, ended, _schema(outcome_leaf=True))

        self.assertEqual(prepared[0]["outcome"], ArrayValue("int32", (), (2,)))


class DictFinalRegistrationTests(unittest.TestCase):
    def test_known_spec_factory_registers_only_the_learner_reader_passively(self):
        module = _module(self)
        schema = _schema()
        observation_spec = _observation_spec(schema)
        factory_calls = []
        vendor_calls = []

        def factory(spec):
            factory_calls.append(spec)
            return module.make_dict_final_reader(
                observation_spec=spec,
                schema=schema,
                torch=_FakeTorch,
                info_reader_type=_BaseInfoDictReader,
                composite_type=_Composite,
            )

        class VendorReader:
            def __call__(self, info, tensordict):
                vendor_calls.append((info, tensordict))

        vendor = VendorReader()
        wrapper_type = importlib.import_module("spire_learn.probes.torchrl")._passive_gym_wrapper_class(
            _WrapperOwner, reader_factory=factory
        )
        wrapper = wrapper_type(observation_spec)

        result = wrapper.auto_register_info_dict(info_dict_reader=vendor)

        self.assertIs(result, wrapper)
        self.assertEqual(factory_calls, [observation_spec])
        self.assertEqual(vendor_calls, [])
        self.assertEqual(wrapper.registered, wrapper.info_dict_reader)
        self.assertEqual(len(wrapper.registered), 1)
        self.assertIs(wrapper.registered[0], wrapper.info_dict_reader[0])
        self.assertEqual(wrapper.reset_calls, 0)
        self.assertEqual(wrapper.step_calls, 0)
        self.assertEqual(
            wrapper.registered[0].adapter_id, module.DICT_FINAL_ADAPTER_ID
        )
        self.assertEqual(
            set(wrapper.registered[0].info_spec["final"]),
            set(observation_spec),
        )

    def test_invalid_known_spec_registers_nothing_and_does_not_call_vendor(self):
        module = _module(self)
        schema = _schema()
        observation_spec = _observation_spec(schema)
        observation_spec["player"] = _LeafSpec("float32", (2,))
        vendor_calls = []

        def factory(spec):
            return module.make_dict_final_reader(
                observation_spec=spec,
                schema=schema,
                torch=_FakeTorch,
                info_reader_type=_BaseInfoDictReader,
                composite_type=_Composite,
            )

        class VendorReader:
            def __call__(self, info, tensordict):
                vendor_calls.append((info, tensordict))

        wrapper_type = importlib.import_module("spire_learn.probes.torchrl")._passive_gym_wrapper_class(
            _WrapperOwner, reader_factory=factory
        )
        wrapper = wrapper_type(observation_spec)
        with self.assertRaises(SetupBlocked):
            wrapper.auto_register_info_dict(info_dict_reader=VendorReader())

        self.assertEqual(wrapper.registered, [])
        self.assertEqual(wrapper.reset_calls, 0)
        self.assertEqual(wrapper.step_calls, 0)
        self.assertEqual(vendor_calls, [])


class DictFinalArtifactTests(unittest.TestCase):
    def test_adapter_identity_and_mapping_mismatch_are_persisted_by_run_writer(self):
        module = _module(self)
        torchrl = importlib.import_module("spire_learn.probes.torchrl")
        schema = _schema()
        expected_rows = torchrl._snapshot_final_rows(
            [_final_observation(37)], (True,), schema
        )
        observed_leaves = {}
        for spec in schema.tensors:
            expected_leaf = expected_rows[0][spec.key]
            observed_leaf = (
                FakeArray("int32", (1,), (38,))
                if spec.key == "player"
                else FakeArray(expected_leaf.dtype, expected_leaf.shape, expected_leaf.values)
            )
            observed_leaves[spec.key] = _CpuTensor(_BatchedArray([observed_leaf]))
        with self.assertRaises(torchrl._TorchRLMappingMismatch) as raised:
            torchrl._mapped_observation_rows(
                observed_leaves,
                schema,
                width=1,
                expected_rows=expected_rows,
                mismatch_check="replacement_reset",
                phase="replacement_reset",
            )
        details = raised.exception.details
        self.assertEqual(raised.exception.check, "replacement_reset")
        self.assertEqual(details["phase"], "replacement_reset")
        self.assertEqual(details["row"], 0)
        self.assertEqual(details["key"], "player")
        self.assertEqual(details["expected_dtype"], "int64")
        self.assertEqual(details["observed_dtype"], "int32")
        self.assertEqual(details["expected_shape"], [])
        self.assertEqual(details["observed_shape"], [1])
        self.assertEqual(details["expected_values"], [37])
        self.assertEqual(details["observed_values"], [38])

        with tempfile.TemporaryDirectory(prefix="spire-learn-dict-final-") as temporary:
            from spire_learn.artifacts.writer import RunWriter

            writer = RunWriter.create(Path(temporary), "dict-final-artifacts")
            try:
                writer.update_manifest(
                    {
                        "event": "torchrl_probe_discovery",
                        "adapter": {
                            "configured_info_dict_reader_id": module.DICT_FINAL_ADAPTER_ID
                        },
                    }
                )
                writer.update_manifest(
                    {
                        "event": "torchrl_probe_reader_registration",
                        "adapter": torchrl._reader_identity_metadata(
                            configured_id=module.DICT_FINAL_ADAPTER_ID,
                            registered_id=module.DICT_FINAL_ADAPTER_ID,
                            registration_succeeded=True,
                        ),
                    }
                )
                torchrl._persist_mapping_mismatch(writer, details)
                writer.finalize(
                    {
                        "adapter": {
                            **torchrl._reader_identity_metadata(
                                configured_id=module.DICT_FINAL_ADAPTER_ID,
                                registered_id=module.DICT_FINAL_ADAPTER_ID,
                                registration_succeeded=True,
                            )
                        }
                    }
                )
            finally:
                writer.close()

            run_dir = Path(temporary) / "dict-final-artifacts"
            manifest_updates = [
                json.loads(line)
                for line in (run_dir / "manifest_updates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            discovery = next(
                item["record"]["adapter"]
                for item in manifest_updates
                if item["record"].get("event") == "torchrl_probe_discovery"
            )
            registration = next(
                item["record"]["adapter"]
                for item in manifest_updates
                if item["record"].get("event") == "torchrl_probe_reader_registration"
            )
            persisted_metric = json.loads(
                (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )["summary"]

        mismatch = persisted_metric["record"]["mapping_mismatch"][0]
        self.assertEqual(persisted_metric["record"]["phase"], "replacement_reset")
        self.assertEqual(mismatch["phase"], "replacement_reset")
        self.assertEqual(
            discovery["configured_info_dict_reader_id"], module.DICT_FINAL_ADAPTER_ID
        )
        self.assertNotIn("info_dict_reader_id", discovery)
        self.assertEqual(registration["info_dict_reader_id"], module.DICT_FINAL_ADAPTER_ID)
        self.assertEqual(
            summary["adapter"]["info_dict_reader_id"],
            module.DICT_FINAL_ADAPTER_ID,
        )
        self.assertEqual(
            (mismatch["row"], mismatch["key"]), (0, "player")
        )
        self.assertEqual(mismatch["expected_dtype"], "int64")
        self.assertEqual(mismatch["observed_dtype"], "int32")
        self.assertEqual(mismatch["expected_shape"], [])
        self.assertEqual(mismatch["observed_shape"], [1])
        self.assertEqual(mismatch["expected_values"], [37])
        self.assertEqual(mismatch["observed_values"], [38])

    def test_failed_registration_artifacts_do_not_claim_a_registered_reader(self):
        module = _module(self)
        torchrl = importlib.import_module("spire_learn.probes.torchrl")
        schema = _schema()
        observation_spec = _observation_spec(schema)
        created = []

        def factory(spec):
            reader = module.make_dict_final_reader(
                observation_spec=spec,
                schema=schema,
                torch=_FakeTorch,
                info_reader_type=_BaseInfoDictReader,
                composite_type=_Composite,
            )
            created.append(reader)
            return reader

        class RaisingOwner(_WrapperOwner):
            def set_info_dict_reader(self, *, info_dict_reader):
                raise RuntimeError("authored registration failure")

        wrapper_type = torchrl._passive_gym_wrapper_class(
            RaisingOwner, reader_factory=factory
        )
        wrapper = wrapper_type(observation_spec)
        with self.assertRaisesRegex(RuntimeError, "registration failure"):
            wrapper.auto_register_info_dict(info_dict_reader=object())
        self.assertEqual(wrapper.registered, [])
        self.assertEqual(len(created), 1)

        identity = torchrl._reader_identity_metadata(
            configured_id=module.DICT_FINAL_ADAPTER_ID,
            registered_id=created[0].adapter_id,
            registration_succeeded=bool(wrapper.registered),
        )
        self.assertEqual(
            identity,
            {
                "configured_info_dict_reader_id": module.DICT_FINAL_ADAPTER_ID,
                "info_dict_reader_id": None,
            },
        )

        with tempfile.TemporaryDirectory(prefix="spire-learn-dict-final-failed-registration-") as temporary:
            from spire_learn.artifacts.writer import RunWriter

            writer = RunWriter.create(Path(temporary), "failed-registration")
            try:
                writer.update_manifest(
                    {
                        "event": "torchrl_probe_discovery",
                        "adapter": {
                            "configured_info_dict_reader_id": module.DICT_FINAL_ADAPTER_ID
                        },
                    }
                )
                writer.finalize({"adapter": identity})
            finally:
                writer.close()

            run_dir = Path(temporary) / "failed-registration"
            discovery = json.loads(
                (run_dir / "manifest_updates.jsonl").read_text(encoding="utf-8")
            )["record"]["adapter"]
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )["summary"]["adapter"]
        self.assertEqual(
            discovery["configured_info_dict_reader_id"], module.DICT_FINAL_ADAPTER_ID
        )
        self.assertNotIn("info_dict_reader_id", discovery)
        self.assertEqual(
            summary["configured_info_dict_reader_id"], module.DICT_FINAL_ADAPTER_ID
        )
        self.assertIsNone(summary["info_dict_reader_id"])


if __name__ == "__main__":
    unittest.main()
