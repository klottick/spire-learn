"""Three opt-in CPU API checks for the Dict-final TorchRL reader."""

import json
import os
import sys
import unittest


_RUN_READER = os.environ.get("SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION") == "1"


def _spire_lab_modules():
    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == "spire_lab" or name.startswith("spire_lab.")
        )
    )


def _native_binding_modules():
    return tuple(
        sorted(
            name
            for name in sys.modules
            if name == "spire_lab._native" or name.startswith("spire_lab._native.")
        )
    )


def _assert_import_gate(test_case, test_name, phase, *, before=None, evidence=None):
    imported = _spire_lab_modules()
    bindings = _native_binding_modules()
    test_case.assertEqual(imported, (), "API test imported spire_lab")
    test_case.assertEqual(bindings, (), "API test imported the native binding")
    if before is not None:
        test_case.assertEqual(imported, before, "API test changed the native import gate")
    print(
        "DICT_FINAL_API_GATE "
        + json.dumps(
            {
                "test": test_name,
                "phase": phase,
                "spire_lab_modules": imported,
                "native_binding_modules": bindings,
                "evidence": evidence or {},
            },
            sort_keys=True,
        )
    )
    return imported


def _schema(*, with_outcome=False):
    from spire_learn.schema import parse_descriptor
    from tests.support import tiny_descriptor

    descriptor = tiny_descriptor()
    if with_outcome:
        descriptor["tensors"].append(
            {
                "key": "outcome",
                "dtype": "int32",
                "shape": [1],
                "fields": ["code"],
                "axes": ["value"],
                "category": None,
                "mask": None,
            }
        )
    return parse_descriptor(
        descriptor, allowed_schema_ids=frozenset({"spire-learn-fixture-v1"})
    )


def _public_observation_spec(torch, Composite, Unbounded, schema, width):
    return Composite(
        {
            spec.key: Unbounded(
                shape=torch.Size([width, *spec.shape]),
                dtype=getattr(torch, spec.dtype),
                device="cpu",
            )
            for spec in schema.tensors
        },
        shape=torch.Size([width]),
        device="cpu",
    )


def _array(np, spec, start):
    count = 1
    for dimension in spec.shape:
        count *= dimension
    values = np.arange(start, start + count, dtype=spec.dtype)
    return values.reshape(spec.shape)


def _row(np, schema, start):
    return {
        spec.key: _array(np, spec, start + index * 11)
        for index, spec in enumerate(schema.tensors)
    }


@unittest.skipUnless(
    _RUN_READER,
    "set SPIRE_LEARN_RUN_TORCHRL_READER_INTEGRATION=1 in the approved offline shard",
)
class TorchRLDictFinalReaderIntegrationTests(unittest.TestCase):
    def test_mixed_dict_final_values_and_presence(self):
        test_name = "test_mixed_dict_final_values_and_presence"
        api_calls = {
            "environment_calls": 0,
            "action_calls": 0,
            "sampler_calls": 0,
            "diagnostic_draws": 0,
        }
        before_imports = _assert_import_gate(
            self,
            test_name,
            "before_optional_imports",
            evidence=api_calls,
        )
        import numpy as np
        import torch
        from tensordict import TensorDict
        from torchrl.data import Composite, Unbounded

        from spire_learn.probes.torchrl_dict_final import make_dict_final_reader
        from torchrl.envs.gym_like import BaseInfoDictReader

        schema = _schema()
        width = 2
        observation_spec = _public_observation_spec(
            torch, Composite, Unbounded, schema, width
        )
        reader = make_dict_final_reader(
            observation_spec=observation_spec,
            schema=schema,
            torch=torch,
            info_reader_type=BaseInfoDictReader,
            composite_type=Composite,
        )

        main = {
            spec.key: torch.full(
                (width, *spec.shape), 3, dtype=getattr(torch, spec.dtype)
            )
            for spec in schema.tensors
        }
        main_before = {key: value.clone() for key, value in main.items()}
        done = torch.tensor([[True], [False]], dtype=torch.bool, device="cpu")
        tensordict = TensorDict({**main, "done": done}, batch_size=[width])
        terminal_row = _row(np, schema, 20)
        terminal_before = {key: value.copy() for key, value in terminal_row.items()}
        final_obs = np.empty((width,), dtype=object)
        final_obs[0] = terminal_row
        final_obs[1] = None
        info = {
            "final_obs": final_obs,
            "_final_obs": np.asarray([True, False], dtype=np.bool_),
        }

        result = reader(info, tensordict)

        self.assertIs(result, tensordict)
        self.assertTrue(np.array_equal(final_obs[0]["player"], terminal_before["player"]))
        self.assertIsNone(final_obs[1])
        for spec in schema.tensors:
            final_leaf = tensordict.get(("final", spec.key))
            self.assertEqual(final_leaf.dtype, getattr(torch, spec.dtype))
            self.assertEqual(tuple(final_leaf.shape), (width, *spec.shape))
            expected = torch.as_tensor(
                terminal_before[spec.key], dtype=getattr(torch, spec.dtype)
            )
            self.assertTrue(torch.equal(final_leaf[0], expected), spec.key)
            self.assertEqual(torch.count_nonzero(final_leaf[1]).item(), 0, spec.key)
            self.assertTrue(torch.equal(tensordict.get(spec.key), main_before[spec.key]))
        self.assertEqual(api_calls, {key: 0 for key in api_calls})
        _assert_import_gate(
            self,
            test_name,
            "after_api_test",
            before=before_imports,
            evidence=api_calls,
        )

    def test_info_outcome_does_not_replace_observation_outcome(self):
        test_name = "test_info_outcome_does_not_replace_observation_outcome"
        api_calls = {
            "environment_calls": 0,
            "action_calls": 0,
            "sampler_calls": 0,
            "diagnostic_draws": 0,
        }
        before_imports = _assert_import_gate(
            self,
            test_name,
            "before_optional_imports",
            evidence=api_calls,
        )
        import numpy as np
        import torch
        from tensordict import TensorDict
        from torchrl.data import Composite, Unbounded

        from spire_learn.probes.torchrl_dict_final import make_dict_final_reader
        from torchrl.envs.gym_like import BaseInfoDictReader

        schema = _schema(with_outcome=True)
        width = 2
        observation_spec = _public_observation_spec(
            torch, Composite, Unbounded, schema, width
        )
        reader = make_dict_final_reader(
            observation_spec=observation_spec,
            schema=schema,
            torch=torch,
            info_reader_type=BaseInfoDictReader,
            composite_type=Composite,
        )
        calls = {"environment": 0, "sampler": 0}
        row = _row(np, schema, 40)
        row["outcome"] = np.asarray([17], dtype=np.int32)
        final_obs = np.empty((width,), dtype=object)
        final_obs[0] = row
        final_obs[1] = None
        info = {
            "final_obs": final_obs,
            "_final_obs": np.asarray([True, False], dtype=np.bool_),
            "outcome": np.asarray(["ongoing", "ongoing"], dtype=object),
            "final_info": {"outcome": np.asarray(["death", ""], dtype=object)},
        }
        tensordict = TensorDict(
            {
                "done": torch.tensor([True, False], dtype=torch.bool),
                **{
                    spec.key: torch.zeros(
                        (width, *spec.shape), dtype=getattr(torch, spec.dtype)
                    )
                    for spec in schema.tensors
                },
            },
            batch_size=[width],
        )

        reader(info, tensordict)

        observed = tensordict.get(("final", "outcome"))
        self.assertEqual(observed.dtype, torch.int32)
        self.assertEqual(tuple(observed.shape), (width, 1))
        self.assertEqual(observed[:, 0].tolist(), [17, 0])
        self.assertEqual(calls, {"environment": 0, "sampler": 0})
        self.assertEqual(api_calls, {key: 0 for key in api_calls})
        _assert_import_gate(
            self,
            test_name,
            "after_api_test",
            before=before_imports,
            evidence=api_calls,
        )

    def test_reader_registration_has_zero_environment_calls(self):
        test_name = "test_reader_registration_has_zero_environment_calls"
        calls = {
            "reset": 0,
            "step": 0,
            "check_env_specs": 0,
            "rollout": 0,
            "actions": 0,
            "diagnostic_draws": 0,
        }
        before_imports = _assert_import_gate(
            self,
            test_name,
            "before_optional_imports",
            evidence=calls,
        )
        import numpy as np
        import torch
        from tensordict import TensorDict
        from torchrl.data import Composite, Unbounded
        from torchrl.envs.gym_like import BaseInfoDictReader

        from spire_learn.contracts import ContractError
        from spire_learn.probes import torchrl
        from spire_learn.probes.torchrl_dict_final import make_dict_final_reader

        schema = _schema()
        width = 2
        observation_spec = _public_observation_spec(
            torch, Composite, Unbounded, schema, width
        )
        class CounterOwner:
            def __init__(self, spec):
                self.observation_spec = spec
                self.info_dict_reader = []
                self.registered = []

            def set_info_dict_reader(self, *, info_dict_reader):
                self.registered.append(info_dict_reader)
                self.info_dict_reader.append(info_dict_reader)
                self.observation_spec.update(info_dict_reader.info_spec)
                return self

            def reset(self, *args, **kwargs):
                calls["reset"] += 1

            def step(self, *args, **kwargs):
                calls["step"] += 1
                calls["actions"] += 1

            def check_env_specs(self, *args, **kwargs):
                calls["check_env_specs"] += 1

            def rollout(self, *args, **kwargs):
                calls["rollout"] += 1

        class VendorReader:
            def __init__(self):
                self.calls = 0

            def __call__(self, info, tensordict):
                self.calls += 1
                return tensordict

        def factory(spec):
            return make_dict_final_reader(
                observation_spec=spec,
                schema=schema,
                torch=torch,
                info_reader_type=BaseInfoDictReader,
                composite_type=Composite,
            )

        wrapper_type = torchrl._passive_gym_wrapper_class(
            CounterOwner, reader_factory=factory
        )
        owner = wrapper_type(observation_spec)
        vendor = VendorReader()
        owner.auto_register_info_dict(info_dict_reader=vendor)
        reader = owner.info_dict_reader[0]

        self.assertEqual(owner.registered, [reader])
        self.assertIsNot(reader, vendor)
        self.assertEqual(vendor.calls, 0)
        self.assertIsInstance(reader, BaseInfoDictReader)
        self.assertEqual(calls, {
            "reset": 0,
            "step": 0,
            "check_env_specs": 0,
            "rollout": 0,
            "actions": 0,
            "diagnostic_draws": 0,
        })

        reset_tensordict = TensorDict(
            {spec.key: torch.zeros((width, *spec.shape), dtype=getattr(torch, spec.dtype)) for spec in schema.tensors},
            batch_size=[width],
        )
        reader({"outcome": ("ongoing", "ongoing")}, reset_tensordict)
        ongoing_tensordict = TensorDict(
            {
                "done": torch.tensor([False, False], dtype=torch.bool),
                **{
                    spec.key: torch.ones(
                        (width, *spec.shape), dtype=getattr(torch, spec.dtype)
                    )
                    for spec in schema.tensors
                },
            },
            batch_size=[width],
        )
        ongoing_before = {
            spec.key: ongoing_tensordict.get(spec.key).clone()
            for spec in schema.tensors
        }
        reader({"outcome": np.asarray(["ongoing", "ongoing"], dtype=object)}, ongoing_tensordict)
        self.assertTrue(
            all(
                torch.equal(ongoing_tensordict.get(key), value)
                for key, value in ongoing_before.items()
            )
        )
        with self.assertRaises(ValueError):
            reader(
                {"final_obs": np.asarray([None, None], dtype=object), "_final_obs": np.asarray([False, False])},
                TensorDict({"done": torch.tensor([True, False], dtype=torch.bool)}, batch_size=[width]),
            )
        malformed_row = _row(np, schema, 70)
        del malformed_row["player"]
        malformed_final_obs = np.empty((width,), dtype=object)
        malformed_final_obs[0] = malformed_row
        malformed_final_obs[1] = None
        malformed_tensordict = TensorDict(
            {"done": torch.tensor([True, False], dtype=torch.bool)},
            batch_size=[width],
        )
        with self.assertRaises(ContractError):
            reader(
                {
                    "final_obs": malformed_final_obs,
                    "_final_obs": np.asarray([True, False], dtype=np.bool_),
                },
                malformed_tensordict,
            )
        for spec in schema.tensors:
            self.assertIsNone(
                malformed_tensordict.get(("final", spec.key), None), spec.key
            )
        self.assertEqual(calls, {
            "reset": 0,
            "step": 0,
            "check_env_specs": 0,
            "rollout": 0,
            "actions": 0,
            "diagnostic_draws": 0,
        })
        evidence = {**calls, "vendor_reader_calls": vendor.calls}
        _assert_import_gate(
            self,
            test_name,
            "after_api_test",
            before=before_imports,
            evidence=evidence,
        )


if __name__ == "__main__":
    unittest.main()
