"""Opt-in tensor tests for public combat features and the masked actor-critic."""

from __future__ import annotations

from copy import copy
from dataclasses import is_dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import time
import unittest
import warnings
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from tests.neural._support import (
    CandidateInput,
    accepted_schema_and_config,
    authored_frame,
    frame_bytes,
    synthetic_schema,
    tensor_leaves,
)


_TENSOR_GATE = "SPIRE_LEARN_RUN_NEURAL_TENSOR"


def setUpModule() -> None:
    if os.environ.get(_TENSOR_GATE) == "1":
        from tests.neural._sample_guard import activate_from_environment

        activate_from_environment()


def _api(test: unittest.TestCase) -> SimpleNamespace:
    try:
        from spire_learn.neural import features, model
        from spire_learn.neural.contracts import ContractError, ModelConfig
    except ImportError as error:
        test.fail(f"opt-in neural model API is unavailable: {error}")
    return SimpleNamespace(
        features=features,
        model=model,
        ContractError=ContractError,
        ModelConfig=ModelConfig,
    )


def _small_model(api: SimpleNamespace, schema: object, torch) -> object:
    return api.model.build_policy(
        schema,
        api.ModelConfig(embedding_width=16, hidden_width=64, max_parameters=10_000_000),
        seed=42,
        device="cpu",
    )


def _replace_coverage(batch: object, coverage: object) -> object:
    if is_dataclass(batch):
        return replace(batch, known_coverage=coverage)
    updated = copy(batch)
    object.__setattr__(updated, "known_coverage", coverage)
    return updated


class NeuralModelTensorTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_accepted_descriptor_validates_empty_and_typed_relation_frames(self):
        from spire_learn.neural.schema import validate_frame

        schema, _config = accepted_schema_and_config()
        empty = authored_frame(schema, populate_relations=False)
        related = authored_frame(schema, power_owners=(0, 1), populate_relations=True)
        validate_frame(empty, schema)
        validate_frame(related, schema)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_parameter_cap_rejects_huge_widths_before_tensor_allocation(self):
        import torch

        api = _api(self)
        with synthetic_schema() as schema:
            config = api.ModelConfig(
                embedding_width=1_000_000_000,
                hidden_width=1_000_000_000,
                max_parameters=10_000_000,
            )

            def forbidden_allocation(*_args, **_kwargs):
                raise AssertionError("parameter allocation began before the configured cap check")

            with (
                patch.object(torch, "empty", new=forbidden_allocation),
                patch.object(torch, "zeros", new=forbidden_allocation),
                patch.object(torch, "ones", new=forbidden_allocation),
                patch.object(torch, "randn", new=forbidden_allocation),
                patch.object(torch.nn, "Embedding", new=forbidden_allocation),
                patch.object(torch.nn, "Linear", new=forbidden_allocation),
            ):
                with self.assertRaises(api.ContractError):
                    api.model.build_policy(schema, config, seed=89, device="cpu")

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_nonfinite_legal_logit_is_rejected_before_sampling(self):
        import torch

        api = _api(self)
        from tests.neural._sample_guard import active_guard

        with synthetic_schema() as schema:
            frame = authored_frame(schema)
            policy = _small_model(api, schema, torch)
            logits = torch.zeros((1, schema.action_space.n), dtype=torch.float32)
            logits[0, -1] = float("nan")
            output = SimpleNamespace(logits=logits, values=torch.zeros((1,), dtype=torch.float32))
            sampling = active_guard()
            before = sampling.attempted_rows
            with patch.object(policy, "forward", return_value=output):
                with self.assertRaises(api.ContractError):
                    policy.act(
                        (frame,),
                        generator=torch.Generator(device="cpu").manual_seed(89),
                    )
            self.assertEqual(sampling.attempted_rows, before)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_tensorization_uses_writable_copies_without_large_python_lists_or_buffer_warnings(self):
        import torch

        api = _api(self)
        with synthetic_schema() as schema:
            frame = authored_frame(schema, populate_relations=False)
            source_hashes = {
                key: hashlib.sha256(buffer.data).hexdigest()
                for key, buffer in frame.tensors.items()
            }
            source_objects = {id(buffer.data) for buffer in frame.tensors.values()}
            read_only_inputs: list[str] = []
            original_frombuffer = torch.frombuffer
            original_tensor = torch.tensor
            original_as_tensor = torch.as_tensor

            def safe_frombuffer(buffer, *args, **kwargs):
                view = memoryview(buffer)
                if view.readonly or id(buffer) in source_objects or id(view.obj) in source_objects:
                    read_only_inputs.append(type(buffer).__name__)
                return original_frombuffer(buffer, *args, **kwargs)

            def sequence_size(data):
                if not isinstance(data, (list, tuple)):
                    return 1
                return sum(sequence_size(item) for item in data)

            def reject_large_sequence(data, constructor, *args, **kwargs):
                count = sequence_size(data)
                if isinstance(data, (list, tuple)) and count > 256:
                    raise AssertionError(
                        f"{constructor} materialized a large Python scalar sequence ({count})"
                    )
                return constructor(data, *args, **kwargs)

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                with (
                    patch.object(torch, "frombuffer", new=safe_frombuffer),
                    patch.object(torch, "tensor", new=lambda data, *a, **kw: reject_large_sequence(
                        data, original_tensor, *a, **kw
                    )),
                    patch.object(torch, "as_tensor", new=lambda data, *a, **kw: reject_large_sequence(
                        data, original_as_tensor, *a, **kw
                    )),
                ):
                    encoded = api.features.encode_frames((frame,), schema, device="cpu")

            self.assertEqual(read_only_inputs, [])
            self.assertFalse(any("not writable" in str(item.message).lower() for item in captured))
            leaves = tensor_leaves(encoded, torch)
            self.assertTrue(leaves)
            probe = leaves[0]
            self.assertGreater(probe.numel(), 0)
            with torch.no_grad():
                first = probe.reshape(-1)[0]
                probe.reshape(-1)[0] = 0 if first.item() != 0 else 1
            self.assertEqual(
                source_hashes,
                {key: hashlib.sha256(buffer.data).hexdigest()
                 for key, buffer in frame.tensors.items()},
            )

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_known_empty_potions_and_unknown_padding_have_distinct_public_coverage(self):
        import torch

        api = _api(self)
        with synthetic_schema() as schema:
            frame = authored_frame(schema, populate_relations=False)
            encoded = api.features.encode_frames((frame,), schema, device="cpu")

            potion_coverage = encoded.known_coverage["potions"]
            card_padding_coverage = encoded.known_coverage["combat_cards"]
            self.assertEqual(potion_coverage.dtype, torch.float32)
            self.assertEqual(card_padding_coverage.dtype, torch.float32)
            self.assertEqual(tuple(potion_coverage.shape), (1, 1))
            self.assertEqual(tuple(card_padding_coverage.shape), (1, 2))
            self.assertTrue(torch.equal(potion_coverage, torch.ones_like(potion_coverage)))
            self.assertTrue(torch.equal(card_padding_coverage, torch.zeros_like(card_padding_coverage)))
            self.assertEqual(frame.tensors["potions"].data, bytes(16))
            self.assertEqual(frame.tensors["potions_known"].data, b"\x01\x01")
            self.assertEqual(frame.tensors["potions_occupied"].data, b"\x00\x00")
            self.assertEqual(frame.tensors["combat_cards_known"].data, bytes(4))
            self.assertEqual(frame.tensors["combat_cards_occupied"].data, b"\x00\x00")

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_model_global_context_consumes_known_coverage(self):
        import torch

        api = _api(self)
        with synthetic_schema() as schema:
            frame = authored_frame(schema)
            batch = api.features.encode_frames((frame,), schema, device="cpu")
            policy = _small_model(api, schema, torch)

            altered_coverage = dict(batch.known_coverage)
            probe = altered_coverage["potions"].detach().clone().requires_grad_(True)
            altered_coverage["potions"] = probe
            perturbed_batch = _replace_coverage(batch, MappingProxyType(altered_coverage))
            output = policy.forward(perturbed_batch)
            gradient = torch.autograd.grad(
                output.values.sum() + output.logits[0, -1],
                probe,
                allow_unused=True,
            )[0]
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_candidate_source_target_permutation_preserves_original_row_order(self):
        import torch

        api = _api(self)
        with synthetic_schema() as schema:
            candidates_a = (
                CandidateInput(2, "play_card", "combat_cards", 0, "enemies", 0),
                CandidateInput(9, "play_card", "combat_cards", 0, "enemies", 1),
            )
            candidates_b = (
                CandidateInput(5, "play_card", "combat_cards", 0, "enemies", 0),
                CandidateInput(7, "play_card", "combat_cards", 0, "enemies", 1),
            )
            frames = (authored_frame(schema, candidates=candidates_a),
                      authored_frame(schema, candidates=candidates_b))
            policy = _small_model(api, schema, torch)
            first_batch = api.features.encode_frames((frames[0],), schema, device="cpu")
            second_batch = api.features.encode_frames((frames[1],), schema, device="cpu")
            first = policy.forward(first_batch)
            second = policy.forward(second_batch)
            self.assertEqual(tuple(first.logits.shape), (1, 10))
            self.assertEqual(tuple(second.logits.shape), (1, 10))
            torch.testing.assert_close(first.logits[0, 2], second.logits[0, 5])
            torch.testing.assert_close(first.logits[0, 9], second.logits[0, 7])
            self.assertEqual(tuple(first.values.shape), (1,))

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_power_owner_swap_moves_features_with_the_typed_enemy_reference(self):
        import torch

        api = _api(self)
        candidates = (
            CandidateInput(2, "play_card", "combat_cards", 0, "enemies", 0),
            CandidateInput(9, "play_card", "combat_cards", 0, "enemies", 1),
        )
        with synthetic_schema() as schema:
            first_frame = authored_frame(
                schema,
                candidates=candidates,
                power_owners=(0, 1),
                identical_enemy_rows=True,
            )
            swapped_frame = authored_frame(
                schema,
                candidates=candidates,
                power_owners=(1, 0),
                identical_enemy_rows=True,
            )
            first_batch = api.features.encode_frames((first_frame,), schema, device="cpu")
            swapped_batch = api.features.encode_frames((swapped_frame,), schema, device="cpu")
            policy = _small_model(api, schema, torch)
            first = policy.forward(first_batch)
            swapped = policy.forward(swapped_batch)

            torch.testing.assert_close(first.logits[0, 2], swapped.logits[0, 9])
            torch.testing.assert_close(first.logits[0, 9], swapped.logits[0, 2])
            self.assertFalse(torch.isclose(first.logits[0, 2], first.logits[0, 9]))

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_actor_and_critic_interfaces_have_public_frames_but_no_metadata_inputs(self):
        import torch

        api = _api(self)
        with synthetic_schema() as schema:
            frame = authored_frame(schema)
            policy = _small_model(api, schema, torch)
            self.assertEqual(
                tuple(inspect.signature(api.features.encode_frames).parameters),
                ("frames", "schema", "device"),
            )
            self.assertEqual(tuple(inspect.signature(policy.forward).parameters), ("batch",))
            self.assertEqual(
                tuple(inspect.signature(policy.act).parameters),
                ("frames", "generator", "deterministic"),
            )
            self.assertEqual(tuple(inspect.signature(policy.values).parameters), ("frames",))

            poison_a = {"seed": -10**9, "visit": 0, "routing": "privileged-a"}
            rng_before = torch.get_rng_state().clone()
            value_a = policy.values((frame,))
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            poison_a.update(seed=10**9, visit=999_999, routing="privileged-b")
            value_b = policy.values((frame,))
            self.assertEqual(value_a, value_b)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_accepted_default_model_reports_parameters_frame_bytes_and_cpu_timings(self):
        import torch

        api = _api(self)
        schema, config = accepted_schema_and_config()
        frames = tuple(authored_frame(schema, populate_relations=False) for _ in range(4))
        policy = api.model.build_policy(schema, config.model, seed=1103, device="cpu")
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        self.assertLessEqual(parameter_count, min(config.model.max_parameters, 10_000_000))
        bytes_per_frame = frame_bytes(frames[0])
        self.assertGreater(bytes_per_frame, 0)

        timings: dict[str, dict[str, float]] = {}
        for batch_size in (1, 4):
            batch = api.features.encode_frames(frames[:batch_size], schema, device="cpu")
            policy.zero_grad(set_to_none=True)
            started = time.perf_counter()
            output = policy.forward(batch)
            forward_ms = (time.perf_counter() - started) * 1000.0
            legal_logits = output.logits[batch.action_mask.to(dtype=torch.bool)]
            self.assertTrue(torch.isfinite(legal_logits).all())
            loss = legal_logits.square().mean() + output.values.square().mean()
            started = time.perf_counter()
            loss.backward()
            backward_ms = (time.perf_counter() - started) * 1000.0
            self.assertTrue(all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in policy.parameters()
            ))
            timings[f"B{batch_size}"] = {
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
            }

        peak_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        self.assertLess(peak_bytes, 8 * 1024**3)
        self.assertEqual(torch.get_num_threads(), 4)
        metrics = {
            "schema_fingerprint": schema.schema_fingerprint,
            "parameter_count": parameter_count,
            "snapshot_bytes_per_frame": bytes_per_frame,
            "device": "cpu",
            "math_threads": torch.get_num_threads(),
            "forward_backward_ms": timings,
            "peak_process_bytes": peak_bytes,
            "native_resets": 0,
            "native_steps": 0,
            "policy_samples": 0,
        }
        metrics_path = os.environ.get("SPIRE_LEARN_TENSOR_METRICS_PATH")
        self.assertTrue(metrics_path, "runbook must provide a writable owned metrics path")
        Path(metrics_path).write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
