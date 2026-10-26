"""Opt-in tests for stable legal-action probabilities and sampling."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

from tests.neural._support import authored_frame, synthetic_schema


_TENSOR_GATE = "SPIRE_LEARN_RUN_NEURAL_TENSOR"


def setUpModule() -> None:
    if os.environ.get(_TENSOR_GATE) == "1":
        from tests.neural._sample_guard import activate_from_environment

        activate_from_environment()


def _api(test: unittest.TestCase) -> SimpleNamespace:
    try:
        from spire_learn.neural import distribution, features, model
        from spire_learn.neural.contracts import ContractError, ModelConfig
    except ImportError as error:
        test.fail(f"opt-in neural distribution API is unavailable: {error}")
    return SimpleNamespace(
        distribution=distribution,
        features=features,
        model=model,
        ContractError=ContractError,
        ModelConfig=ModelConfig,
    )


class NeuralDistributionTensorTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_illegal_extreme_logit_is_zero_probability_and_legal_log_probs_are_finite(self):
        import torch

        api = _api(self)
        logits = torch.tensor([[1000.0, -1000.0, -1001.0]], dtype=torch.float32)
        mask = torch.tensor([[False, True, True]], dtype=torch.bool)
        dist = api.distribution.masked_categorical(logits, mask)

        self.assertEqual(dist.probabilities[0, 0].item(), 0.0)
        self.assertTrue(torch.isneginf(dist.log_probs[0, 0]))
        self.assertTrue(torch.isfinite(dist.log_probs[0, 1:]).all())
        self.assertTrue(torch.isfinite(dist.entropy).all())

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_singleton_action_has_zero_log_probability_and_entropy(self):
        import torch

        api = _api(self)
        logits = torch.tensor([[0.0, 500.0, -500.0, 1000.0]], dtype=torch.float32)
        mask = torch.tensor([[False, False, True, False]], dtype=torch.bool)
        dist = api.distribution.masked_categorical(logits, mask)

        log_probability = dist.log_prob(torch.tensor([2], dtype=torch.int64))
        self.assertEqual(log_probability.shape, (1,))
        self.assertEqual(log_probability.item(), 0.0)
        self.assertEqual(dist.entropy.shape, (1,))
        self.assertEqual(dist.entropy[0].item(), 0.0)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_action_mask_requires_matching_shape_and_boolean_or_binary_int8(self):
        import torch

        api = _api(self)
        logits = torch.zeros((2, 4), dtype=torch.float32)
        valid_masks = (
            torch.tensor([[True, False, True, False], [False, True, False, True]]),
            torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.int8),
        )
        for mask in valid_masks:
            dist = api.distribution.masked_categorical(logits, mask)
            self.assertEqual(tuple(dist.probabilities.shape), (2, 4))
            self.assertEqual(tuple(dist.log_probs.shape), (2, 4))
            self.assertEqual(tuple(dist.entropy.shape), (2,))

        invalid_masks = (
            torch.ones((4,), dtype=torch.bool),
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones((2, 4), dtype=torch.int64),
            torch.tensor([[1, 2, 0, 0], [0, 1, 0, 1]], dtype=torch.int8),
        )
        for mask in invalid_masks:
            with self.subTest(mask_dtype=mask.dtype, mask_shape=tuple(mask.shape)):
                with self.assertRaises(api.ContractError):
                    api.distribution.masked_categorical(logits, mask)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_log_prob_uses_original_indices_and_rejects_illegal_or_out_of_range_actions(self):
        import torch

        api = _api(self)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        mask = torch.tensor([[False, True, False, True], [True, False, True, False]])
        dist = api.distribution.masked_categorical(logits, mask)

        legal = dist.log_prob(torch.tensor([3, 2], dtype=torch.int64))
        self.assertEqual(tuple(legal.shape), (2,))
        self.assertTrue(torch.isfinite(legal).all())
        for actions in (
            torch.tensor([0, 2], dtype=torch.int64),
            torch.tensor([4, 1], dtype=torch.int64),
            torch.tensor([-1, 2], dtype=torch.int64),
        ):
            with self.subTest(actions=actions.tolist()):
                with self.assertRaises(api.ContractError):
                    dist.log_prob(actions)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_empty_ongoing_mask_is_rejected_before_any_sample(self):
        import torch

        api = _api(self)
        from tests.neural._sample_guard import active_guard

        sampling = active_guard()
        before = sampling.attempted_rows
        empty_logits = torch.zeros((1, 10), dtype=torch.float32)
        empty_mask = torch.zeros((1, 10), dtype=torch.bool)
        with self.assertRaises(api.ContractError):
            api.distribution.masked_categorical(empty_logits, empty_mask)

        with synthetic_schema() as schema:
            frame = authored_frame(schema, candidates=())
            policy = api.model.build_policy(
                schema,
                api.ModelConfig(embedding_width=16, hidden_width=64, max_parameters=10_000_000),
                seed=71,
                device="cpu",
            )
            with self.assertRaises(api.ContractError):
                policy.act((frame,), generator=torch.Generator().manual_seed(71))
        self.assertEqual(sampling.attempted_rows, before)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_sampling_returns_original_legal_indices_and_uses_only_its_generator(self):
        import torch

        api = _api(self)
        from tests.neural._sample_guard import active_guard

        logits = torch.tensor([
            [100.0, -1.0, 1.0, -100.0],
            [1.0, 0.0, -100.0, 100.0],
        ], dtype=torch.float32)
        mask = torch.tensor([
            [False, True, True, False],
            [True, True, False, False],
        ], dtype=torch.int8)
        dist = api.distribution.masked_categorical(logits, mask)
        guard = active_guard()
        attempted_before = guard.attempted_rows
        completed_before = guard.completed_rows
        first = dist.sample(generator=torch.Generator(device="cpu").manual_seed(9901))
        second = dist.sample(generator=torch.Generator(device="cpu").manual_seed(9901))

        self.assertEqual(first.dtype, torch.int64)
        self.assertEqual(tuple(first.shape), (2,))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(mask.gather(1, first.unsqueeze(1)).all())
        self.assertEqual(guard.attempted_rows - attempted_before, 4)
        self.assertEqual(guard.completed_rows - completed_before, 4)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_deterministic_mode_breaks_equal_logit_ties_by_smallest_original_index(self):
        import torch

        api = _api(self)
        logits = torch.tensor([[0.0, 5.0, 5.0, 100.0]], dtype=torch.float32)
        mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)
        dist = api.distribution.masked_categorical(logits, mask)
        mode = dist.mode()
        self.assertEqual(mode.dtype, torch.int64)
        self.assertEqual(tuple(mode.shape), (1,))
        self.assertEqual(mode.item(), 1)

    @unittest.skipUnless(os.environ.get(_TENSOR_GATE) == "1", "set SPIRE_LEARN_RUN_NEURAL_TENSOR=1")
    def test_probability_entropy_log_prob_and_mode_queries_do_not_sample(self):
        import torch

        api = _api(self)
        from tests.neural._sample_guard import active_guard

        logits = torch.tensor([[0.0, 1.0, 1000.0], [2.0, 0.0, 1000.0]])
        mask = torch.tensor([[True, True, False], [True, True, False]])
        dist = api.distribution.masked_categorical(logits, mask)
        guard = active_guard()
        attempted_before = guard.attempted_rows
        completed_before = guard.completed_rows

        _ = dist.probabilities
        _ = dist.entropy
        _ = dist.log_prob(torch.tensor([1, 0], dtype=torch.int64))
        _ = dist.mode()

        self.assertEqual(guard.attempted_rows, attempted_before)
        self.assertEqual(guard.completed_rows, completed_before)


if __name__ == "__main__":
    unittest.main()
