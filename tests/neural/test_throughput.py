"""Opt-in, no-gameplay CPU equivalence checks for raw neural data reuse.

The default discovery path skips before importing Torch. The explicit runtime
suite uses at most ten actual Adam calls and forbids all actor draws.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

from tests.neural._support import CandidateInput, authored_frame, synthetic_schema


_GATE = "SPIRE_LEARN_RUN_NEURAL_THROUGHPUT"
_RUNTIME = None
_ADAM_CALLS = 0


def setUpModule():
    if os.environ.get(_GATE) != "1":
        return
    import torch

    global _RUNTIME, _ADAM_CALLS
    _ADAM_CALLS = 0
    _RUNTIME = ExitStack()
    original_step = torch.optim.Adam.step

    def bounded_step(optimizer, *args, **kwargs):
        global _ADAM_CALLS
        if _ADAM_CALLS >= 16:
            raise AssertionError("throughput suite exceeded its sixteen-Adam-call cap")
        _ADAM_CALLS += 1
        return original_step(optimizer, *args, **kwargs)

    _RUNTIME.enter_context(patch.object(torch.optim.Adam, "step", bounded_step))
    _RUNTIME.enter_context(patch.object(
        torch, "multinomial", side_effect=AssertionError("actor draws are forbidden")
    ))


def tearDownModule():
    if _RUNTIME is not None:
        _RUNTIME.close()
        print(f"THROUGHPUT_TEST_ADAM_CALLS={_ADAM_CALLS}; ACTOR_DRAWS=0")
        if _ADAM_CALLS > 16:
            raise AssertionError("throughput suite Adam budget exceeded")


def _legacy_encode(frames, schema, *, device):
    """Independent former per-frame bytearray/clone/transfer/stack tensorizer."""
    import torch
    from spire_learn.neural.features import EncodedBatch
    from spire_learn.neural.schema import validate_frame

    frames = tuple(frames)
    for frame in frames:
        validate_frame(frame, schema)
    raw = {}
    for key, spec in schema.tensors.items():
        dtype = {"int8": torch.int8, "int64": torch.int64}[spec.dtype]
        rows = []
        for frame in frames:
            writable = bytearray(frame.tensors[key].data)
            rows.append(torch.frombuffer(writable, dtype=dtype).reshape(spec.shape).clone().to(device=device))
        raw[key] = torch.stack(rows, dim=0)
    values, known, occupied, coverage = {}, {}, {}, {}
    for key, spec in schema.tensors.items():
        if not spec.columns:
            continue
        rows = spec.shape[0] if len(spec.shape) == 2 else 1
        shape = (len(frames), rows, len(spec.columns))
        values[key] = raw[key].reshape(shape)
        known[key] = raw[spec.known_key].reshape(shape).to(torch.bool)
        if spec.occupied_key is not None:
            occupied[key] = raw[spec.occupied_key].to(torch.bool)
            coverage[key] = known[key].to(torch.float32).mean(dim=1)
        else:
            coverage[key] = known[key][:, 0, :].to(torch.float32)
    return EncodedBatch(schema.schema_fingerprint, values, known, occupied,
                        raw["action_mask"].to(torch.bool), coverage)


def _tensors(batch):
    return {
        **{f"{group}/{key}": tensor for group in ("values", "known", "occupied", "known_coverage")
           for key, tensor in getattr(batch, group).items()},
        "action_mask": batch.action_mask,
    }


def _clone_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_clone_tree(item, torch) for item in value)
    return value


@unittest.skipUnless(os.environ.get(_GATE) == "1", f"set {_GATE}=1")
class NeuralThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No driver/facade/native API construction is needed to authenticate
        # the checked-in public descriptor or parse the learner config.
        import torch
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.schema import parse_common_descriptor

        root = Path(__file__).resolve().parents[2]
        cls.torch = torch
        cls.schema = parse_common_descriptor(json.loads(
            (root / "tests/fixtures/neural/common-afb51f5.json").read_text()
        ), revision="afb51f5f222b4c4331ed34067c5eebeda606db91")
        cls.config = load_neural_config(root / "configs/neural/gym-20261102/standalone.toml")
        cls.candidates = (
            CandidateInput(2, "play_card", "combat_cards", 0, "enemies", 0),
            CandidateInput(9, "play_card", "combat_cards", 0, "enemies", 1),
        )

    def setUp(self):
        from spire_learn.neural import features
        self.features = features

    def _frames(self, schema):
        return tuple(authored_frame(schema, candidates=self.candidates, power_owners=owners)
                     for owners in ((0, 1), (1, 0), (1, 1), (0, 0)))

    def _assert_tree_equal(self, left, right):
        if self.torch.is_tensor(left):
            self.assertTrue(self.torch.is_tensor(right))
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.device, right.device)
            self.assertEqual(left.shape, right.shape)
            self.assertTrue(self.torch.equal(left, right))
        elif isinstance(left, Mapping):
            self.assertEqual(tuple(left), tuple(right))
            for key in left:
                self._assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for item, other in zip(left, right, strict=True):
                self._assert_tree_equal(item, other)
        else:
            self.assertEqual(left, right)

    def _assert_batches_equal(self, left, right):
        self.assertEqual(left.schema_fingerprint, right.schema_fingerprint)
        self.assertEqual(left.batch_size, right.batch_size)
        self._assert_tree_equal(_tensors(left), _tensors(right))

    def test_batch_packing_matches_legacy_and_transfers_once_per_key(self):
        # Catches reordered bytes, dtype/mask/coverage drift and per-row transfers.
        with synthetic_schema() as legacy:
            for schema in (self.schema, legacy):
                frames = self._frames(schema)
                for count in (1, 4):
                    with self.subTest(fingerprint=schema.schema_fingerprint, batch=count):
                        expected = _legacy_encode(frames[:count], schema, device="cpu")
                        transfers = []
                        original_to = self.torch.Tensor.to

                        def record_to(tensor, *args, **kwargs):
                            if "device" in kwargs:
                                transfers.append((tensor.dtype, tuple(tensor.shape)))
                            return original_to(tensor, *args, **kwargs)

                        with warnings.catch_warnings(record=True) as caught, patch.object(
                            self.torch.Tensor, "to", record_to
                        ):
                            warnings.simplefilter("always")
                            actual = self.features.encode_frames(frames[:count], schema, device="cpu")
                        self._assert_batches_equal(actual, expected)
                        self.assertEqual(caught, [])
                        self.assertEqual(len(transfers), len(schema.tensors))
                        self.assertEqual([shape for _, shape in transfers],
                                         [(count, *spec.shape) for spec in schema.tensors.values()])
                        self.assertTrue(all(tensor.is_contiguous() for tensor in _tensors(actual).values()))
                        self.assertTrue(all(not tensor.requires_grad for tensor in _tensors(actual).values()))
                        self.assertEqual(actual.action_mask[0].nonzero().flatten().tolist(), [2, 9])

    def test_owned_tensors_survive_gc_and_do_not_alias_frames_or_other_encodes(self):
        # Catches dangling frombuffer storage or reuse of a mutable packing arena.
        frames = self._frames(self.schema)
        original_bytes = tuple(tuple(buffer.data for buffer in frame.tensors.values()) for frame in frames)
        first = self.features.encode_frames(frames, self.schema, device="cpu")
        second = self.features.encode_frames(frames, self.schema, device="cpu")
        expected = _legacy_encode(frames, self.schema, device="cpu")
        del frames
        gc.collect()
        self._assert_batches_equal(first, expected)
        for key, tensor in _tensors(first).items():
            tensor.reshape(-1)[0] = ~tensor.reshape(-1)[0] if tensor.dtype == self.torch.bool else tensor.reshape(-1)[0] + 1
            self.assertFalse(self.torch.equal(tensor, _tensors(second)[key]), key)
        self._assert_batches_equal(second, expected)
        regenerated = self._frames(self.schema)
        self.assertEqual(original_bytes,
                         tuple(tuple(buffer.data for buffer in frame.tensors.values()) for frame in regenerated))

    def test_select_rows_reorders_duplicates_and_owns_every_selected_tensor(self):
        # Catches source aliasing and lost metadata/masks in indexed minibatches.
        frames = self._frames(self.schema)
        source = self.features.encode_frames(frames, self.schema, device="cpu")
        indices = self.torch.tensor([3, 1, 3, 0], dtype=self.torch.int64)
        selected = source.select_rows(indices)
        expected = _legacy_encode(tuple(frames[i] for i in (3, 1, 3, 0)), self.schema, device="cpu")
        self._assert_batches_equal(selected, expected)
        before = {key: value.clone() for key, value in _tensors(source).items()}
        for tensor in _tensors(selected).values():
            tensor.reshape(-1)[0] = ~tensor.reshape(-1)[0] if tensor.dtype == self.torch.bool else tensor.reshape(-1)[0] + 1
        self._assert_tree_equal(_tensors(source), before)
        del source, frames, indices
        gc.collect()
        # Each duplicate must also be independent from the other selected row.
        for key, tensor in _tensors(selected).items():
            self._assert_tree_equal(tensor[2], _tensors(expected)[key][2])

    def test_select_rows_rejects_malformed_or_out_of_range_before_indexing(self):
        from spire_learn.contracts import ContractError
        batch = self.features.encode_frames(self._frames(self.schema), self.schema, device="cpu")
        torch = self.torch
        invalid = (None, [0], (), torch.tensor(0), torch.tensor([[0]]),
                   torch.tensor([0], dtype=torch.int32), torch.tensor([0.0]),
                   torch.tensor([True]), torch.tensor([], dtype=torch.int64),
                   torch.tensor([-1], dtype=torch.int64), torch.tensor([4], dtype=torch.int64))
        for indices in invalid:
            with self.subTest(indices=repr(indices)), self.assertRaises(ContractError):
                batch.select_rows(indices)

    def _policy_and_rollout(self, prefix="throughput"):
        from spire_learn.neural.contracts import CumulativeCounters, PolicyChoice, Rollout, StepRecord, TaskSignal
        from spire_learn.neural.distribution import masked_categorical
        from spire_learn.neural.model import build_policy

        policy = build_policy(self.schema, self.config.model, seed=2207, device="cpu")
        ppo_config = replace(self.config.ppo, epochs=2, minibatch_rows=2)
        encoded = json.dumps(asdict(ppo_config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        identity = self.config.accepted_identities[0]
        identity = replace(identity, config_hashes={**identity.config_hashes, "ppo": hashlib.sha256(encoded).hexdigest()})
        frames = self._frames(self.schema)
        batch = _legacy_encode(frames, self.schema, device="cpu")
        with self.torch.no_grad():
            output = policy(batch)
            actions = self.torch.tensor([2, 9, 2, 9], dtype=self.torch.int64)
            logp = masked_categorical(output.logits, batch.action_mask).log_prob(actions).tolist()
        records = tuple(StepRecord(
            stream_id=f"{prefix}-stream-{i}", task_id=f"{prefix}-task-{i}", decision_id=f"{prefix}-decision-{i}",
            source=frame, successor=frame, native_reward=0.0, terminated=False, truncated=False,
            task_signal=TaskSignal(float(i - 1), True, False, False),
            choice=PolicyChoice(int(actions[i]), float(logp[i]), float(output.values[i]), 0),
            actor_training_mask=True, next_value=float(output.values[i]),
        ) for i, frame in enumerate(frames))
        rollout = Rollout("standalone", "train", identity, records, CumulativeCounters(4, 4, 0, 4, 0, 0))
        return policy, rollout, ppo_config

    def _update(self, policy, optimizer, rollout, config, generator):
        from spire_learn.experiments.budget import Budget
        from spire_learn.neural.ppo import ppo_update
        return ppo_update(policy, optimizer, rollout, config, shuffle_generator=generator,
                          budget=Budget(30.0, lambda: 0.0))

    def test_ppo_encodes_once_and_matches_old_minibatch_outputs_gradients_adam_and_rng(self):
        # Catches cached trainable outputs, shuffle changes and repeated encoding.
        self.assertTrue(callable(getattr(self.features.EncodedBatch, "select_rows", None)))
        results = []
        global_rng = self.torch.get_rng_state().clone()
        for legacy in (True, False):
            prefix = "legacy" if legacy else "optimized"
            policy, rollout, config = self._policy_and_rollout(prefix)
            optimizer = self.torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=config.adam_eps)
            generator = self.torch.Generator(device="cpu").manual_seed(91)
            frames = tuple(record.source for record in rollout.records)
            snapshots, outputs = [], []
            original_forward, original_step = policy.forward, optimizer.step

            def record_forward(batch):
                result = original_forward(batch)
                outputs.append((result.logits.detach().clone(), result.values.detach().clone()))
                return result

            def record_step(*args, **kwargs):
                gradients = tuple(None if p.grad is None else p.grad.detach().clone() for p in policy.parameters())
                result = original_step(*args, **kwargs)
                snapshots.append((gradients, _clone_tree(policy.state_dict(), self.torch),
                                  _clone_tree(optimizer.state_dict(), self.torch)))
                return result

            def old_select(_batch, indices):
                return _legacy_encode(tuple(frames[i] for i in indices.tolist()), self.schema, device="cpu")

            with ExitStack() as stack:
                stack.enter_context(patch.object(policy, "forward", record_forward))
                stack.enter_context(patch.object(optimizer, "step", record_step))
                if legacy:
                    stack.enter_context(patch.object(self.features, "encode_frames", _legacy_encode))
                    stack.enter_context(patch.object(self.features.EncodedBatch, "select_rows", old_select))
                else:
                    encode = stack.enter_context(patch.object(self.features, "encode_frames", wraps=self.features.encode_frames))
                metrics = self._update(policy, optimizer, rollout, config, generator)
                if not legacy:
                    self.assertEqual(encode.call_count, 1)
            self.assertEqual((metrics.attempted_steps, metrics.accepted_steps, metrics.consumed_decisions), (4, 4, 4))
            self.assertEqual((metrics.admitted_behavior_policy_version, metrics.resulting_behavior_policy_version), (0, 1))
            self.assertIsNone(metrics.stop_reason)
            self.assertEqual(len(snapshots), 4)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(any(g is not None and self.torch.count_nonzero(g).item()
                                for g in snapshots[0][0]))
            self.assertTrue(all(not self.torch.equal(outputs[1][1], item[1]) for item in outputs[2:]))
            results.append((metrics, outputs, snapshots, generator.get_state(),
                            policy._behavior_policy_version,
                            {key.replace(prefix, "throughput") for key in policy._consumed_decision_ids}))
        self._assert_tree_equal(results[0], results[1])
        self._assert_tree_equal(self.torch.get_rng_state(), global_rng)

    def test_minibatch_failure_and_control_keep_one_accepted_prefix_and_consumption(self):
        from spire_learn.contracts import ContractError
        from spire_learn.neural.ppo import UpdateFault

        for control in (False, True):
            with self.subTest(control=control):
                policy, rollout, config = self._policy_and_rollout()
                optimizer = self.torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=config.adam_eps)
                generator = self.torch.Generator(device="cpu").manual_seed(91)
                original_forward = policy.forward
                calls = 0
                injected = KeyboardInterrupt("authored control") if control else RuntimeError("authored minibatch fault")

                def fail_second_minibatch(batch):
                    nonlocal calls
                    calls += 1
                    if calls == 3:  # admission, accepted minibatch, then failure
                        raise injected
                    return original_forward(batch)

                with patch.object(policy, "forward", fail_second_minibatch):
                    with self.assertRaises(KeyboardInterrupt if control else UpdateFault) as caught:
                        self._update(policy, optimizer, rollout, config, generator)
                error = caught.exception
                if control:
                    self.assertIs(error, injected)
                    notes = " ".join(error.__notes__)
                    for text in ("phase='forward-loss'", "attempted_steps=1", "accepted_steps=1",
                                 "uncertain_steps=0", "consumed_decisions=4",
                                 "admitted_behavior_policy_version=0", "resulting_behavior_policy_version=1"):
                        self.assertIn(text, notes)
                else:
                    self.assertIs(error.__cause__, injected)
                    self.assertEqual(error.phase, "forward-loss")
                    self.assertEqual((error.attempted_steps, error.accepted_steps, error.uncertain_steps), (1, 1, 0))
                    self.assertEqual((error.admitted_behavior_policy_version, error.resulting_behavior_policy_version,
                                      error.consumed_decisions), (0, 1, 4))
                self.assertEqual(policy._behavior_policy_version, 1)
                self.assertEqual(len(policy._consumed_decision_ids), 4)
                self.assertTrue(all(int(state["step"].item()) == 1 for state in optimizer.state.values()))
                with self.assertRaises(ContractError):
                    self._update(policy, optimizer, rollout, config, generator)


if __name__ == "__main__":
    unittest.main()
