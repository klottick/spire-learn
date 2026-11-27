"""Distillation trainer: loader exactness, label isolation, masked loss, microbatching, batch order, checkpoints.

Needs numpy and torch (runtime container only) and skips cleanly on the host. Run it in the container
with ``gated_tests_entry.py --distill-train``. Shards are written with the real compact fixture schema.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from spire_learn.neural.config import (
    DISTILL_OBJECTIVE_ID, STANDARD_LEARNER_IDENTITY_IDS, DistillConfig, DistillHyperparameters,
    distill_config_hash, model_config_hash,
)
from spire_learn.neural.contracts import ModelConfig, NeuralIdentity

TINY = ModelConfig(embedding_width=4, hidden_width=8, max_parameters=10_000_000, value_heads=3)
MAPPING = "spire-learn-standard-combat-compact-v1@fixture+test"
NAN = math.nan


def _load_modules() -> dict:
    try:
        import numpy
        import torch
    except ImportError:
        raise unittest.SkipTest("numpy and torch are only importable in the runtime container")
    from spire_learn.neural import distill_data, distill_train
    from tests.neural import test_distill_data as shards

    shards._load_numpy_modules()
    return {"np": numpy, "torch": torch, "shards": shards, "data": distill_data, "train": distill_train}


def identity(hyper: DistillHyperparameters, model: ModelConfig = TINY) -> NeuralIdentity:
    model_id, features_id, action_id = STANDARD_LEARNER_IDENTITY_IDS
    return NeuralIdentity(model_id=model_id, features_id=features_id, action_id=action_id,
                          objective_id=DISTILL_OBJECTIVE_ID, accepted_raw_mode_mappings={"standard-standalone": MAPPING},
                          config_hashes={"model": model_config_hash(model), "distill": distill_config_hash(hyper)})


def distill_config(hyper: DistillHyperparameters, *, seed=3, eval_every=2) -> DistillConfig:
    return DistillConfig(
        source=Path("fixture.toml"), config_sha256="0" * 64, mode="standard-standalone",
        environment_revision="fixture", standard_view_layout="compact",
        standard_view_preset="standard-combat-compact-v1", standard_view_fingerprint="fixture",
        output_root=Path("."), device="cpu", runtime_gpu_uuid=None, seed=seed, eval_every=eval_every,
        model=TINY, distill=hyper, identity=identity(hyper))


class FixedPolicy:
    """A stub student: the same per-action logits for every row, constant value heads."""

    def __init__(self, torch, preference: dict[int, float], width: int, *, values=0.5, aux=(0.25, -0.5)):
        from spire_learn.neural.model import PolicyOutput

        self.torch, self.output_type = torch, PolicyOutput
        self.logits = torch.zeros(width)
        for action, logit in preference.items():
            self.logits[action] = logit
        self.value, self.aux = values, aux
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, batch):
        rows = batch.action_mask.shape[0]
        torch = self.torch
        return self.output_type(self.logits.expand(rows, -1).clone(), torch.full((rows,), self.value),
                                torch.tensor([list(self.aux)] * rows))


class GateSpy:
    """A gate that records its stages and emits one wait event per stage through its journal."""

    def __init__(self):
        self.stages, self.waited_seconds, self._journal = [], 0.0, None
        self.events = [{"event": "gpu_wait", "stage": "qualify"}]

    def set_journal(self, journal):
        self._journal = journal
        if journal is not None:
            for event in self.events:
                journal(event)

    def wait(self, stage):
        self.stages.append(stage)
        event = {"event": "gpu_wait", "stage": stage}
        self.events.append(event)
        if self._journal is not None:
            self._journal(event)

    def release(self):
        return None


def _grads(policy, torch) -> list:
    return [torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.clone()
            for parameter in policy.parameters()]


class DistillTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = _load_modules()
        for name, module in modules.items():
            setattr(cls, name, module)
        cls.torch.set_num_threads(1)
        cls.schema = cls.shards._schema()
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        build = cls.shards.build_frame
        cls.frames = [
            build(cls.schema, 1, {"permanent_cards": [0, 1, 2], "actions": [0, 3, 4, 9], "potions": [1, 3],
                                  "orbs": [0, 2, 3]}, {"potions": [0, 4], "orbs": [1, 5, 9]}),
            build(cls.schema, 2, {"combat_cards": [5, 6, 40], "enemies": [1, 4], "actions": [2],
                                  "relics": [0, 7]}, {"potions": [2], "orbs": [0, 9]}),
            build(cls.schema, 3, {"actions": [1000, 1023], "powers": [3, 100], "known_draw": [9]}, {}),
            build(cls.schema, 4, {"actions": [0, 1, 2], "enemies": [0], "relics": [1, 2]}, {"potions": [1]}),
            build(cls.schema, 5, {"actions": [5, 6], "combat_cards": [0, 1], "powers": [0]}, {}),
            build(cls.schema, 6, {"actions": [7, 8, 11], "relics": [3], "permanent_cards": [4]}, {"orbs": [2]}),
        ]
        # kind per frame: a searched root (policy and value), a single legal action, an all-way tie,
        # a refused root, a turn-cap prefix (policy only) and a searched root again.
        cls.kinds = ["searched", "single", "tie", "refused", "prefix", "searched"]
        cls.shard_a = cls.write("train", 1000, list(range(6)))
        cls.shard_b = cls.write("train", 1001, [5, 0, 2, 4], run="run-b")
        cls.validation = cls.write("validation", 1000, [0, 2, 4, 5, 3], run="run-v")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def labels(cls, index: int, kind: str, *, run: str):
        np = cls.np
        frame = cls.frames[index]
        legal = [int(action) for action in np.flatnonzero(np.frombuffer(frame.tensors["action_mask"].data, np.int8))]
        values = None
        choice, root, policy_ok, value_ok = legal[0], "searched", False, True
        if kind in ("searched", "prefix"):
            values = [0.1 * ((action * 7 + index) % 5) - 0.2 for action in legal]
            if len(set(values)) == 1:
                values[-1] += 0.5
            choice, policy_ok = legal[int(np.argmax(values))], True
        elif kind == "tie":
            values = [0.25] * len(legal)
        elif kind == "single":
            root = "single_legal"
        elif kind == "refused":
            root, choice = "refused", -1
        value_ok = kind not in ("prefix", "refused")
        outcome = ("win", "death", "escaped")[index % 3] if value_ok else "none"
        return cls.data.DecisionLabels(
            legal=legal, values=values, samples=32, choice=choice, played=legal[0], root=root,
            outcome=outcome, hp_kept=0.1 * index if value_ok else NAN,
            realised_v2=0.2 - 0.05 * index if value_ok else NAN, value_ok=value_ok, policy_ok=policy_ok,
            research_seed=index, case=(1000, index), run_id=f"{run}-{index % 2}", turn=index)

    @classmethod
    def write(cls, series: str, index: int, frames: list[int], *, run="run-a"):
        path = cls.root / f"{series}-{index}-{run}"
        writer = cls.data.ShardWriter(path, cls.schema, {"fixture": series})
        for frame in frames:
            writer.add(cls.frames[frame], cls.labels(frame, cls.kinds[frame], run=run))
        manifest = writer.close()
        return cls.train.ShardRef(series, index, path, manifest["rows"], "f" * 64)

    def shard_set(self, *refs):
        shards = self.train.ShardSet(refs or (self.shard_a, self.shard_b), self.schema)
        self.addCleanup(shards.close)
        return shards

    def policy(self, seed=11):
        from spire_learn.neural.model import build_policy

        model_id, features_id, _ = STANDARD_LEARNER_IDENTITY_IDS
        return build_policy(self.schema, TINY, seed=seed, device="cpu", objective_id=DISTILL_OBJECTIVE_ID,
                            model_id=model_id, features_id=features_id)

    def items(self, pairs):
        return self.np.asarray(pairs, dtype=self.np.int64)

    # --- loader ------------------------------------------------------------------------------

    def test_loader_equals_encode_frames(self):
        from spire_learn.neural.distill_data import ShardReader, expand_decision
        from spire_learn.neural.features import encode_frames

        torch = self.torch
        shards = self.shard_set()
        pairs = [(1, 2), (0, 0), (0, 5), (1, 0), (0, 2), (0, 0), (0, 3), (1, 3), (0, 1), (0, 4)]
        for subset in (pairs, pairs[:1], [(0, 1)], [(0, 2), (1, 2)]):
            with self.subTest(subset=subset):
                batch = self.train.collate_observations(shards, self.items(subset))
                frames = []
                for shard, row in subset:
                    with ShardReader(shards.refs[shard].path, self.schema) as reader:
                        frames.append(expand_decision(reader, row, self.schema))
                expected = encode_frames(frames, self.schema, device="cpu", trim_padding=True)
                for name in ("values", "known", "occupied", "known_coverage", "_known_padding"):
                    actual, wanted = getattr(batch, name), getattr(expected, name)
                    self.assertEqual(list(actual), list(wanted), name)
                    for key in wanted:
                        self.assertEqual(actual[key].dtype, wanted[key].dtype, (name, key))
                        self.assertTrue(torch.equal(actual[key], wanted[key]), (name, key))
                self.assertTrue(torch.equal(batch.action_mask, expected.action_mask))
                self.assertEqual(batch._row_padding, expected._row_padding)
                self.assertEqual(batch.schema_fingerprint, expected.schema_fingerprint)
                policy = self.policy()
                with torch.no_grad():
                    ours, theirs = policy(batch), policy(expected)
                self.assertTrue(torch.equal(ours.logits, theirs.logits))
                self.assertTrue(torch.equal(ours.aux_values, theirs.aux_values))
        self.assertLessEqual(shards.peak_open, self.train.OPEN_SHARD_LIMIT)
        # The open-shard limit unmaps the least recently used shard.
        limited = self.train.ShardSet((self.shard_a, self.shard_b), self.schema, open_limit=1)
        self.addCleanup(limited.close)
        self.train.collate_observations(limited, self.items(pairs))
        self.assertEqual((limited.open_shards, limited.peak_open), (1, 1))

    def test_loader_reads_no_label_file(self):
        from spire_learn.neural.distill_data import ShardReader

        shards = self.shard_set()
        original = ShardReader.array
        seen: list[str] = []

        def spy(reader, name):
            seen.append(name)
            return original(reader, name)

        with mock.patch.object(ShardReader, "array", spy):
            batch = self.train.collate_observations(shards, self.items([(0, 0), (1, 1), (0, 3)]))
        self.assertTrue(seen)
        self.assertFalse([name for name in seen if name.startswith("labels.")])
        # Same observations under different labels collate to the same batch.
        relabelled = self.write("train", 1002, list(range(6)), run="other-run")
        other = self.shard_set(relabelled)
        again = self.train.collate_observations(other, self.items([(0, 0), (0, 4), (0, 3)]))
        first = self.train.collate_observations(shards, self.items([(0, 0), (0, 4), (0, 3)]))
        for name in ("values", "known", "occupied", "known_coverage"):
            for key, tensor in getattr(first, name).items():
                self.assertTrue(self.torch.equal(tensor, getattr(again, name)[key]))
        self.assertIsNotNone(batch)

    # --- targets and loss --------------------------------------------------------------------

    def test_soft_target_and_policy_mask(self):
        torch, np = self.torch, self.np
        shards = self.shard_set()
        items = self.items([(0, row) for row in range(6)])
        batch = self.train.collate_observations(shards, items)
        tau = 0.1
        targets = self.train.batch_targets(shards, items, batch.action_mask, tau=tau)
        # Policy rows: searched and the turn-cap prefix; never the tie, single or refused roots.
        self.assertEqual(targets.policy_index.tolist(), [0, 4, 5])
        # Value rows: everything but the prefix and the refused root.
        self.assertEqual(targets.value_index.tolist(), [0, 1, 2, 5])
        # Regret rows: searched roots that are not all-way ties (the prefix counts, the tie does not).
        self.assertEqual(targets.regret_index.tolist(), [0, 4, 5])
        self.assertTrue(torch.isfinite(targets.policy_target).all())
        self.assertTrue(torch.isfinite(targets.value_target).all())
        mask = batch.action_mask
        for slot, position in enumerate(targets.policy_index.tolist()):
            labels = self.labels(position, self.kinds[position], run="run-a")
            scaled = np.asarray(labels.values) / tau
            expected = np.exp(scaled - scaled.max())
            expected /= expected.sum()
            row = targets.policy_target[slot]
            self.assertTrue(torch.equal(row[~mask[position]], torch.zeros_like(row[~mask[position]])))
            np.testing.assert_allclose(row[mask[position]].numpy(), expected, rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(float(row.sum()), 1.0, places=6)
        outcome = targets.value_target[:, 0].tolist()
        self.assertEqual(outcome, [1.0, -1.0, 0.0, 0.0])  # win, death, escaped by frame index % 3
        # The policy term is the cross-entropy to the soft target over legal actions only.
        policy = self.policy()
        output = policy(batch)
        terms = self.train.loss_terms(output, mask, targets)
        manual = 0.0
        for slot, position in enumerate(targets.policy_index.tolist()):
            legal = mask[position]
            log_probs = torch.log_softmax(output.logits[position][legal], dim=-1)
            manual += -float((targets.policy_target[slot][legal] * log_probs).sum())
        self.assertAlmostEqual(float(terms.policy), manual / 3, places=5)
        predicted = torch.cat((output.values.unsqueeze(-1), output.aux_values), dim=-1)[targets.value_index]
        for head in range(3):
            self.assertAlmostEqual(float(terms.values[head]),
                                   float((predicted[:, head] - targets.value_target[:, head]).square().mean()), places=5)
        self.assertAlmostEqual(float(terms.total), float(terms.policy) + sum(float(v) for v in terms.values) / 3,
                               places=5)
        # Changing the logit of an illegal action never changes the loss.
        logits = output.logits.detach().clone()
        logits[~mask] = 1e6
        changed = self.train.loss_terms(type(output)(logits, output.values, output.aux_values), mask, targets)
        self.assertAlmostEqual(float(changed.policy), float(terms.policy), places=5)

    def test_masked_nan_rows_give_finite_zero_grads(self):
        torch = self.torch
        shards = self.shard_set()
        # Row 3 is a refused root: NaN teacher values and NaN value targets, so it selects nothing.
        policy = self.policy()
        masked = self.items([(0, 3), (0, 3)])
        batch = self.train.collate_observations(shards, masked)
        targets = self.train.batch_targets(shards, masked, batch.action_mask, tau=0.1)
        self.assertEqual((targets.policy_rows, targets.value_rows), (0, 0))
        terms = self.train.loss_terms(policy(batch), batch.action_mask, targets)
        terms.total.backward()
        self.assertEqual(float(terms.total), 0.0)
        grads = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        self.assertTrue(grads)
        for grad in grads:
            self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

        def gradients(pairs):
            policy = self.policy()
            items = self.items(pairs)
            batch = self.train.collate_observations(shards, items)
            targets = self.train.batch_targets(shards, items, batch.action_mask, tau=0.1)
            self.train.loss_terms(policy(batch), batch.action_mask, targets).total.backward()
            return _grads(policy, torch)

        # Masked rows (refused: NaN everywhere; single legal and tie: NaN or tied values, value terms
        # only; prefix: NaN value targets, policy only) leave finite gradients, and adding refused rows
        # changes nothing.
        mixed = gradients([(0, 3), (0, 0), (0, 4), (0, 5), (0, 3), (0, 1), (0, 2)])
        without_refused = gradients([(0, 0), (0, 4), (0, 5), (0, 1), (0, 2)])
        for left, right in zip(without_refused, mixed):
            self.assertTrue(torch.isfinite(right).all())
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)

    def test_microbatch_gradient_equal(self):
        torch = self.torch
        shards = self.shard_set()
        items = self.items([(0, 0), (1, 1), (0, 3), (0, 4), (1, 0), (0, 2), (0, 5)])

        class NoStep:
            def __init__(self, policy):
                self.policy = policy

            def zero_grad(self, set_to_none=True):
                self.policy.zero_grad(set_to_none=set_to_none)

            def step(self):
                return None

        results = {}
        for microbatch in (7, 3, 1):
            policy = self.policy()
            metrics = self.train.train_step(policy, NoStep(policy), shards, items, tau=0.2, microbatch_rows=microbatch)
            results[microbatch] = (metrics, _grads(policy, torch))
        full_metrics, full = results[7]
        for microbatch in (3, 1):
            metrics, grads = results[microbatch]
            for left, right in zip(full, grads):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
            for key in ("policy_loss", "outcome_loss", "hp_kept_loss", "realised_v2_loss", "grad_norm"):
                self.assertAlmostEqual(metrics[key], full_metrics[key], places=5)
            self.assertEqual((metrics["policy_rows"], metrics["value_rows"]),
                             (full_metrics["policy_rows"], full_metrics["value_rows"]))

    # --- batch order -------------------------------------------------------------------------

    def test_same_seed_same_batches(self):
        np = self.np
        counts = (5, 0, 3, 7, 2, 4)

        def batches(seed, count=12):
            order = self.train.BatchOrder(counts, seed=seed, batch_rows=4, window=2)
            return [order.next_batch().tolist() for _ in range(count)], order

        first, order = batches(9)
        self.assertEqual(first, batches(9)[0])
        self.assertNotEqual(first, batches(10)[0])
        flat = [tuple(item) for batch in first for item in batch]
        total = sum(counts)
        # Each pass visits every row exactly once; batches run across pass boundaries.
        self.assertEqual(sorted(flat[:total]), sorted((shard, row) for shard, count in enumerate(counts)
                                                      for row in range(count)))
        self.assertEqual(sorted(flat[total:2 * total]), sorted(flat[:total]))
        self.assertNotEqual(flat[:total], flat[total:2 * total])
        self.assertGreaterEqual(order.passes, 2)
        # The order lives in the dedicated minibatch domain: the global torch RNG is untouched.
        state = self.torch.random.get_rng_state()
        batches(9, 3)
        self.assertTrue(self.torch.equal(state, self.torch.random.get_rng_state()))
        self.assertTrue(np.all(np.asarray(first) >= 0))

    # --- validation metrics ------------------------------------------------------------------

    def test_validation_metrics_by_hand(self):
        torch, np = self.torch, self.np
        # Every frame: 0 searched with a partial tie (4 and 9 at 0.1; the teacher chose 4),
        # 1 single legal, 2 all-way tie, 3 refused, 4 turn-cap prefix, 5 searched.
        ref = self.write("validation", 1001, list(range(6)), run="run-m")
        shards = self.shard_set(ref)
        width = self.frames[0].tensors["action_mask"].shape[0]
        # The student prefers 9 (frame 0: the other tied best), 11 (frame 5: worse than 7) and
        # 5 (frame 4: the teacher's choice); it takes index 0-first legal elsewhere.
        stub = FixedPolicy(torch, {9: 3.0, 11: 2.0, 5: 1.0}, width)
        tau = 0.1
        result = self.train.evaluate(stub, shards, tau=tau, batch_rows=4)
        self.assertTrue(stub.training)  # mode restored
        labels = {index: self.labels(index, self.kinds[index], run="run-m") for index in range(6)}
        picks = {0: 9, 4: 5, 5: 11}
        regrets = {index: max(labels[index].values) - labels[index].values[labels[index].legal.index(pick)]
                   for index, pick in picks.items()}
        self.assertEqual(regrets[0], 0.0)
        self.assertEqual(regrets[4], 0.0)
        self.assertGreater(regrets[5], 0.0)
        metrics = result.metrics
        self.assertEqual(metrics["regret_rows"], 3)  # the tie, single-legal and refused rows are excluded
        self.assertEqual(result.regret_rows, [regrets[0], regrets[4], regrets[5]])
        self.assertAlmostEqual(metrics["regret"], sum(regrets.values()) / 3, places=12)
        # Strict: only frame 4 picks the teacher's choice; tie-aware also counts frame 0's tie.
        self.assertEqual(labels[0].choice, 4)
        self.assertAlmostEqual(metrics["strict_agreement"], 1 / 3, places=12)
        self.assertAlmostEqual(metrics["tie_aware_agreement"], 2 / 3, places=12)
        # Policy loss: cross-entropy to softmax(v / tau) on the policy rows 0, 4 and 5.
        cross_entropy = []
        for index in (0, 4, 5):
            legal = labels[index].legal
            scaled = np.asarray(labels[index].values) / tau
            target = np.exp(scaled - scaled.max())
            target /= target.sum()
            logits = stub.logits[legal].double()
            log_probs = (logits - torch.logsumexp(logits, dim=0)).numpy()
            cross_entropy.append(-float((target * log_probs).sum()))
        self.assertAlmostEqual(metrics["policy_loss"], sum(cross_entropy) / 3, places=5)
        # Value losses on the value rows 0, 1, 2 and 5 (outcome by index % 3: win, death, escaped).
        value_rows = [index for index in range(6) if labels[index].value_ok]
        self.assertEqual(value_rows, [0, 1, 2, 5])
        outcome = {"win": 1.0, "death": -1.0, "escaped": 0.0}
        expected = {
            "outcome_mse": np.mean([(0.5 - outcome[labels[i].outcome]) ** 2 for i in value_rows]),
            "hp_kept_mse": np.mean([(0.25 - labels[i].hp_kept) ** 2 for i in value_rows]),
            "realised_v2_mse": np.mean([(-0.5 - labels[i].realised_v2) ** 2 for i in value_rows]),
        }
        for name, value in expected.items():
            self.assertAlmostEqual(metrics[name], float(value), places=5, msg=name)
        self.assertAlmostEqual(metrics["value_loss"], sum(expected.values()) / 3, places=5)
        self.assertAlmostEqual(metrics["loss"], metrics["policy_loss"] + metrics["value_loss"], places=9)
        self.assertEqual((metrics["policy_rows"], metrics["value_rows"], metrics["rows"]), (3, 4, 6))
        self.assertEqual(shards.open_shards, 0)  # validation unmaps its shards

    def test_label_codes_follow_distill_data(self):
        from spire_learn.neural.distill_data import OUTCOMES, ROOT_KINDS

        searched, targets = self.train.label_codes()
        self.assertEqual(searched, ROOT_KINDS.index("searched"))
        self.assertEqual(targets, {OUTCOMES.index("win"): 1.0, OUTCOMES.index("death"): -1.0,
                                   OUTCOMES.index("escaped"): 0.0})
        self.assertNotIn(OUTCOMES.index("none"), targets)
        self.assertEqual(set(self.train.OUTCOME_TARGETS) | {"none"}, set(OUTCOMES))
        # A negative or unknown outcome code on a value row is refused.
        np = self.np
        labels = self.train.ShardSet((self.shard_a,), self.schema).labels[0]
        for code in (-1, OUTCOMES.index("none"), len(OUTCOMES)):
            broken = self.train.ShardLabels(**{**labels.__dict__, "outcome": np.full_like(labels.outcome, code)})
            with self.subTest(code=code), self.assertRaises(self.train.DistillError):
                broken._check()
        # A policy row with a non-finite value is refused (the vectorised check).
        values = labels.values.copy()
        values[labels.legal_offsets[0]] = np.nan
        with self.assertRaises(self.train.DistillError):
            self.train.ShardLabels(**{**labels.__dict__, "values": values})._check()
        labels._check()

    def test_verify_dataset_refuses_tampering(self):
        import shutil

        copy = self.root / "verify" / "shard"
        shutil.copytree(self.shard_a.path, copy)
        ref = self.train.ShardRef("train", 1000, copy, self.shard_a.rows,
                                  self.train._file_sha256(copy / "manifest.json"))
        dataset = self.train.Dataset(chunks=(), chunk_manifest_sha256={}, train=(ref,), validation=())
        self.assertEqual(self.train.verify_dataset(dataset, self.schema), self.shard_a.rows)
        target = copy / "action_mask.npy"
        data = bytearray(target.read_bytes())
        data[-1] ^= 1
        target.write_bytes(bytes(data))
        with self.assertRaisesRegex(self.train.DistillError, "fails verification"):
            self.train.verify_dataset(dataset, self.schema)
        stale = self.train.Dataset(chunks=(), chunk_manifest_sha256={}, validation=(),
                                   train=(self.train.ShardRef("train", 1000, copy, 6, "0" * 64),))
        with self.assertRaisesRegex(self.train.DistillError, "manifest differs"):
            self.train.verify_dataset(stale, self.schema)

    # --- training run ------------------------------------------------------------------------

    def test_final_step_evaluated(self):
        from spire_learn.neural.checkpoint import CheckpointCompatibilityError, load_checkpoint

        hyper = DistillHyperparameters(tau=0.1, learning_rate=1e-3, batch_rows=4, gradient_steps=5,
                                       data_fraction=0.5, microbatch_rows=2)
        config = distill_config(hyper, eval_every=2)
        dataset = self.train.Dataset(chunks=(), chunk_manifest_sha256={}, train=(self.shard_a, self.shard_b),
                                     validation=(self.validation,))
        output = self.root / "run"
        spy = GateSpy()
        summary = self.train.train_student(config, self.schema, dataset, output, log_every=1, gate=spy)
        self.assertEqual(spy.stages, ["train-start", "eval-2", "eval-4", "eval-5"])
        # Evaluated every eval_every steps and at the last step, which is not a multiple.
        self.assertEqual([item["step"] for item in summary["evaluations"]], [2, 4, 5])
        self.assertEqual(summary["final"]["step"], 5)
        self.assertEqual(summary["counters"]["optimizer_steps"], 5)
        self.assertEqual(summary["counters"]["rows_seen"], 20)
        # data_fraction 0.5 of two training shards is the first one.
        self.assertEqual(summary["run_metadata"]["train_shards"], [1000])
        events = [json.loads(line) for line in (output / "train_log.jsonl").read_text().splitlines()]
        self.assertEqual([event["step"] for event in events if event["event"] == "validation"], [2, 4, 5])
        # Wait events reach the log as they happen, including one from before the log existed.
        self.assertEqual([event["stage"] for event in events if event["event"] == "gpu_wait"],
                         ["qualify", "train-start", "eval-2", "eval-4", "eval-5"])
        timing = summary["timing"]
        self.assertIsNone(timing["gpu_step_seconds"])
        self.assertNotIn("gpu_seconds", timing)
        for name in ("train_load_seconds", "train_model_seconds", "eval_load_seconds", "eval_model_seconds"):
            self.assertGreater(timing[name], 0.0, name)
        self.assertLessEqual(timing["train_load_seconds"] + timing["train_model_seconds"],
                             timing["train_wall_seconds"] + 1e-6)
        for evaluation in summary["evaluations"]:
            for key in ("regret", "strict_agreement", "tie_aware_agreement", "policy_loss", "outcome_mse",
                        "hp_kept_mse", "realised_v2_mse", "value_loss"):
                self.assertTrue(math.isfinite(evaluation[key]), key)
            self.assertEqual(evaluation["regret_rows"], 3)  # rows 0, 4 and 5 of the validation shard
        # Only the chosen and the final checkpoints are kept.
        chosen = summary["chosen"]["step"]
        kept = sorted(path.name for path in output.iterdir() if path.is_dir())
        self.assertEqual(kept, ["checkpoint-final"] if chosen == 5 else ["checkpoint-best", "checkpoint-final"])
        self.assertIn(chosen, summary["chosen"]["tied_steps"])
        loaded = load_checkpoint(output / "checkpoint-final", config.identity, purpose="resume", device="cpu")
        state = loaded.metadata["training_state"]
        self.assertEqual((state["optimizer_step"], state["next_environment_seed_ordinal"]), (5, 0))
        self.assertEqual(state["hyperparameter_id"], config.identity.config_hashes["distill"])
        student = self.train.build_student(config, self.schema, device="cpu")
        student.load_state_dict(dict(loaded.model_tensors), strict=True)
        ppo = NeuralIdentity(**{**{name: getattr(config.identity, name) for name in (
            "model_id", "features_id", "action_id", "accepted_raw_mode_mappings")},
            "objective_id": "spire-learn-combat-winloss-v1",
            "config_hashes": {"model": config.identity.config_hashes["model"], "ppo": "b" * 64}})
        with self.assertRaises(CheckpointCompatibilityError):
            load_checkpoint(output / "checkpoint-final", ppo, purpose="warm_start", device="cpu")
        # The same seed and shards give the same run on CPU.
        repeat = self.train.train_student(config, self.schema, dataset, self.root / "repeat", log_every=1)
        self.assertEqual(repeat["evaluations"], summary["evaluations"])
        with self.assertRaises(Exception):
            self.train.train_student(config, self.schema, dataset, output)


if __name__ == "__main__":
    unittest.main()
