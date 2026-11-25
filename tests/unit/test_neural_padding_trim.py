"""CPU parity checks for trimmed Standard combat batches."""

from dataclasses import asdict, replace
import struct
import random
import unittest

from spire_learn.experiments.budget import Budget
from spire_learn.neural.checkpoint import validate_checkpoint_metadata
from spire_learn.neural.collector import Collector
from spire_learn.neural.config import (
    STANDARD_LEARNER_IDENTITY_IDS, _canonical_hash, _stream_runtime_projection, load_neural_config,
)
from spire_learn.neural.features import encode_frames
from spire_learn.neural.model import build_policy
from spire_learn.neural.ppo import ppo_update
from spire_learn.neural.schema import candidates, validate_frame
from tests.unit import test_neural_collector as scripted
from tests.unit import test_neural_standard_compact as compact
from tests.unit.test_neural_checkpoint import _metadata
from tests.unit.test_neural_standard_compact import COMPACT_TEMPLATE, compact_text, load_text


def standard_policy(schema, config, *, trim_padding, experimental_trim_compute=False):
    model_id, features_id = STANDARD_LEARNER_IDENTITY_IDS[:2]
    return build_policy(schema, config.model, seed=1103, device="cpu", model_id=model_id,
                        features_id=features_id, trim_padding=trim_padding,
                        experimental_trim_compute=experimental_trim_compute)


class PaddingConfigTests(unittest.TestCase):
    def test_trim_padding_absent_keeps_config_hash(self):
        config = load_neural_config(COMPACT_TEMPLATE)
        self.assertEqual(config.config_sha256,
                         "fbee35dd59d3efae8162ac47dafc30866155ce75f7789d073278824070c06a04")
        self.assertEqual(_canonical_hash(_stream_runtime_projection(config)),
                         "6995403dcd19f340f8b214e323f7342725bea64209654533c0915d3c1129d8ee")
        for flag in ("true", "false"):
            explicit = load_text(compact_text().replace("[model]", f"[model]\ntrim_padding = {flag}"))
            self.assertIs(explicit.trim_padding, flag == "true")
            self.assertEqual(explicit.accepted_identities, config.accepted_identities)
            self.assertIs(_stream_runtime_projection(explicit)["model"]["trim_padding"], flag == "true")
            validate_checkpoint_metadata(_metadata(identity=config.accepted_identities[0]),
                                         explicit.accepted_identities[0], purpose="evaluate")
        with self.assertRaisesRegex(ValueError, "trim_padding"):
            load_text(compact_text().replace("[model]", "[model]\ntrim_padding = 1"))


class PaddingTensorTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            self.skipTest("CPU Torch is required")

        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.config = load_neural_config(COMPACT_TEMPLATE)
        self.driver, _ = compact.CompactDriverTests()._driver(self.config, compact.FakeCompactVector)
        self.addCleanup(self.driver.close)
        self.schema = self.driver.schema
        self.base = self.driver.reset(seeds=[1] * self.driver.batch_size, entry_ids=[
            self.config.cases.for_mode(self.config.mode)[0].entry_id
        ] * self.driver.batch_size)[0].frame

    def frame(self, count, *, offset=0):
        spec = self.schema.tensors["combat_cards"]
        spec_mask = self.base.tensors[spec.occupied_key]
        data = bytearray(spec_mask.data)
        data[offset:offset + count] = bytes([1]) * count
        buffers = dict(self.base.tensors)
        buffers[spec.occupied_key] = replace(spec_mask, data=bytes(data))
        raw = buffers["combat_cards"]
        code = compact._CODES[raw.dtype]
        fields = [column.unavailable if column.reference else 0 for column in spec.columns]
        raw_known = buffers[spec.known_key]
        card_kind = next(i for i, column in enumerate(spec.columns) if column.field == "card_kind")
        damage = next(i for i, column in enumerate(spec.columns) if column.field == "base_damage")
        row_format = struct.Struct("=" + code * len(fields))
        values, known = bytearray(raw.data), bytearray(raw_known.data)
        for index in range(offset, offset + count):
            fields[card_kind] = count + index
            fields[damage] = count * (index + 2)
            row_format.pack_into(values, index * row_format.size, *fields)
            known[index * len(fields) + card_kind] = 1
            known[index * len(fields) + damage] = 1
        buffers["combat_cards"] = replace(raw, data=bytes(values))
        buffers[spec.known_key] = replace(raw_known, data=bytes(known))
        if count:
            self.add_play_card(buffers, source_row=offset)
        frame = replace(self.base, tensors=buffers)
        validate_frame(frame, self.schema)
        return frame

    def add_play_card(self, buffers, *, source_row):
        spec = self.schema.tensors["actions"]
        raw = buffers["actions"]
        code = compact._CODES[raw.dtype]
        record = struct.Struct("=" + code * len(spec.columns))
        data, known = [], bytearray(len(spec.columns))
        for index, column in enumerate(spec.columns):
            value = column.unavailable if column.reference else 0
            if column.field == "verb":
                value = self.schema.vocabularies[column.vocabulary].index("play_card")
                known[index] = 1
            elif column.field == "source_group":
                value = self.schema.vocabularies[column.vocabulary].index("combat_cards")
                known[index] = 1
            elif column.field == "source_row":
                value = source_row
                known[index] = 1
            data.append(value)
        values = bytearray(raw.data)
        record.pack_into(values, record.size, *data)
        buffers["actions"] = replace(raw, data=bytes(values))
        raw_known = buffers[spec.known_key]
        width = len(known)
        buffers[spec.known_key] = replace(
            raw_known, data=raw_known.data[:width] + bytes(known) + raw_known.data[2 * width:],
        )
        for key in ("action_mask", spec.occupied_key):
            mask = buffers[key]
            buffers[key] = replace(mask, data=b"\x01\x01" + mask.data[2:])

    def test_trimmed_batch_cuts_rows_to_batch_max_occupied(self):
        import torch

        frames = tuple(self.frame(count) for count in (3, 7, 0))
        full = encode_frames(frames, self.schema, device="cpu")
        trim = encode_frames(frames, self.schema, device="cpu", trim_padding=True)
        self.assertEqual(trim.values["combat_cards"].shape[1], 7)
        self.assertEqual(trim.values["powers"].shape[1], 1)
        self.assertEqual(trim.values["actions"].shape[1], 2)
        for group, coverage in full.known_coverage.items():
            self.assertTrue(torch.equal(trim.known_coverage[group], coverage), group)
        gapped = encode_frames((self.frame(3, offset=8),), self.schema, device="cpu", trim_padding=True)
        self.assertEqual(gapped.values["combat_cards"].shape[1], 11)

    def test_trimmed_and_untrimmed_give_identical_policy_outputs(self):
        import torch

        frames = tuple(self.frame(i % 8) for i in range(64))
        full = standard_policy(self.schema, self.config, trim_padding=False)
        trim = standard_policy(self.schema, self.config, trim_padding=True)
        with torch.no_grad():
            expected = full(encode_frames(frames, self.schema, device="cpu"))
            actual = trim(encode_frames(frames, self.schema, device="cpu", trim_padding=True))
        self.assertTrue(torch.equal(expected.logits.view(torch.int32), actual.logits.view(torch.int32)))
        self.assertTrue(torch.equal(expected.values.view(torch.int32), actual.values.view(torch.int32)))
        for deterministic in (False, True):
            def choices(policy):
                return policy.act(frames, generator=torch.Generator().manual_seed(7),
                                  deterministic=deterministic)
            self.assertEqual(choices(full), choices(trim))

    def test_experimental_trim_cpu_one_update(self):
        import torch

        frames = tuple(self.frame(i % 8) for i in range(64))
        batch = encode_frames(frames, self.schema, device="cpu", trim_padding=True)
        exact, experimental = (
            standard_policy(self.schema, self.config, trim_padding=True, experimental_trim_compute=flag)
            for flag in (False, True)
        )

        def update(policy):
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.ppo.learning_rate,
                                         eps=self.config.ppo.adam_eps)
            output = policy(batch)
            legal_logits = output.logits[batch.action_mask]
            loss = (legal_logits - 0.25).square().mean() + (output.values + 0.5).square().mean()
            loss.backward()
            optimizer.step()
            return output

        before = tuple(update(policy) for policy in (exact, experimental))
        different_grads = sum(
            left.grad is None or right.grad is None or not torch.equal(left.grad, right.grad)
            for left, right in zip(exact.parameters(), experimental.parameters(), strict=True)
            if left.grad is not None or right.grad is not None
        )
        with torch.no_grad():
            after = exact(batch), experimental(batch)
        max_logit, max_value = 0.0, 0.0
        for left, right in (before, after):
            logit_diff = (left.logits[batch.action_mask] - right.logits[batch.action_mask]).abs().max().item()
            value_diff = (left.values - right.values).abs().max().item()
            max_logit = max(max_logit, logit_diff)
            max_value = max(max_value, value_diff)
        metrics = (f"64 frames, one Adam update: max legal logit diff={max_logit:.9g}, "
                   f"max value diff={max_value:.9g}, gradient tensors differing={different_grads}")
        print(metrics, flush=True)
        self.assertLessEqual(max_logit, 1e-5, metrics)
        self.assertLessEqual(max_value, 1e-5, metrics)
        for left, right in (before, after):
            self.assertTrue(torch.equal(
                left.logits.masked_fill(~batch.action_mask, -torch.inf).argmax(dim=1),
                right.logits.masked_fill(~batch.action_mask, -torch.inf).argmax(dim=1),
            ), metrics)

    def test_trimmed_training_segment_is_byte_identical(self):
        import torch

        def run(trim_padding):
            model_id, features_id = STANDARD_LEARNER_IDENTITY_IDS[:2]
            ppo = replace(self.config.ppo, epochs=1, minibatch_rows=8)
            original = self.config.accepted_identities[0]
            identity = replace(original, model_id=model_id, features_id=features_id,
                               config_hashes={**original.config_hashes, "ppo": _canonical_hash(asdict(ppo))})
            config = replace(self.config, accepted_identities=(identity,), rollout_steps=4,
                             budgets=replace(self.config.budgets, max_turn_actions=None),
                             ppo=ppo, trim_padding=trim_padding)
            def observed_row(frame):
                return replace(scripted.row(0, active=True, entered=1, visit=0),
                               frame=frame, candidates=candidates(frame, self.schema))

            rows = tuple(observed_row(self.frame(count)) for count in (0, 3, 7, 20))

            def advance(before, actions):
                return tuple(scripted.transition(source, rows[(slot + action + 1) % len(rows)])
                             for slot, (source, action) in enumerate(zip(before, actions, strict=True)))

            driver = scripted.ScriptedDriver(
                "standalone", config.rollout_rows, steps=[advance] * 8,
                reset_rows=lambda slot, *_: rows[slot],
            )
            driver.schema = self.schema
            self.addCleanup(driver.close)
            policy = standard_policy(driver.schema, config, trim_padding=config.trim_padding)
            optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.learning_rate,
                                         eps=config.ppo.adam_eps)
            budget = Budget(max_wall_seconds=60, clock=lambda: 0.0)
            collector = Collector(driver, policy, config, controller_rng=random.Random(9),
                                  curriculum_rng=random.Random(10), budget=budget,
                                  policy_generator=torch.Generator().manual_seed(11))
            shuffle = torch.Generator().manual_seed(12)
            initial_state = {name: value.clone() for name, value in policy.state_dict().items()}
            updates = []
            for version in range(2):
                rollout = collector.collect(max_learned_decisions=32, max_env_actions=32,
                                            policy_version=version)
                metrics = ppo_update(policy, optimizer, rollout, config.ppo,
                                     shuffle_generator=shuffle, budget=budget)
                self.assertGreater(metrics.accepted_steps, 0)
                updates.append((metrics, {name: tensor.detach().clone()
                                          for name, tensor in policy.state_dict().items()}))
            self.assertTrue(any(not torch.equal(initial_state[name], value)
                                for name, value in updates[0][1].items()))
            return updates

        expected, actual = run(False), run(True)
        for (expected_metrics, expected_state), (actual_metrics, actual_state) in zip(expected, actual):
            self.assertEqual(expected_metrics, actual_metrics)
            for name in expected_state:
                self.assertTrue(torch.equal(expected_state[name].view(torch.int32), actual_state[name].view(torch.int32)), name)
