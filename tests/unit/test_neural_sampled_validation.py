"""Training sampling and full-validation controls on scripted public frames."""

from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from spire_learn.artifacts.codec import canonical_fingerprint
from spire_learn.contracts import ContractError
from spire_learn.experiments.budget import Budget
from spire_learn.neural import envs, features
from spire_learn.neural.artifacts import NeuralWriter
from spire_learn.neural.collector import Collector
from spire_learn.neural.config import _stream_runtime_projection, load_neural_config
from spire_learn.neural.contracts import PolicyChoice
from spire_learn.neural.runner import _checkpoint_recovery, _config_projection, _write_collection_events
from tests.unit.test_neural_collector import FakePolicy
from tests.unit.test_neural_envs import _raw, _ScriptedProvider
from tests.unit.test_neural_runner import _resume_bundle
from tests.unit.test_neural_schema import _api, _synthetic_schema


class SampledValidationTests(unittest.TestCase):
    def setUp(self):
        self.api = _api(self)
        self.schema = self.enterContext(_synthetic_schema(self.api))

    def driver(self, *, corrupt=None, slots=2, steps=68):
        initial = [_raw(self.api, self.schema) for _ in range(slots)]
        script = []
        for observation in range(1, steps + 1):
            boundaries = [_raw(self.api, self.schema) for _ in initial]
            if observation == corrupt:
                boundaries[0].observation["player_known"]._data = b"\x02"
            script.append(tuple(envs._RawTransition(row, 0, False, False) for row in boundaries))
        driver = envs.CombatDriver(_ScriptedProvider(initial, script), self.schema)
        self.addCleanup(driver.close)
        return driver

    def test_stride_16_validates_first_of_rollout_and_every_16th(self):
        driver = self.driver()
        with patch.object(envs, "validate_frame", wraps=envs.validate_frame) as validate:
            for rollout in range(2):
                before = validate.call_count
                driver.begin_rollout(16, (rollout == 0,) * 2)
                if rollout == 0:
                    driver.reset(seeds=(0, 1), entry_ids=("entry", "entry"))
                self.assertEqual(validate.call_count - before, 2)
                for index in range(1, 35):
                    before = validate.call_count
                    driver.step((2, 2))
                    self.assertEqual(validate.call_count - before, 2 if index % 16 == 0 else 0)

    def test_sampled_validation_raises_on_corrupted_sampled_observation(self):
        driver = self.driver(corrupt=16, slots=1, steps=16)
        driver.begin_rollout(16, (True,))
        driver.reset(seeds=(0,), entry_ids=("entry",))
        for _ in range(15):
            driver.step((2,))
        with self.assertRaises(envs.DriverFault) as raised:
            driver.step((2,))
        self.assertIsInstance(raised.exception.__cause__, ContractError)

    def test_unsampled_corruption_is_not_checked_but_eval_still_validates_all(self):
        for config_name, expected_stride in (("standalone", 16), ("standalone-selection", 1)):
            with self.subTest(config=config_name):
                config = load_neural_config(Path(f"configs/neural/{config_name}.toml"))
                config = replace(config, budgets=replace(config.budgets, validation_stride=16))
                driver = self.driver(corrupt=5, slots=1, steps=5)
                config = replace(config, rollout_rows=1, rollout_steps=5)

                class ScriptedPolicy(FakePolicy):
                    def act(self, frames, *, generator, deterministic=False):
                        return tuple(PolicyChoice(2, -0.25, 0.5, 0) for _ in frames)

                collector = Collector(driver, ScriptedPolicy(config), config,
                                      controller_rng=random.Random(1), curriculum_rng=random.Random(2),
                                      budget=Budget(60, lambda: 0.0), policy_generator=object())
                self.addCleanup(collector.close)
                with patch.object(envs, "validate_frame", wraps=envs.validate_frame) as validate:
                    args = dict(max_learned_decisions=5, max_env_actions=5, policy_version=0)
                    if expected_stride == 16:
                        rollout = collector.collect(**args)
                        self.assertEqual(len(rollout.records), 5)
                        self.assertEqual(validate.call_count, 1)
                    else:
                        with self.assertRaises(envs.DriverFault) as raised:
                            collector.collect(**args)
                        self.assertIsInstance(raised.exception.__cause__, ContractError)
                        self.assertEqual(validate.call_count, 6)

    def test_validation_stride_absent_keeps_hashes(self):
        config = load_neural_config(Path("configs/neural/standalone.toml"))
        self.assertEqual(config.effective_validation_stride, 1)
        self.assertNotIn("validation_stride", _config_projection(config)["budgets"])
        self.assertEqual(config.config_sha256, "027ce4460a351d32a89c90763fb7012814c55b257127650da9e181b4b3166ca0")
        self.assertEqual(canonical_fingerprint(_stream_runtime_projection(config)),
                         "86860ab19b1c6436416244be7a08b0356770702ef973ee25e898d7fa33d910dd")

    def test_accepted_action_hashes_follow_observation_sampling(self):
        driver = self.driver(slots=1, steps=2)
        driver.begin_rollout(16, (True,))
        driver.reset(seeds=(0,), entry_ids=("entry",))
        records = []
        for decision in range(2):
            result, = driver.step((2,))
            records.append(SimpleNamespace(
                stream_id="slot-0", task_id="fight-0", decision_id=str(decision),
                choice=SimpleNamespace(original_action_index=2), actor_training_mask=True,
                native_reward=0, task_signal=SimpleNamespace(
                    reward=0, task_terminal=False, bootstrap=True, continuation=True),
                terminated=False, truncated=False, source=result.source.frame,
                successor=result.successor.frame,
            ))
        writer = Mock()
        _write_collection_events(writer, SimpleNamespace(records=records, collection=None),
                                 cursors={}, operation="train")
        first, second = [call.args[0] for call in writer.event.call_args_list]
        self.assertRegex(first["source_public_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNone(first["successor_public_sha256"])
        self.assertIsNone(second["source_public_sha256"])
        self.assertIsNone(second["successor_public_sha256"])

    def test_parser_and_projection_admit_only_positive_integer_strides(self):
        original = Path("configs/neural/standalone.toml")
        text = original.read_text()
        for key in ("manifest_path", "output_root"):
            text = text.replace(f'{key} = "', f'{key} = "{original.parent.resolve()}/')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stride.toml"
            for raw in ("1", "16", "0", "-1", "true", "1.5"):
                with self.subTest(stride=raw):
                    path.write_text(text.replace("[budgets]", f"[budgets]\nvalidation_stride = {raw}"))
                    if raw not in ("1", "16"):
                        with self.assertRaisesRegex(ValueError, "validation_stride"):
                            load_neural_config(path)
                    else:
                        config = load_neural_config(path)
                        self.assertEqual(config.effective_validation_stride, int(raw))
                        self.assertEqual(_config_projection(config)["budgets"]["validation_stride"], int(raw))
                        self.assertEqual(_stream_runtime_projection(config)["budgets"]["validation_stride"],
                                         int(raw))

    def test_resume_accepts_null_hashes_only_in_training_records(self):
        event = {"event": "accepted_action", "source_public_sha256": "a" * 64,
                 "successor_public_sha256": None}
        with tempfile.TemporaryDirectory() as directory:
            for operation in ("train", "evaluate", "calibrate"):
                with self.subTest(operation=operation):
                    writer = NeuralWriter.create(Path(directory), operation)
                    try:
                        writer.start({"operation": operation}, {})
                        if operation == "train":
                            writer.event(event)
                            saved = json.loads((writer.run_dir / "events.jsonl").read_text())
                            self.assertEqual(saved["record"], event)
                            config = load_neural_config(Path("configs/neural/standalone.toml"))
                            config = replace(config, budgets=replace(config.budgets, validation_stride=16))
                            _, checkpoint, *_ = _resume_bundle(config, {})
                            # The checkpoint recovery contract is independent of the sibling event log.
                            recovery, _ = _checkpoint_recovery(checkpoint, config)
                            self.assertEqual(recovery.next_decision_sequence, 0)
                        else:
                            with self.assertRaisesRegex(ValueError, "successor_public_sha256"):
                                writer.event(event)
                            writer.event(event | {"successor_public_sha256": "b" * 64})
                    finally:
                        writer.close()

    def test_encode_validates_authored_frames_and_reuses_driver_evidence(self):
        driver = self.driver(slots=1, steps=1)
        driver.begin_rollout(16, (True,))
        driver.reset(seeds=(0,), entry_ids=("entry",))
        transition, = driver.step((2,))

        class EncodingReached(Exception):
            pass

        with patch.dict(sys.modules, torch=Mock()), \
                patch.object(features, "_pack_tensor", side_effect=EncodingReached), \
                patch.object(features, "validate_frame", wraps=features.validate_frame) as validate:
            for frame in (transition.source.frame, transition.successor.frame,
                          replace(transition.source.frame)):
                with self.assertRaises(EncodingReached):
                    features.encode_frames((frame,), self.schema, device="cpu")
            validate.assert_called_once()
            self.assertIsNot(validate.call_args.args[0], transition.source.frame)

    def test_masked_reset_advances_only_selected_slot(self):
        driver = self.driver(steps=16)
        driver.begin_rollout(16, (True, True))
        driver.reset(seeds=(0, 1), entry_ids=("entry", "entry"))
        for _ in range(14):
            driver.step((2, 2))
        driver.reset(seeds=(2, 1), entry_ids=("entry", None), reset_mask=(True, False))
        step_15 = driver.step((2, 2))
        step_16 = driver.step((2, 2))
        self.assertEqual([row.successor.frame._driver_validation[1] for row in step_15], [True, False])
        self.assertEqual([row.successor.frame._driver_validation[1] for row in step_16], [False, True])
