"""The model-family registry and the student evaluation seam (stdlib only; no torch)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import model_families as families
from spire_learn.neural.config import DISTILL_OBJECTIVE_ID, STANDARD_LEARNER_IDENTITY_IDS
from spire_learn.neural.contracts import ModelConfig

FF_IDS = ("spire-learn-combat-ff-v1", "spire-learn-combat-ff-standard-v1")
TRANSFORMER_ID = "spire-learn-combat-transformer-v1"
MAPPING = ("spire-learn-standard-combat-compact-v1@49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8"
           "+gym-1.2.4")


@dataclass(frozen=True)
class FakeTransformerConfig:
    layers: int
    width: int


class _Policy:
    """Exposes the Collector interface; ``drop`` removes one attribute."""

    def __init__(self, model_id, objective_id="spire-learn-combat-winloss-v1", *, drop=None, parameters=0):
        self.model_id, self.features_id = model_id, "features"
        self.action_id, self.objective_id = "spire-learn-combat-dispatch-v1", objective_id
        self._behavior_policy_version = 0
        self._parameters = parameters
        if drop is not None:
            setattr(self, drop, None) if drop in families.POLICY_METHODS else delattr(self, drop)

    def act(self, frames, *, generator, deterministic=False):
        return ()

    def values(self, frames):
        return ()

    def forward(self, batch):
        return None

    def parameters(self):
        return [SimpleNamespace(numel=lambda: self._parameters)]


def _transformer_family(**changes):
    family = families.ModelFamily(
        family_id="transformer", model_ids=(TRANSFORMER_ID,), config_type=FakeTransformerConfig,
        parameter_cap=16_000_000, build=lambda schema, config, **options: _Policy(TRANSFORMER_ID, **{
            key: value for key, value in options.items() if key == "objective_id"}),
        encode=lambda frames, schema, *, device, trim_padding=False: frames)
    return replace(family, **changes)


class ModelFamilyTests(unittest.TestCase):
    def test_ff_family_unchanged(self):
        # Both FF learners map to the one FF family: ModelConfig, the 10M cap, build_policy. The only
        # other registered families are the entity transformers S2 and S3 (tests/unit/test_transformer_config.py).
        self.assertEqual(families.registered_model_ids(),
                         tuple(sorted((*FF_IDS, "spire-learn-combat-entity-transformer-v1",
                                      "spire-learn-combat-entity-transformer-cardmeta-v1"))))
        for model_id in FF_IDS:
            family = families.model_family(model_id)
            self.assertIs(family, families.FF_FAMILY)
            self.assertIs(family.config_type, ModelConfig)
            self.assertEqual(family.parameter_cap, 10_000_000)
        from spire_learn.neural import model

        self.assertEqual(families.FF_PARAMETER_CAP, model._HARD_PARAMETER_CAP)
        config = ModelConfig(embedding_width=16, hidden_width=64, max_parameters=10_000_000)
        built = _Policy(FF_IDS[1], DISTILL_OBJECTIVE_ID)
        options = dict(seed=3, device="cpu", objective_id=DISTILL_OBJECTIVE_ID, model_id=FF_IDS[1],
                       features_id=STANDARD_LEARNER_IDENTITY_IDS[1], trim_padding=True)
        with patch.object(model, "build_policy", return_value=built) as build:
            self.assertIs(families.build_family_policy(FF_IDS[1], "schema", config, **options), built)
        # The FF build is build_policy with the caller's arguments, unchanged.
        build.assert_called_once_with("schema", config, **options)
        # The historical v1 call (no identity keywords) also passes through unchanged.
        v1 = _Policy(FF_IDS[0])
        with patch.object(model, "build_policy", return_value=v1) as build:
            families.build_family_policy(FF_IDS[0], "schema", config, seed=1, device="cpu")
        build.assert_called_once_with("schema", config, seed=1, device="cpu")
        # The FF encoder is encode_frames.
        from spire_learn.neural import features

        with patch.object(features, "encode_frames", return_value="batch") as encode:
            self.assertEqual(families.FF_FAMILY.encode(["frame"], "schema", device="cpu", trim_padding=True), "batch")
        encode.assert_called_once_with(["frame"], "schema", device="cpu", trim_padding=True)

    def test_unknown_family_refused(self):
        with self.assertRaisesRegex(ContractError, "no registered model family.*FF"):
            families.model_family(TRANSFORMER_ID)
        with self.assertRaisesRegex(ContractError, "no registered model family"):
            families.build_family_policy(TRANSFORMER_ID, "schema", FakeTransformerConfig(2, 8))
        with self.assertRaises(ContractError):
            families.model_family(None)
        # CheckpointPlayer dispatches through the registry and refuses before importing torch.
        from spire_learn.neural.policy_player import CheckpointPlayer, standard_ff_identity

        config = ModelConfig(embedding_width=4, hidden_width=8, max_parameters=10_000_000)
        identity = replace(standard_ff_identity(config, MAPPING), model_id=TRANSFORMER_ID)
        with self.assertRaisesRegex(ContractError, "no registered model family"):
            CheckpointPlayer("unused", "schema", seed=0, expected_identity=identity, model_config=config)
        # The FF family takes a ModelConfig only.
        with self.assertRaisesRegex(TypeError, "ModelConfig"):
            families.build_family_policy(FF_IDS[1], "schema", FakeTransformerConfig(2, 8))
        with self.assertRaisesRegex(TypeError, "ModelConfig"):
            CheckpointPlayer("unused", "schema", seed=0, expected_identity=standard_ff_identity(config, MAPPING),
                             model_config=FakeTransformerConfig(2, 8))

    def test_registered_family_dispatches_and_is_checked(self):
        family = _transformer_family()
        with patch.dict(families._FAMILIES):
            families.register_model_family(family)
            self.assertIs(families.model_family(TRANSFORMER_ID), family)
            policy = families.build_family_policy(TRANSFORMER_ID, "schema", FakeTransformerConfig(2, 8),
                                                  seed=0, device="cpu", objective_id=DISTILL_OBJECTIVE_ID)
            self.assertEqual(policy.model_id, TRANSFORMER_ID)
            with self.assertRaisesRegex(ContractError, "already"):
                families.register_model_family(family)
            with self.assertRaisesRegex(ContractError, "already"):
                families.register_model_family(_transformer_family(family_id="ff-copy", model_ids=(FF_IDS[1],)))
            with self.assertRaisesRegex(TypeError, "FakeTransformerConfig"):
                families.build_family_policy(TRANSFORMER_ID, "schema", ModelConfig(4, 8, 10_000_000))
        self.assertNotIn(TRANSFORMER_ID, families.registered_model_ids())
        with self.assertRaises(TypeError):
            families.register_model_family(SimpleNamespace(model_ids=(TRANSFORMER_ID,)))
        for bad in (dict(family_id=""), dict(model_ids=()), dict(model_ids=("a", "a")), dict(parameter_cap=0),
                    dict(build=None), dict(config_type="x")):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                _transformer_family(**bad)

    def test_built_policy_must_expose_the_collector_interface(self):
        for name in ("act", "values", "forward", "_behavior_policy_version", "model_id", "features_id",
                     "action_id", "objective_id"):
            with self.subTest(missing=name), self.assertRaisesRegex(ContractError, name):
                families.check_policy(_Policy(TRANSFORMER_ID, drop=name), model_id=TRANSFORMER_ID)
        with self.assertRaisesRegex(ContractError, "model_id"):
            families.check_policy(_Policy(FF_IDS[0]), model_id=TRANSFORMER_ID)
        with self.assertRaisesRegex(ContractError, "objective_id"):
            families.check_policy(_Policy(TRANSFORMER_ID), model_id=TRANSFORMER_ID, objective_id=DISTILL_OBJECTIVE_ID)
        over = _transformer_family(build=lambda schema, config, **options: _Policy(TRANSFORMER_ID, parameters=16_000_001))
        with patch.dict(families._FAMILIES, {TRANSFORMER_ID: over}), self.assertRaisesRegex(ContractError, "cap"):
            families.build_family_policy(TRANSFORMER_ID, "schema", FakeTransformerConfig(2, 8))

    def test_median_forward_timing(self):
        ticks = iter(range(10_000))
        calls, syncs = [], []
        result = families.median_forward_seconds(lambda: calls.append(1), passes=5, warmup=3,
                                                 synchronize=lambda: syncs.append(1),
                                                 timer=lambda: float(next(ticks)) ** 2)
        # Samples (2k+1)^2 - (2k)^2 = 4k + 1 for k = 0..4: 1, 5, 9, 13, 17.
        self.assertEqual(result, {"median_seconds": 9.0, "passes": 5, "warmup": 3, "min_seconds": 1.0,
                                  "max_seconds": 17.0})
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(syncs), 6)  # once after warm-up, then after every timed pass
        for bad in (dict(passes=0), dict(warmup=-1), dict(passes=1.5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                families.median_forward_seconds(lambda: None, **bad)

    def test_import_is_torch_free(self):
        source = inspect.getsource(families)
        top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        self.assertFalse([line for line in top_level if "torch" in line or "numpy" in line])


class RunnerDispatchTests(unittest.TestCase):
    def test_evaluation_builds_through_the_registry_and_training_through_build_policy(self):
        from spire_learn.neural import model, runner
        from spire_learn.neural.policy_player import standard_ff_identity

        config = ModelConfig(embedding_width=4, hidden_width=8, max_parameters=10_000_000)
        identity = standard_ff_identity(config, MAPPING)
        options = dict(seed=9, device="cpu", model_id=FF_IDS[1], features_id=STANDARD_LEARNER_IDENTITY_IDS[1])
        for purpose in ("evaluate", "train", "resume", "warm_start"):
            with self.subTest(purpose=purpose), \
                    patch.object(runner, "build_family_policy", wraps=runner.build_family_policy) as family, \
                    patch.object(model, "build_policy", return_value="policy") as build:
                self.assertEqual(runner._build_runtime_policy(purpose, identity, "schema", config, **options),
                                 "policy")
            build.assert_called_once_with("schema", config, **options)
            self.assertEqual(family.call_count, int(purpose == "evaluate"))
        # An unregistered family is refused at evaluation, before anything is built.
        with patch.object(model, "build_policy") as build, self.assertRaisesRegex(ContractError, "no registered"):
            runner._build_runtime_policy("evaluate", replace(identity, model_id=TRANSFORMER_ID), "schema",
                                         FakeTransformerConfig(2, 8))
        build.assert_not_called()
        family = _transformer_family()
        with patch.dict(families._FAMILIES, {TRANSFORMER_ID: family}):
            policy = runner._build_runtime_policy("evaluate", replace(identity, model_id=TRANSFORMER_ID), "schema",
                                                  FakeTransformerConfig(2, 8), seed=0, device="cpu")
        self.assertEqual(policy.model_id, TRANSFORMER_ID)

    def test_create_runtime_admits_the_family_before_device_work(self):
        from spire_learn.neural import runner
        from spire_learn.neural.policy_player import standard_ff_identity

        config = ModelConfig(embedding_width=4, hidden_width=8, max_parameters=10_000_000)
        identity = replace(standard_ff_identity(config, MAPPING), model_id=TRANSFORMER_ID)
        fake = SimpleNamespace(experimental_trim_compute=None, operation="evaluate", split="selection",
                               rollout_rows=1, rollout_steps=1, ppo=None, scenario_stream=None, model=config)
        with patch.object(runner, "NeuralConfig", SimpleNamespace), \
                patch.object(runner, "_selected_identity", return_value=identity), \
                patch.object(runner, "plan_runtime_device", side_effect=AssertionError("device work")), \
                self.assertRaisesRegex(ContractError, "no registered model family"):
            runner.create_runtime(fake, purpose="evaluate", checkpoint="unused", policy_kind="checkpoint",
                                  action_selection="deterministic")


if __name__ == "__main__":
    unittest.main()
