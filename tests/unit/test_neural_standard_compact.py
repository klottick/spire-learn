"""The opt-in compact Standard combat view (Gym v1.2.0, ``spire-gym-standard-combat-view-compact-v1``).

Host tests use the captured compact descriptors and fake vector environments. The gated tests
(``SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1``, runtime container on Gym v1.2.0, with torch) step the
same seeded states in the dense and the compact layout and require byte-identical features and
policy outputs on CPU, and check that a real capacity overflow is a driver fault.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import envs
from spire_learn.neural.checkpoint import CheckpointCompatibilityError, validate_checkpoint_metadata
from spire_learn.neural.config import _stream_runtime_projection, load_neural_config
from spire_learn.neural.gym_contracts import CONTRACTS, GYM_V1_1_1_CONTRACT, GYM_V1_2_0_CONTRACT
from spire_learn.neural.schema import parse_combat_descriptor, validate_frame
from tests.unit import test_neural_envs_standard as standard_env_tests
from tests.unit.test_neural_checkpoint import _metadata
from tests.unit.test_neural_envs import _DType
from tests.unit.test_neural_gym_1_1_1 import _refuse_torch

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/development/neural-combat"
WORK = Path(os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT") or ARTIFACTS)
FIXTURES = ROOT / "tests/fixtures/neural"
GYM_1_1_1 = "e3c13836bd1458be8a87b93160f5c350f78fcd80"
GYM_1_2_0 = "4d198064b3b568b6f0d08d6e3937d236fbc48e9b"
GYM_1_2_4 = "673eb8f220023b0c95401bbf96bec814a359457b"
GYM_1_2_5 = "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1"  # a drop-in in the gym-1.2.4 rules epoch
EPOCH_1_2_4 = (GYM_1_2_4, GYM_1_2_5)
DENSE = "5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e"
COMPACT = "49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8"
COMPACT_PLAY = "fbfade07003bd64ad6071243ec3a52e23413c88c5758ca702a515945935439ae"
COMPACT_MAPPING = f"spire-learn-standard-combat-compact-v1@{COMPACT}+gym-1.2.0"
COMPACT_PLAY_MAPPING = f"spire-learn-standard-combat-compact-v1@{COMPACT_PLAY}+gym-1.2.0"
DENSE_120_MAPPING = f"spire-learn-standard-combat-v1@{DENSE}+gym-1.2.0"
DENSE_120_RUN_MAPPING = (
    "spire-learn-standard-run-v1@bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae"
    f":{DENSE}+gym-1.2.0")
COMPACT_TEMPLATE = ROOT / "configs/neural/gym-1.2.0/standard-standalone.toml"
CASES = ROOT / "configs/cases/neural/standard-standalone-train.json"
_RUN_NATIVE = os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1"

# Byte-identity pins of the dense v1.1.1 series (config hashes before the compact view existed).
V1_1_1_CONFIG_PINS = {
    "configs/neural/gym-1.1.1/standard-standalone.toml":
        "6a2a364df760e92ceffbf21746b8bcf316a9e5cf3cf60952834b0c3202e4d34d",
    "configs/neural/gym-1.1.1/standard-run.toml":
        "e7c6e092e2f65b597e789089778a7831e32cf353b484bb29dde92c42c98edaf8",
    "configs/neural/gym-1.1.1/standard-stream.toml":
        "3a7e76ad13d145fea127bd023f6c8962cf6db3cbdc983b051a9264e597477262",
    "configs/neural/baseline-standard-1.1.1-r2/template.toml":
        "d48fbc4b4cb889dcd8ae9ebbac8993cd74748dda5c230b2fbc41a96973adebcc",
    "configs/neural/profile-standard-1.1.1/template-16.toml":
        "eac2a4390b5181d838a930aab75efb63e486317ef093fd9e27778e9355313be0",
}
V1_1_1_MAPPINGS = {
    "standard-standalone": f"spire-learn-standard-combat-v1@{DENSE}+gym-1.1.1",
    "standard-run": ("spire-learn-standard-run-v1@bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae"
                     f":{DENSE}+gym-1.1.1"),
}


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _retarget(text: str, revision: str) -> str:
    """The v1.2.0 template text for a later release (same compact view; revision, epoch and output)."""
    if revision not in EPOCH_1_2_4:
        return text
    return text.replace(GYM_1_2_0, revision).replace("+gym-1.2.0", "+gym-1.2.4")


def compact_text(*, preset="standard-combat-compact-v1", fingerprint=COMPACT, mapping=COMPACT_MAPPING,
                 revision=GYM_1_2_0, rows=None) -> str:
    """The shipped compact v1.2.0 template, self-contained (absolute manifest and output paths)."""
    text = COMPACT_TEMPLATE.read_text()
    text = text.replace('standard_view_preset = "standard-combat-compact-v1"', f'standard_view_preset = "{preset}"')
    text = text.replace(COMPACT_MAPPING, mapping).replace(f'"{COMPACT}"', f'"{fingerprint}"')
    text = text.replace(GYM_1_2_0, revision)
    text = text.replace('manifest_path = "../../cases/neural/standard-standalone-train.json"',
                        f'manifest_path = "{CASES}"')
    text = text.replace('output_root = "../../../artifacts/development/neural-combat/runs/gym-1.2.0/standard-standalone"',
                        f'output_root = "{ARTIFACTS / "runs/test-compact"}"')
    if rows is not None:
        text = text.replace("rows = 4\n", f"rows = {rows}\n")
    return _retarget(text, revision)


def dense_text(**kwargs) -> str:
    """The same template on the dense layout (the v1.2.0 dense identity, both Standard modes)."""
    revision = kwargs.get("revision", GYM_1_2_0)
    text = compact_text(**{**kwargs, "revision": GYM_1_2_0 if revision in EPOCH_1_2_4 else revision})
    lines = [line for line in text.splitlines()
             if not line.startswith(("standard_view_layout", "standard_view_preset"))]
    text = "\n".join(lines) + "\n"
    text = text.replace(f'"{COMPACT}"', f'"{DENSE}"').replace(
        f'standard-standalone = "{COMPACT_MAPPING}"',
        f'standard-standalone = "{DENSE_120_MAPPING}"\nstandard-run = "{DENSE_120_RUN_MAPPING}"')
    return _retarget(text, revision)


def load_text(text: str):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.toml"
        path.write_text(text)
        return load_neural_config(path)


class CompactConfigTests(unittest.TestCase):
    def test_compact_config_requires_gym_1_2_0(self):
        config = load_neural_config(COMPACT_TEMPLATE)
        self.assertEqual((config.standard_view_layout, config.standard_view_preset, config.standard_view_fingerprint),
                         ("compact", "standard-combat-compact-v1", COMPACT))
        self.assertEqual(dict(config.accepted_identities[0].accepted_raw_mode_mappings),
                         {"standard-standalone": COMPACT_MAPPING})
        self.assertEqual(dict(GYM_V1_2_0_CONTRACT.raw_mode_mappings_for_view(COMPACT)),
                         {"standard-standalone": COMPACT_MAPPING})
        play = load_text(compact_text(preset="standard-combat-compact-play-v1", fingerprint=COMPACT_PLAY,
                                      mapping=COMPACT_PLAY_MAPPING))
        self.assertEqual((play.standard_view_preset, play.standard_view_fingerprint),
                         ("standard-combat-compact-play-v1", COMPACT_PLAY))
        # gym-1.1.1 (and every earlier release) is dense only.
        with self.assertRaisesRegex(ValueError, "compact"):
            load_text(compact_text(revision=GYM_1_1_1, mapping=COMPACT_MAPPING.replace("1.2.0", "1.1.1")))
        refused = {
            "dense fingerprint": compact_text(fingerprint=DENSE),
            "other preset's fingerprint": compact_text(fingerprint=COMPACT_PLAY),
            "unknown preset": compact_text(preset="standard-combat-compact-v9"),
            "unknown layout": compact_text().replace('standard_view_layout = "compact"',
                                                     'standard_view_layout = "sparse"'),
            "no preset": compact_text().replace('standard_view_preset = "standard-combat-compact-v1"\n', ""),
            "dense mapping": compact_text(mapping=DENSE_120_MAPPING),
            "standard-run mapping": compact_text().replace(
                f'standard-standalone = "{COMPACT_MAPPING}"',
                f'standard-standalone = "{COMPACT_MAPPING}"\nstandard-run = "{DENSE_120_RUN_MAPPING}"'),
            "preset on the dense layout": dense_text().replace(
                f'standard_view_fingerprint = "{DENSE}"',
                f'standard_view_fingerprint = "{DENSE}"\nstandard_view_preset = "standard-combat-compact-v1"'),
        }
        for name, text in refused.items():
            with self.subTest(name), self.assertRaises(ValueError):
                load_text(text)
        # standard-run reads the Run's combat projection, which is dense only.
        run_text = (ROOT / "configs/neural/gym-1.2.0/standard-run.toml").read_text().replace(
            f'standard_view_fingerprint = "{DENSE}"',
            f'standard_view_layout = "compact"\nstandard_view_preset = "standard-combat-compact-v1"\n'
            f'standard_view_fingerprint = "{COMPACT}"')
        run_text = run_text.replace('manifest_path = "../../cases/neural/standard-run-train.json"',
                                    f'manifest_path = "{ROOT / "configs/cases/neural/standard-run-train.json"}"')
        run_text = run_text.replace(
            'output_root = "../../../artifacts/development/neural-combat/runs/gym-1.2.0/standard-run"',
            f'output_root = "{ARTIFACTS / "runs/test-compact"}"')
        with self.assertRaisesRegex(ValueError, "compact"):
            load_text(run_text)
        # The default stays dense: no layout key, no preset.
        dense = load_text(dense_text())
        self.assertEqual((dense.standard_view_layout, dense.standard_view_preset), ("dense", None))
        explicit = load_text(dense_text().replace(f'standard_view_fingerprint = "{DENSE}"',
                                                  f'standard_view_layout = "dense"\nstandard_view_fingerprint = "{DENSE}"'))
        self.assertEqual(explicit.standard_view_layout, "dense")

    def test_compact_identity_bound_in_checkpoint(self):
        compact = load_neural_config(COMPACT_TEMPLATE).accepted_identities[0]
        dense = load_text(dense_text()).accepted_identities[0]
        play = load_text(compact_text(preset="standard-combat-compact-play-v1", fingerprint=COMPACT_PLAY,
                                      mapping=COMPACT_PLAY_MAPPING)).accepted_identities[0]
        old = load_neural_config(ROOT / "configs/neural/gym-1.1.1/standard-standalone.toml").accepted_identities[0]
        with patch("builtins.__import__", side_effect=_refuse_torch):
            validate_checkpoint_metadata(_metadata(identity=compact), compact, purpose="evaluate")
            for saved, expected in ((dense, compact), (compact, dense), (play, compact), (compact, play),
                                    (old, compact), (compact, old)):
                with self.subTest(saved=dict(saved.accepted_raw_mode_mappings),
                                  expected=dict(expected.accepted_raw_mode_mappings)):
                    with self.assertRaisesRegex(CheckpointCompatibilityError, "raw-mode mapping"):
                        validate_checkpoint_metadata(_metadata(identity=saved), expected, purpose="evaluate")
        # Experiment records and the stream projection name the layout, preset and fingerprint.
        from spire_learn.neural.runner import _manifest_projection
        config = load_neural_config(COMPACT_TEMPLATE)
        record = _manifest_projection(config, SimpleNamespace(), policy_kind="neural",
                                      action_selection="sampled", allow_final=False)
        self.assertEqual((record["standard_view_layout"], record["standard_view_preset"],
                          record["standard_view_fingerprint"], record["standard_view_schema_id"]),
                         ("compact", "standard-combat-compact-v1", COMPACT,
                          "spire-gym-standard-combat-view-compact-v1"))
        self.assertEqual(record["learner_identity"]["accepted_raw_mode_mappings"],
                         {"standard-standalone": COMPACT_MAPPING})
        projection = _stream_runtime_projection(config)
        self.assertEqual((projection["standard_view_layout"], projection["standard_view_preset"]),
                         ("compact", "standard-combat-compact-v1"))
        # Dense v1.1.1 and dense v1.2.0 standard-standalone identities do not cross either.
        with patch("builtins.__import__", side_effect=_refuse_torch):
            for saved, expected in ((old, dense), (dense, old)):
                with self.subTest(dense_epochs=True), self.assertRaisesRegex(
                        CheckpointCompatibilityError, "raw-mode mapping"):
                    validate_checkpoint_metadata(_metadata(identity=saved), expected, purpose="evaluate")

    def test_mappings_for_view_accept_only_known_fingerprints(self):
        contract = GYM_V1_2_0_CONTRACT
        self.assertEqual(dict(contract.raw_mode_mappings_for_view(None)), dict(contract.raw_mode_mappings))
        self.assertEqual(dict(contract.raw_mode_mappings_for_view(DENSE)), dict(contract.raw_mode_mappings))
        self.assertEqual(dict(contract.raw_mode_mappings_for_view(COMPACT_PLAY)),
                         {"standard-standalone": COMPACT_PLAY_MAPPING})
        for fingerprint in ("0" * 64, contract.common_fingerprint):
            with self.subTest(fingerprint=fingerprint), self.assertRaises(ValueError):
                contract.raw_mode_mappings_for_view(fingerprint)
        with self.assertRaises(ValueError):  # dense-only releases have no compact presets
            GYM_V1_1_1_CONTRACT.raw_mode_mappings_for_view(COMPACT)

    def test_baselines_refuse_the_play_preset(self):
        from scripts.neural.baseline_entry import _refuse_overflowing_preset
        _refuse_overflowing_preset(COMPACT_TEMPLATE)
        _refuse_overflowing_preset(ROOT / "configs/neural/gym-1.1.1/standard-stream.toml")
        with self.assertRaisesRegex(ValueError, "play compact preset"):
            _refuse_overflowing_preset(ROOT / "configs/neural/profile-standard-1.2.0/template-16-play.toml")


class CompactSchemaTests(unittest.TestCase):
    def test_compact_descriptors_parse_as_the_compact_view(self):
        from spire_learn.neural.views import STANDARD_COMBAT_COMPACT_V1, STANDARD_COMBAT_V1
        dense = parse_combat_descriptor(_fixture("standard-combat-1929e18.json"), revision=GYM_1_2_0)
        for name, fingerprint in (("standard-combat-compact-4d19806.json", COMPACT),
                                  ("standard-combat-compact-play-4d19806.json", COMPACT_PLAY)):
            with self.subTest(name):
                schema = parse_combat_descriptor(_fixture(name), revision=GYM_1_2_0)
                self.assertIs(schema.view, STANDARD_COMBAT_COMPACT_V1)
                self.assertEqual(schema.schema_fingerprint, fingerprint)
                with self.assertRaises(ContractError):
                    parse_combat_descriptor(_fixture(name), revision=GYM_1_1_1)
                # Same groups, columns, vocabularies and row-group order as the dense view.
                self.assertEqual(tuple(schema.tensors), tuple(dense.tensors))
                self.assertEqual(dict(schema.vocabularies), dict(dense.vocabularies))
                for key, spec in schema.tensors.items():
                    other = dense.tensors[key]
                    self.assertEqual([(c.field, c.unit, c.vocabulary, c.reference) for c in spec.columns],
                                     [(c.field, c.unit, c.vocabulary, c.reference) for c in other.columns])
                    self.assertEqual(spec.shape[1:], other.shape[1:])
                    self.assertLessEqual(spec.shape[0], other.shape[0])
                    if spec.columns:
                        self.assertEqual(STANDARD_COMBAT_COMPACT_V1.dense_rows.get(key, 1),
                                         other.shape[0] if len(other.shape) == 2 else 1)
                self.assertEqual(STANDARD_COMBAT_COMPACT_V1.dense_action_width, dense.action_space.n)
        self.assertIs(dense.view, STANDARD_COMBAT_V1)
        tampered = _fixture("standard-combat-compact-4d19806.json")
        tampered["tensors"][0]["dtype"] = "int64"
        with self.assertRaises(ContractError):
            parse_combat_descriptor(tampered, revision=GYM_1_2_0)

    def test_compact_model_plan_matches_dense(self):
        from spire_learn.neural.contracts import ModelConfig
        from spire_learn.neural.model import _compile_model_plan
        model = ModelConfig(embedding_width=16, hidden_width=64, max_parameters=10_000_000)
        dense = _compile_model_plan(parse_combat_descriptor(_fixture("standard-combat-1929e18.json"),
                                                            revision=GYM_1_2_0), model)
        for name in ("standard-combat-compact-4d19806.json", "standard-combat-compact-play-4d19806.json"):
            plan = _compile_model_plan(parse_combat_descriptor(_fixture(name), revision=GYM_1_2_0), model)
            # Parameter shapes depend on columns and vocabularies only, never on row capacity.
            self.assertEqual(plan.parameter_count, dense.parameter_count)
            self.assertEqual(dict(plan.feature_contract.groups), dict(dense.feature_contract.groups))
            self.assertEqual(dict(plan.row_input_widths), dict(dense.row_input_widths))
            self.assertEqual(plan.reference_plans, dense.reference_plans)
            self.assertEqual(plan.global_input_width, dense.global_input_width)


# ---------------------------------------------------------------- fake compact vector environment

_CODES = {"int8": "b", "int16": "h", "int32": "i", "int64": "q"}


class _TypedArray(standard_env_tests._Array):
    def __init__(self, values, shape, dtype):
        super().__init__(values, shape, dtype)
        self.dtype = _DType(dtype, "i", struct.calcsize("=" + _CODES[dtype]))

    def tobytes(self, order="C"):
        assert order == "C"
        return struct.pack("=" + _CODES[self.dtype.name] * len(self.values), *self.values)


class FakeCompactVector(standard_env_tests.FakeStandardVector):
    descriptor = "standard-combat-compact-4d19806.json"
    calls: list

    def __init__(self, catalog, num_envs, *, observation="dense", capacity_preset=None, capacities=None, **kwargs):
        super().__init__(catalog, num_envs, **kwargs)
        self.layout = {"observation": observation, "capacity_preset": capacity_preset, "capacities": capacities}
        self.calls = []

    def observation_spec(self):
        return _fixture(self.descriptor)

    def environment_spec(self):
        return _fixture("standard-combat-compact-environment-4d19806.json") | {
            "profile_id": self.profile_id, "entry_catalog_fingerprint":
                _fixture("standard-combat-environment-1929e18.json")["entry_catalog_fingerprint"],
            "rules_id": GYM_V1_2_0_CONTRACT.standard_rules_id}

    def _info(self):
        info = super()._info()
        info["combat_view_id"] = [standard_env_tests._NumpyScalar("spire-gym-standard-combat-view-compact-v1")
                                  ] * self.num_envs
        return info

    def _obs(self):
        with patch.object(standard_env_tests, "_Array", _TypedArray):
            return super()._obs()

    def reset(self, *, seed=None, options=None):
        self.calls.append("reset")
        return super().reset(seed=seed, options=options)

    def step(self, actions):
        self.calls.append("step")
        return super().step(actions)


class FakeDenseVector(standard_env_tests.FakeStandardVector):
    """The dense fake, recording any layout keyword (a dense config must pass none)."""

    def __init__(self, catalog, num_envs, **kwargs):
        layout = {key: kwargs.pop(key) for key in ("observation", "capacity_preset", "capacities") if key in kwargs}
        super().__init__(catalog, num_envs, **kwargs)
        self.layout = layout


def _capacity_fault():
    """The Gym's capacity fault: a batch fault whose retained cause names the overflowing group."""
    error = standard_env_tests.FakeBatchFaultError([
        {"status": "faulted", "stage": "reset", "committed": False, "recoverable": False,
         "message": "Run capacity exceeded",
         "capacity": {"group": "powers", "required": 129, "capacity": 128}}])
    return error


class CompactDriverTests(unittest.TestCase):
    def _driver(self, config, vector_class):
        made = []

        class Vector(vector_class):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                inner._parsed_schema = parse_combat_descriptor(inner.observation_spec(),
                                                               revision=config.environment_revision)
                made.append(inner)

        native = SimpleNamespace(
            build_info=lambda: {"extension": "spire-python", "revision": config.environment_revision,
                                "dirty": False, "profile": "release",
                                "source_digest": GYM_V1_2_0_CONTRACT.build.source_digest
                                if config.environment_revision == GYM_1_2_0 else GYM_V1_1_1_CONTRACT.build.source_digest,
                                "file_digest": GYM_V1_2_0_CONTRACT.build.file_digest
                                if config.environment_revision == GYM_1_2_0 else GYM_V1_1_1_CONTRACT.build.file_digest,
                                "features": ["spire-python/extension"]},
            CombatInputError=standard_env_tests.FakeCombatInputError,
            CombatStateError=standard_env_tests.FakeStateError,
            RunInputError=standard_env_tests.FakeInputError, RunStateError=standard_env_tests.FakeStateError,
            RunFaultError=standard_env_tests.FakeRunFaultError,
            BatchFaultError=standard_env_tests.FakeBatchFaultError)
        runtime = envs._NativeRuntime(standard_env_tests._NumpyStub(), native, object, object, Vector)
        with patch.object(envs, "_load_native_runtime", return_value=runtime):
            driver = envs.make_driver(config, config.cases)
        return driver, made[0]

    def test_compact_driver_selects_the_compact_layout(self):
        from spire_learn.neural.views import STANDARD_COMBAT_COMPACT_V1
        config = load_neural_config(COMPACT_TEMPLATE)
        driver, fake = self._driver(config, FakeCompactVector)
        self.assertEqual(fake.layout, {"observation": "compact", "capacity_preset": "standard-combat-compact-v1",
                                       "capacities": None})
        self.assertIs(driver.schema.view, STANDARD_COMBAT_COMPACT_V1)
        rows = driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
        self.assertEqual(rows[0].frame.schema_fingerprint, COMPACT)
        self.assertEqual(rows[0].frame.tensors["powers"].dtype, "int32")
        self.assertEqual(rows[0].frame.tensors["powers"].shape, (128, 4))
        validate_frame(rows[0].frame, driver.schema)
        driver.step([0] * driver.batch_size)
        driver.close()
        # A dense descriptor offered to a compact config is refused before any reset.
        with self.assertRaises(ContractError):
            self._driver(config, FakeDenseVector)

    def test_capacity_fault_is_an_error(self):
        config = load_neural_config(COMPACT_TEMPLATE)
        for phase in ("reset", "step"):
            with self.subTest(phase=phase):
                driver, fake = self._driver(config, FakeCompactVector)
                if phase == "step":
                    driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
                fake.next_error = _capacity_fault()
                calls = len(fake.calls)
                with self.assertRaises(envs.DriverFault) as caught:
                    if phase == "reset":
                        driver.reset(seeds=[1] * driver.batch_size, entry_ids=[None] * driver.batch_size)
                    else:
                        driver.step([0] * driver.batch_size)
                self.assertEqual(caught.exception.evidence.exception_type, "FakeBatchFaultError")
                # Not retried, not a loss or a truncation: the driver is frozen.
                self.assertEqual(len(fake.calls), calls + 1)
                with self.assertRaises(envs.DriverStateError):
                    driver.step([0] * driver.batch_size)
                self.assertEqual(len(fake.calls), calls + 1)

    def test_dense_gym_1_1_1_paths_unchanged(self):
        for path, pinned in V1_1_1_CONFIG_PINS.items():
            with self.subTest(path):
                config = load_neural_config(ROOT / path)
                self.assertEqual(config.config_sha256, pinned)
                self.assertEqual((config.standard_view_layout, config.standard_view_preset), ("dense", None))
                self.assertEqual(config.standard_view_fingerprint, DENSE)
                self.assertEqual(dict(config.accepted_identities[0].accepted_raw_mode_mappings), V1_1_1_MAPPINGS)
                self.assertNotIn("standard_view_layout", _stream_runtime_projection(config))
        self.assertEqual(dict(GYM_V1_1_1_CONTRACT.raw_mode_mappings)["standard-standalone"],
                         V1_1_1_MAPPINGS["standard-standalone"])
        for contract in CONTRACTS:
            self.assertEqual(dict(contract.raw_mode_mappings_for_view(contract.standard_combat_fingerprint)),
                             dict(contract.raw_mode_mappings))
            self.assertEqual(dict(contract.raw_mode_mappings_for_view(None)), dict(contract.raw_mode_mappings))
        # A dense config constructs the vector environment with no layout keyword at all.
        config = load_neural_config(ROOT / "configs/neural/gym-v1.0.0/standard-standalone.toml")
        made = []

        class Vector(FakeDenseVector):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                inner._parsed_schema = parse_combat_descriptor(inner.observation_spec(),
                                                               revision=config.environment_revision)
                made.append(inner)

        from tests.unit.test_neural_gym_v1 import PUBLISHED_BUILD
        native = SimpleNamespace(
            build_info=lambda: dict(PUBLISHED_BUILD),
            CombatInputError=standard_env_tests.FakeCombatInputError,
            CombatStateError=standard_env_tests.FakeStateError,
            RunInputError=standard_env_tests.FakeInputError, RunStateError=standard_env_tests.FakeStateError,
            RunFaultError=standard_env_tests.FakeRunFaultError,
            BatchFaultError=standard_env_tests.FakeBatchFaultError)
        runtime = envs._NativeRuntime(standard_env_tests._NumpyStub(), native, object, object, Vector)
        with patch.object(envs, "_load_native_runtime", return_value=runtime):
            driver = envs.make_driver(config, config.cases)
        self.assertEqual(made[0].layout, {})
        driver.close()
        # The existing Standard stream pins (shard bytes, shipped-recipe shard 0, panel hash).
        names = ("tests.unit.test_neural_standard_stream.ShardTest.test_shard_bytes_pinned",
                 "tests.unit.test_neural_standard_stream.ShardTest.test_shipped_recipe_shard_zero_pinned",
                 "tests.unit.test_neural_standard_stream.PanelTest.test_panel_deterministic_hash")
        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromNames(names).run(result)
        self.assertEqual((result.testsRun, result.failures, result.errors), (3, [], []))


# ---------------------------------------------------------------- gated: real Gym v1.2.0 and torch

def _widened(buffer, dense_shape):
    """A compact buffer widened to int64 (values) and zero-padded on the row axis, as raw bytes."""
    code = _CODES[buffer.dtype]
    values = struct.unpack(f"={len(buffer.data) // struct.calcsize('=' + code)}{code}", buffer.data)
    dense_code = "b" if buffer.dtype == "int8" else "q"
    if len(buffer.shape) == len(dense_shape) == 2 or (len(buffer.shape) == 1 and dense_shape != buffer.shape):
        row = 1 if len(buffer.shape) == 1 else buffer.shape[1]
        values = values + (0,) * ((dense_shape[0] - buffer.shape[0]) * row)
    return struct.pack(f"={len(values)}{dense_code}", *values)


def _cases_manifest(directory: Path, entries: list[dict]) -> Path:
    cases = [{"case_id": f"compact-check-{index}", "lineage_id": f"lineage:compact-check:{index}",
              "mode": "standard-standalone", "environment_seed": index, "cohort": "standard_combat",
              "entry": entry} for index, entry in enumerate(entries)]
    path = directory / "train.json"
    path.write_text(json.dumps({"schema": "spire-learn-neural-cases-v1", "split": "train", "partition": "train",
                                "cases": cases}))
    return path


TABLE_1_2_0 = ARTIFACTS / "run-states/run-states-gym-1.2.0.json"
TABLE_1_2_4 = ARTIFACTS / "run-states/run-states-gym-1.2.4.json"
# The live runtime's revision -> its table; the v1.2.5 drop-in plays the gym-1.2.4 table.
TABLES = {GYM_1_2_0: TABLE_1_2_0, GYM_1_2_4: TABLE_1_2_4, GYM_1_2_5: TABLE_1_2_4}
CHARACTERS = ("ironclad", "silent", "defect", "watcher")


def _table_states(revision: str = GYM_1_2_0) -> list[dict]:
    return json.loads(TABLES[revision].read_text())["states"]


def diverse_entries(states: list[dict], per_character: int = 8) -> list[dict]:
    """A deterministic, diverse sample of real v1.2.0 entries for each character.

    Per character (states in (run_id, floor) order): the largest deck, the most relics, the most
    filled potion slots, the latest act, then evenly strided states up to ``per_character``.
    """
    chosen = []
    for character in CHARACTERS:
        rows = sorted((s for s in states if s["character"] == character),
                      key=lambda s: (s["run_id"], s["floor"]))
        picks = [max(rows, key=lambda s: len(s["entry"]["deck"])),
                 max(rows, key=lambda s: len(s["entry"]["relics"])),
                 max(rows, key=lambda s: sum(p is not None for p in s["entry"]["potions"])),
                 max(rows, key=lambda s: (s["act"], s["floor"]))]
        stride = max(1, len(rows) // per_character)
        picks += rows[stride // 2::stride]
        seen = []
        for state in picks:
            if state["entry"]["id"] not in [e["id"] for e in seen]:
                seen.append(state["entry"])
            if len(seen) == per_character:
                break
        chosen.extend(seen)
    return chosen


def near_capacity_entry(states: list[dict]) -> dict:
    """An Ironclad entry filling the play preset: 96 permanent cards and 32 relics, full potions."""
    from spire_learn.neural.entry_resources import load_entry_resources
    resources = load_entry_resources()
    ironclad = sorted((s for s in states if s["character"] == "ironclad"), key=lambda s: (s["run_id"], s["floor"]))
    base = max(ironclad, key=lambda s: (len(s["entry"]["relics"]), len(s["entry"]["deck"])))["entry"]
    refused = (set(resources.counter_required_relics()) | set(resources.bottle_relics())
               | {"potion_belt", "black_blood", "mark_of_pain"})
    counts = {}
    for state in ironclad:
        for relic in state["entry"]["relics"]:
            if relic not in refused and relic != "burning_blood" and resources.admits_relic("ironclad", relic):
                counts[relic] = counts.get(relic, 0) + 1
    relics = ["burning_blood"] + sorted(counts, key=lambda r: (-counts[r], r))[:31]
    deck = [dict(row) for row in base["deck"]]
    deck = (deck * (96 // len(deck) + 1))[:96]
    potions = next(s["entry"]["potions"] for s in ironclad
                   if len(s["entry"]["potions"]) == 2 and all(p is not None for p in s["entry"]["potions"]))
    entry = {key: value for key, value in base.items() if key != "bottled_cards"}  # no bottles here
    return {**entry, "id": "near-capacity-play", "deck": sorted(deck, key=lambda r: (r["card"], r["upgrades"])),
            "relics": relics, "relic_counters": {}, "potions": list(potions), "hp": base["max_hp"]}


@unittest.skipUnless(_RUN_NATIVE, "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the approved v1.2.0 runtime")
class GatedCompactTests(unittest.TestCase):
    ROWS = 8

    def setUp(self):
        from spire_lab import _native
        self.revision = _native.build_info().get("revision")
        if self.revision not in (GYM_1_2_0, *EPOCH_1_2_4):
            self.skipTest("the compact view needs the Gym v1.2.0, v1.2.4 or v1.2.5 runtime")
        self._tmp = tempfile.TemporaryDirectory(dir=WORK)
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _configs(self, entries, *, preset="standard-combat-compact-v1", rows=None):
        rows = self.ROWS if rows is None else rows
        manifest = _cases_manifest(self.dir, entries)
        fingerprint, mapping = ((COMPACT, COMPACT_MAPPING) if preset == "standard-combat-compact-v1"
                                else (COMPACT_PLAY, COMPACT_PLAY_MAPPING))
        texts = {"dense": dense_text(rows=rows, revision=self.revision),
                 "compact": compact_text(rows=rows, preset=preset, fingerprint=fingerprint, mapping=mapping,
                                         revision=self.revision)}
        configs = {}
        for name, text in texts.items():
            path = self.dir / f"{name}-{preset}.toml"
            path.write_text(text.replace(f'manifest_path = "{CASES}"', f'manifest_path = "{manifest}"'))
            configs[name] = load_neural_config(path)
        return configs

    def _compare(self, entries, *, preset, rows, steps, seed):
        """Step both layouts with the dense policy's sampled actions and compare everything.

        Every frame either layout returns (reset outputs, every step's successors including
        terminal ones, before any reset) is compared: widened bytes, encoded values/known/
        occupied/coverage/mask, critic values and logits. Each pre-step batch also compares
        sampled and deterministic ``act`` choices.
        """
        import torch
        from spire_learn.neural.features import encode_frames
        from spire_learn.neural.model import build_policy
        configs = self._configs(entries, preset=preset, rows=rows)
        drivers = {name: envs.make_driver(config, config.cases) for name, config in configs.items()}
        self.addCleanup(lambda: [driver.close() for driver in drivers.values()])
        compact_schema = drivers["compact"].schema
        width = compact_schema.action_space.n
        policies = {name: build_policy(drivers[name].schema, configs["dense"].model, seed=1103, device="cpu",
                                       model_id="spire-learn-combat-ff-v1",
                                       features_id="spire-learn-combat-features-v1") for name in drivers}
        for (name, tensor), (_, other) in zip(policies["dense"].state_dict().items(),
                                              policies["compact"].state_dict().items()):
            self.assertTrue(torch.equal(tensor, other), name)
        stats = {"frames_compared": 0, "terminal_frames_compared": 0, "decisions_compared": 0,
                 "states_by_character": {}, "max_rows": {}}
        character_of = {entry["id"]: entry["character"] for entry in entries}

        def compare(tag, dense_frames, compact_frames):
            for dense_frame, compact_frame in zip(dense_frames, compact_frames):
                for key, buffer in compact_frame.tensors.items():
                    self.assertEqual(_widened(buffer, dense_frame.tensors[key].shape),
                                     dense_frame.tensors[key].data, (tag, key))
            batches = {"dense": encode_frames(dense_frames, drivers["dense"].schema, device="cpu"),
                       "compact": encode_frames(compact_frames, compact_schema, device="cpu")}
            dense_batch, compact_batch = batches["dense"], batches["compact"]
            for group, values in compact_batch.values.items():
                count = values.shape[1]
                self.assertTrue(torch.equal(values, dense_batch.values[group][:, :count]), (tag, group))
                self.assertFalse(dense_batch.values[group][:, count:].any().item(), (tag, group))
                self.assertTrue(torch.equal(compact_batch.known[group], dense_batch.known[group][:, :count]))
                self.assertFalse(dense_batch.known[group][:, count:].any().item())
                if group in compact_batch.occupied:
                    occupied = compact_batch.occupied[group]
                    self.assertTrue(torch.equal(occupied, dense_batch.occupied[group][:, :count]))
                    stats["max_rows"][group] = max(stats["max_rows"].get(group, 0),
                                                   int(occupied.sum(dim=1).max().item()))
                self.assertEqual(compact_batch.known_coverage[group].numpy().tobytes(),
                                 dense_batch.known_coverage[group].numpy().tobytes(), (tag, group))
            self.assertTrue(torch.equal(compact_batch.action_mask, dense_batch.action_mask[:, :width]))
            self.assertFalse(dense_batch.action_mask[:, width:].any().item())
            with torch.no_grad():
                outputs = {name: policies[name](batches[name]) for name in batches}
            self.assertEqual(outputs["compact"].values.numpy().tobytes(),
                             outputs["dense"].values.numpy().tobytes(), tag)
            self.assertEqual(outputs["compact"].logits.numpy().tobytes(),
                             outputs["dense"].logits[:, :width].contiguous().numpy().tobytes(), tag)
            stats["frames_compared"] += len(dense_frames)

        ids = [entry["id"] for entry in entries]
        row_ids = [ids[index % len(ids)] for index in range(rows)]
        rows_now = {name: list(driver.reset(seeds=list(range(seed, seed + rows)), entry_ids=row_ids))
                    for name, driver in drivers.items()}
        compare("reset", [r.frame for r in rows_now["dense"]], [r.frame for r in rows_now["compact"]])
        next_entry = rows
        generators = {name: torch.Generator(device="cpu").manual_seed(seed) for name in drivers}
        for step in range(steps):
            frames = {name: [row.frame for row in rows_now[name]] for name in drivers}
            sampled = {name: policies[name].act(frames[name], generator=generators[name]) for name in drivers}
            self.assertEqual(sampled["compact"], sampled["dense"], step)
            greedy = {name: policies[name].act(frames[name], generator=None, deterministic=True)
                      for name in drivers}
            self.assertEqual(greedy["compact"], greedy["dense"], step)
            stats["decisions_compared"] += rows
            for entry_id in row_ids:
                character = character_of[entry_id]
                stats["states_by_character"][character] = stats["states_by_character"].get(character, 0) + 1
            actions = [choice.original_action_index for choice in sampled["dense"]]
            transitions = {name: driver.step(actions) for name, driver in drivers.items()}
            ended = [t.terminated or t.truncated for t in transitions["dense"]]
            self.assertEqual([t.terminated or t.truncated for t in transitions["compact"]], ended)
            self.assertEqual([t.native_reward for t in transitions["compact"]],
                             [t.native_reward for t in transitions["dense"]])
            compare(f"step-{step}", [t.successor.frame for t in transitions["dense"]],
                    [t.successor.frame for t in transitions["compact"]])
            stats["terminal_frames_compared"] += sum(ended)
            rows_now = {name: [t.successor for t in transitions[name]] for name in drivers}
            if any(ended):
                reset_ids = []
                for index, flag in enumerate(ended):
                    if flag:
                        row_ids[index] = ids[next_entry % len(ids)]
                        next_entry += 1
                    reset_ids.append(row_ids[index] if flag else None)
                for name, driver in drivers.items():
                    rows_now[name] = list(driver.reset(seeds=[seed + 1000 + step * rows + i for i in range(rows)],
                                                       entry_ids=reset_ids, reset_mask=list(ended)))
                compare(f"reset-{step}", [r.frame for r in rows_now["dense"]],
                        [r.frame for r in rows_now["compact"]])
        return stats

    def test_compact_and_dense_give_identical_policy_outputs(self):
        """Real entries (the live runtime's own table) for all four characters: byte-identical on CPU."""
        table = TABLES.get(self.revision)
        if table is None or not table.is_file():
            self.skipTest("the local run-states table for this runtime's revision is absent")
        states = _table_states(self.revision)
        evidence = {"bounds": self._compare(diverse_entries(states), preset="standard-combat-compact-v1",
                                            rows=self.ROWS, steps=300, seed=500)}
        for character in CHARACTERS:
            self.assertGreater(evidence["bounds"]["states_by_character"].get(character, 0), 0, character)
        self.assertGreater(evidence["bounds"]["terminal_frames_compared"], 0)
        # The play preset at its capacity edge: 96 permanent cards and 32 relics fill it exactly.
        small = [s["entry"] for s in sorted(states, key=lambda s: (s["run_id"], s["floor"]))
                 if s["character"] == "ironclad" and len(s["entry"]["deck"]) <= 40
                 and len(s["entry"]["relics"]) <= 12][:3]
        edge = near_capacity_entry(states)
        evidence["play"] = self._compare([edge] + small, preset="standard-combat-compact-play-v1",
                                         rows=4, steps=120, seed=900)
        self.assertEqual(evidence["play"]["max_rows"]["permanent_cards"], 96)
        self.assertEqual(evidence["play"]["max_rows"]["relics"], 32)
        (WORK / ("compact-identical-outputs.json" if self.revision == GYM_1_2_0 else f"compact-identical-outputs-{self.revision[:8]}.json")).write_text(json.dumps(evidence, sort_keys=True) + "\n")

    def test_capacity_overflow_is_a_driver_fault(self):
        """The play preset holds 96 permanent cards; a 120-card deck overflows at reset: an error, not a loss."""
        from tests.unit.test_neural_standard_stream_runtime import smoke_states
        entry = dict(smoke_states(runs=1)[0]["entry"])
        entry["deck"] = [{"card": "strike", "upgrades": 0, "misc": 0}] * 120
        entry["id"] = "compact-overflow"
        config = self._configs([entry], preset="standard-combat-compact-play-v1")["compact"]
        driver = envs.make_driver(config, config.cases)
        try:
            with self.assertRaises(envs.DriverFault) as caught:
                driver.reset(seeds=[1] * driver.batch_size, entry_ids=["compact-overflow"] * driver.batch_size)
            self.assertIn("FaultError", caught.exception.evidence.exception_type)
            with self.assertRaises(envs.DriverStateError):
                driver.reset(seeds=[1] * driver.batch_size, entry_ids=["compact-overflow"] * driver.batch_size)
        finally:
            driver.close()


if __name__ == "__main__":
    unittest.main()
