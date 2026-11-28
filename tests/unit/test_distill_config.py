"""Distillation config, identities and hash sites; dataset discovery and checkpoint choice (stdlib only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spire_learn.contracts import ContractError
from spire_learn.neural import config as config_module
from spire_learn.neural.checkpoint import (
    DISTILL_COUNTER_FIELDS, CheckpointCompatibilityError, identity_payload, validate_checkpoint_metadata,
)
from spire_learn.neural.config import (
    DISTILL_OBJECTIVE_ID, STANDARD_LEARNER_IDENTITY_IDS, DistillHyperparameters, _canonical_hash,
    _stream_runtime_projection, distill_config_hash, load_distill_config, load_neural_config,
    model_config_hash, model_config_hash_payload,
)
from spire_learn.neural.contracts import ModelConfig, NeuralIdentity
from spire_learn.neural.distill_train import (
    EvalRecord, build_dataset_manifest, choose_checkpoint, discover_dataset, domain_seed, load_dataset_manifest,
    open_dataset, run_level_se, timing_summary,
)
from spire_learn.neural.policy_player import model_config_hash as player_model_config_hash
from spire_learn.neural.runner import _config_projection

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/development/neural-combat"
GYM = "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1"
COMPACT = "49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8"
MAPPING = f"spire-learn-standard-combat-compact-v1@{COMPACT}+gym-1.2.4"
FF_HASH = "4fb628e06bf4952847795ac5d8548d12dfeede4c97e874ab843857ae0f91cd2f"
FF_PAYLOAD = {"embedding_width": 16, "hidden_width": 64, "max_parameters": 10_000_000}
# Recorded at base 0e5985a (before value_heads existed): config sha256, the canonical hash of the
# stream runtime projection, and the sha256 of runner._config_projection serialized in insertion
# order without its two absolute paths.
PINS = {
    "configs/neural/gym-1.2.5/standard-standalone.toml": (
        "19c37fbc12e952466cd5b5d85bcd77b5f2c0c441814e4e0b112d207452e83f23",
        "9b81624d8e4b8e778ff9090945c01bc29af2b199aeb3c47771f8d9777a39ec81",
        "e3930f4e08f40fe4b940bb8700a5851098bc3a32f83e3484080d4df06d1b39dd"),
    "configs/neural/gym-1.2.0/standard-standalone.toml": (
        "fbee35dd59d3efae8162ac47dafc30866155ce75f7789d073278824070c06a04",
        "6995403dcd19f340f8b214e323f7342725bea64209654533c0915d3c1129d8ee",
        "480e8af919658d0273f98e715771d879a5fc6760101feea7a8db3c0869522f3a"),
    "configs/neural/standalone-selection.toml": (
        "3cd945710330a867a8cd5ba312e31b9f37c188ec91ba3157a31d244f9002a70e",
        "6e5c55fab281618322357fd594aff9bd7b70ca80682ab50de1d61a526517de58",
        "416332f0adaaf457ef736358cffa26dcc8351473acb1330f826dd2d70f48c7f1"),
    "configs/neural/gym-1.2.5/run.toml": (
        "5b98739dfc013962a8ed2a945763f27e6eee99971346be2b06fe0c802eef5097",
        "becadb8a8330484394aa0676537c0364fa4dc1806194f6f7d7ce97d889680872",
        "98843d0c9784161922dd02a6eaf5547987cc408b0e559cd7cc295da40e58b791"),
}
STUDENT = ModelConfig(embedding_width=16, hidden_width=64, max_parameters=10_000_000, value_heads=3)
HYPER = DistillHyperparameters(tau=0.05, learning_rate=0.0003, batch_rows=512, gradient_steps=18000,
                               data_fraction=1.0)


def _ordered_projection_sha256(config) -> str:
    projection = _config_projection(config)
    projection.pop("manifest_path")
    projection.pop("output_root")
    return hashlib.sha256(json.dumps(projection).encode()).hexdigest()


def _write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text)
    return path


def _identity_toml(*, objective=DISTILL_OBJECTIVE_ID, hashes: dict[str, str], table="identity",
                   learner=STANDARD_LEARNER_IDENTITY_IDS) -> str:
    model_id, features_id, action_id = learner
    lines = [f"[{table}]" if table == "identity" else f"[[{table}]]",
             f'model_id = "{model_id}"', f'features_id = "{features_id}"', f'action_id = "{action_id}"',
             f'objective_id = "{objective}"', "", f"[{table}.accepted_raw_mode_mappings]",
             f'standard-standalone = "{MAPPING}"', "", f"[{table}.config_hashes]"]
    lines += [f'{name} = "{digest}"' for name, digest in hashes.items()]
    return "\n".join(lines) + "\n"


def distill_toml(*, model=STUDENT, hyper=HYPER, seed=1, eval_every=1000,
                 extra="", hashes=None, microbatch_rows=None) -> str:
    hashes = hashes or {"model": model_config_hash(model), "distill": distill_config_hash(hyper)}
    microbatch = "" if microbatch_rows is None else f"microbatch_rows = {microbatch_rows}\n"
    return (
        'schema = "spire-learn-distill-config-v1"\nmode = "standard-standalone"\n'
        f'output_root = "{ARTIFACTS / "distill/train/test"}"\ndevice = "cpu"\n'
        f"seed = {seed}\neval_every = {eval_every}\n{extra}\n"
        f'[environment]\ngym_revision = "{GYM}"\nstandard_view_layout = "compact"\n'
        f'standard_view_preset = "standard-combat-compact-v1"\nstandard_view_fingerprint = "{COMPACT}"\n\n'
        f"[model]\nembedding_width = {model.embedding_width}\nhidden_width = {model.hidden_width}\n"
        f"max_parameters = {model.max_parameters}\nvalue_heads = {model.value_heads}\n\n"
        f"[distill]\ntau = {hyper.tau}\nlearning_rate = {hyper.learning_rate}\nbatch_rows = {hyper.batch_rows}\n"
        f"gradient_steps = {hyper.gradient_steps}\ndata_fraction = {hyper.data_fraction}\n{microbatch}\n"
        + _identity_toml(hashes=hashes))


def eval_toml(folder: Path, *, model_table: str | None = None, hashes=None, ppo=False,
              objective=DISTILL_OBJECTIVE_ID, operation="evaluate", split="selection",
              learner=STANDARD_LEARNER_IDENTITY_IDS) -> str:
    """The v1.2.5 compact template as a selection-split distillation evaluation config."""
    manifest = json.loads((ROOT / "configs/cases/neural/standard-standalone-train.json").read_text())
    if split == "selection":
        from spire_learn.neural import cases
        manifest.update(split="selection", partition="selection")
        manifest["cases"][0].update(environment_seed=cases._SELECTION_START, case_id="standard-selection-distill-eval",
                                    lineage_id="lineage:standard:selection:standalone:ironclad:a0")
    _write(folder, "selection.json", json.dumps(manifest))
    text = (ROOT / "configs/neural/gym-1.2.5/standard-standalone.toml").read_text()
    head, _, _ = text.partition("[[accepted_identities]]")
    head = head.replace('operation = "train"', f'operation = "{operation}"')
    head = head.replace('split = "train"\npartition = "train"', f'split = "{split}"\npartition = "{split}"')
    head = head.replace('manifest_path = "../../cases/neural/standard-standalone-train.json"',
                        f'manifest_path = "{folder / "selection.json"}"')
    head = head.replace('output_root = "../../../artifacts/development/neural-combat/runs/gym-1.2.5/standard-standalone"',
                        f'output_root = "{ARTIFACTS / "runs/test-distill-eval"}"')
    head = head.replace("rows = 4\nsteps = 32", "rows = 1\nsteps = 1")
    if not ppo:
        before, _, after = head.partition("[ppo]")
        head = before + after[after.index("[rollout]"):]
    head = head.replace("max_parameters = 10000000\n", "max_parameters = 10000000\nvalue_heads = 3\n")
    if model_table is not None:
        before, _, after = head.partition("[model]\n")
        head = before + model_table + after[after.index("\n\n") + 2:]
    hashes = hashes or {"model": model_config_hash(STUDENT)}
    return head + _identity_toml(objective=objective, hashes=hashes, table="accepted_identities", learner=learner)


def write_chunk(root: Path, first: int, count: int, rows=lambda index: 3, validation_rows=lambda index: 1) -> Path:
    """A fake finished chunk: manifests only (discovery never opens the arrays)."""
    chunk = root / f"chunk-{first}"
    shards = {"train": {}, "validation": {}}
    for series, counter in (("train", rows), ("validation", validation_rows)):
        for index in range(first, first + count):
            name = f"shard-{index:06d}"
            path = chunk / series / name
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(json.dumps({"rows": counter(index), "files": {}}))
            shards[series][name] = {"manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
                                    "rows": counter(index)}
    (chunk / "chunk-manifest.json").write_text(json.dumps({
        "schema": "spire-learn-distill-label-chunk-v1", "first_shard": first, "shard_count": count,
        "observation": {"schema_fingerprint": COMPACT}, "shards": shards}))
    return chunk


class HashSiteTests(unittest.TestCase):
    def test_existing_hashes_unchanged_at_every_site(self):
        for name, (config_sha, stream_sha, projection_sha) in PINS.items():
            with self.subTest(config=name):
                config = load_neural_config(ROOT / name)
                self.assertEqual(config.config_sha256, config_sha)
                # The config model hash site (identity hash parsed and checked against the TOML).
                self.assertEqual(config.accepted_identities[0].config_hashes["model"], FF_HASH)
                self.assertEqual(model_config_hash(config.model), FF_HASH)
                # _stream_runtime_projection and runner._config_projection (keys and order).
                self.assertEqual(_canonical_hash(_stream_runtime_projection(config)), stream_sha)
                self.assertEqual(_ordered_projection_sha256(config), projection_sha)
                self.assertNotIn("value_heads", _stream_runtime_projection(config)["model"])
                self.assertIsNotNone(config.ppo)
        model = ModelConfig(16, 64, 10_000_000)
        # The player's identity hash and the labeller/full-run provenance payload.
        self.assertEqual(player_model_config_hash(model), FF_HASH)
        self.assertEqual(model_config_hash_payload(model), FF_PAYLOAD)
        self.assertEqual(_canonical_hash(FF_PAYLOAD), FF_HASH)

    def test_value_heads_one_hashes_as_absent(self):
        explicit = ModelConfig(16, 64, 10_000_000, value_heads=1)
        self.assertEqual(explicit, ModelConfig(16, 64, 10_000_000))
        self.assertEqual(model_config_hash(explicit), FF_HASH)
        self.assertNotEqual(model_config_hash(STUDENT), FF_HASH)
        self.assertEqual(model_config_hash_payload(STUDENT), {**FF_PAYLOAD, "value_heads": 3})
        source = (ROOT / "configs/neural/gym-1.2.5/standard-standalone.toml").read_text()
        base = load_neural_config(ROOT / "configs/neural/gym-1.2.5/standard-standalone.toml")
        with tempfile.TemporaryDirectory() as folder:
            text = source.replace('manifest_path = "../../cases/neural/standard-standalone-train.json"',
                                  f'manifest_path = "{ROOT / "configs/cases/neural/standard-standalone-train.json"}"')
            text = text.replace('output_root = "../../../artifacts', f'output_root = "{ROOT}/artifacts')
            loaded = load_neural_config(_write(Path(folder), "one.toml", text.replace(
                "max_parameters = 10000000\n", "max_parameters = 10000000\nvalue_heads = 1\n")))
            # The authored identity still declares the historical hash, and every projection agrees.
            self.assertEqual(loaded.accepted_identities, base.accepted_identities)
            self.assertEqual(_stream_runtime_projection(loaded), _stream_runtime_projection(base))
            self.assertEqual(_config_projection(loaded)["model"], _config_projection(base)["model"])
            with self.assertRaisesRegex(ValueError, "PPO configs keep one critic"):
                load_neural_config(_write(Path(folder), "three.toml", text.replace(
                    "max_parameters = 10000000\n", "max_parameters = 10000000\nvalue_heads = 3\n")))
        for bad in (0, 2, True, 1.0):
            with self.subTest(value_heads=bad), self.assertRaises(ContractError):
                ModelConfig(16, 64, 10_000_000, value_heads=bad)


class DistillConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def load(self, text: str):
        return load_distill_config(_write(self.folder, "distill.toml", text))

    def test_distill_config_without_ppo(self):
        config = self.load(distill_toml())
        self.assertEqual(config.identity.objective_id, DISTILL_OBJECTIVE_ID)
        self.assertEqual(set(config.identity.config_hashes), {"model", "distill"})
        self.assertEqual(config.identity.config_hashes["distill"], _canonical_hash(
            {"tau": 0.05, "learning_rate": 0.0003, "batch_rows": 512, "gradient_steps": 18000, "data_fraction": 1.0}))
        self.assertEqual(config.model, STUDENT)
        for absent in ("ppo", "manifest_path", "chunks", "dataset_manifest"):
            self.assertFalse(hasattr(config, absent), absent)
        # Seed, eval_every, the shards and microbatching are run metadata, outside the hashes.
        for variant in (distill_toml(seed=7), distill_toml(eval_every=250), distill_toml(microbatch_rows=128)):
            self.assertEqual(self.load(variant).identity, config.identity)
        self.assertEqual(self.load(distill_toml(microbatch_rows=128)).distill.effective_microbatch_rows, 128)
        # Every hashed field changes the hash; an integer data_fraction normalizes to the float.
        for change in ({"tau": 0.1}, {"learning_rate": 0.001}, {"batch_rows": 256}, {"gradient_steps": 9},
                       {"data_fraction": 0.5}):
            with self.subTest(change=change):
                self.assertNotEqual(distill_config_hash(replace(HYPER, **change)), distill_config_hash(HYPER))
        self.assertEqual(distill_config_hash(replace(HYPER, data_fraction=1)), distill_config_hash(HYPER))
        refusals = {
            "[ppo]": distill_toml(extra="[ppo]\nlearning_rate = 0.0003\n"),
            "[rollout]": distill_toml(extra="[rollout]\nrows = 1\nsteps = 1\n"),
            "manifest_path": distill_toml(extra='manifest_path = "x.json"'),
            "one value head": distill_toml(model=ModelConfig(16, 64, 10_000_000),
                                           hashes={"model": FF_HASH, "distill": distill_config_hash(HYPER)}),
            "stale distill hash": distill_toml(hashes={"model": model_config_hash(STUDENT), "distill": "0" * 64}),
            "PPO hash key": distill_toml(hashes={"model": model_config_hash(STUDENT), "ppo": "0" * 64}),
            "shard paths in the config": distill_toml(extra='[data]\nchunks = ["a"]\n'),
            "fraction above one": distill_toml(hyper=replace(HYPER, data_fraction=0.5)).replace(
                "data_fraction = 0.5", "data_fraction = 1.5"),
            "microbatch above batch": distill_toml(microbatch_rows=1024),
        }
        for name, text in refusals.items():
            with self.subTest(refusal=name), self.assertRaises((ValueError, ContractError)):
                self.load(text)
        # load_neural_config admits the distillation objective only to evaluate.
        with self.assertRaisesRegex(ValueError, "only to evaluate"):
            load_neural_config(_write(self.folder, "train.toml", eval_toml(self.folder, operation="train",
                                                                             split="train")))

    def test_dataset_manifest_and_data_fraction_prefix(self):
        chunks = [write_chunk(self.folder, 1008, 2), write_chunk(self.folder, 1000, 3, rows=lambda index: index - 999)]
        manifest = build_dataset_manifest(chunks, relative_to=self.folder)
        self.assertEqual(manifest["schema"], "spire-learn-distill-dataset-v1")
        self.assertEqual([item["path"] for item in manifest["chunks"]], ["chunk-1008", "chunk-1000"])
        self.assertEqual(manifest["chunks"][0]["chunk_manifest_sha256"],
                         hashlib.sha256((chunks[0] / "chunk-manifest.json").read_bytes()).hexdigest())
        path = _write(self.folder, "dataset.json", json.dumps(manifest))
        resolved, digest = load_dataset_manifest(path)
        self.assertEqual(resolved, tuple(chunk.resolve() for chunk in chunks))
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        dataset = open_dataset(path)
        self.assertEqual(dataset.manifest_sha256, digest)
        self.assertEqual(dataset.identity()["dataset_manifest_sha256"], digest)
        self.assertEqual([ref.index for ref in dataset.train], [1000, 1001, 1002, 1008, 1009])
        self.assertEqual([ref.rows for ref in dataset.train], [1, 2, 3, 3, 3])
        self.assertEqual([ref.index for ref in dataset.validation], [1000, 1001, 1002, 1008, 1009])
        self.assertEqual([ref.index for ref in dataset.train_prefix(0.4)], [1000, 1001])
        self.assertEqual([ref.index for ref in dataset.train_prefix(1.0)], [1000, 1001, 1002, 1008, 1009])
        self.assertEqual(dataset.identity()["train_rows"], 12)
        # Absolute paths work too; a chunk listed twice or an unfinished chunk is refused.
        self.assertEqual(load_dataset_manifest(_write(self.folder, "absolute.json", json.dumps(
            build_dataset_manifest(chunks))))[0], resolved)
        with self.assertRaisesRegex(Exception, "twice"):
            build_dataset_manifest([chunks[0], chunks[0]])
        with self.assertRaisesRegex(Exception, "not a finished"):
            build_dataset_manifest([self.folder / "missing"])
        # A shard listed twice, or a shard directory that differs from its chunk manifest, is refused.
        with self.assertRaisesRegex(Exception, "two chunks"):
            discover_dataset([chunks[1], write_chunk(self.folder / "again", 1002, 1)])
        (chunks[0] / "train" / "shard-001008" / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(Exception, "differs from its chunk manifest"):
            discover_dataset(chunks)
        # A changed chunk manifest no longer matches the dataset manifest.
        chunk_manifest = chunks[1] / "chunk-manifest.json"
        chunk_manifest.write_text(chunk_manifest.read_text() + " ")
        with self.assertRaisesRegex(Exception, "differs from the dataset manifest"):
            load_dataset_manifest(path)
        for bad in ({"schema": "x", "chunks": manifest["chunks"]}, {"schema": manifest["schema"], "chunks": []},
                    {"schema": manifest["schema"], "chunks": ["chunk-1000"]}):
            with self.subTest(bad=bad), self.assertRaisesRegex(Exception, "dataset manifest is"):
                load_dataset_manifest(_write(self.folder, "bad.json", json.dumps(bad)))

    def test_timing_summary_names_each_part(self):
        timing = timing_summary(device="cuda:0", rows=1000, train_wall=10.0, train_load=4.0, train_model=5.0,
                                eval_wall=3.0, eval_load=1.0, eval_model=1.5, wall=20.0, gpu_wait_seconds=2.5)
        self.assertEqual(timing, {
            "train_wall_seconds": 10.0, "train_load_seconds": 4.0, "train_model_seconds": 5.0,
            "eval_wall_seconds": 3.0, "eval_load_seconds": 1.0, "eval_model_seconds": 1.5,
            "train_eval_wall_seconds": 13.0, "gpu_step_seconds": 6.5, "gpu_wait_seconds": 2.5,
            "wall_seconds": 20.0, "rows_per_second": 100.0, "model_rows_per_second": 200.0})
        cpu = timing_summary(device="cpu", rows=0, train_wall=0.0, train_load=0.0, train_model=0.0,
                             eval_wall=0.0, eval_load=0.0, eval_model=0.0, wall=1.0, gpu_wait_seconds=0)
        self.assertIsNone(cpu["gpu_step_seconds"])
        self.assertEqual((cpu["rows_per_second"], cpu["model_rows_per_second"]), (0.0, 0.0))
        self.assertNotIn("gpu_seconds", timing)

    def test_domain_seeds_are_distinct_and_stable(self):
        seeds = {domain: domain_seed(1, domain) for domain in ("policy", "minibatch", "curriculum", "controller")}
        self.assertEqual(len(set(seeds.values())), 4)
        self.assertEqual(seeds["minibatch"], domain_seed(1, "minibatch"))
        self.assertNotEqual(domain_seed(2, "minibatch"), seeds["minibatch"])
        self.assertTrue(all(0 <= value < 2**63 for value in seeds.values()))


@dataclass(frozen=True)
class FakeTransformerConfig:
    layers: int
    width: int
    heads: int
    value_heads: int = 3


class EvaluationConfigTests(unittest.TestCase):
    def test_eval_config_projection_distill_and_transformer(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = load_neural_config(_write(folder, "eval.toml", eval_toml(folder)))
            self.assertIsNone(config.ppo)
            self.assertEqual(config.model, STUDENT)
            self.assertEqual(dict(config.accepted_identities[0].config_hashes), {"model": model_config_hash(STUDENT)})
            projection = _config_projection(config)
            self.assertNotIn("ppo", projection)
            self.assertNotIn("distill", projection)
            self.assertEqual(projection["model"], {**FF_PAYLOAD, "value_heads": 3})
            self.assertEqual(list(projection)[list(projection).index("model") + 1], "rollout")
            self.assertNotIn("ppo", _stream_runtime_projection(config))
            # A distill hyperparameter record is projected only when the config carries one.
            self.assertEqual(_config_projection(replace(config, distill=HYPER))["distill"],
                             {"tau": 0.05, "learning_rate": 0.0003, "batch_rows": 512, "gradient_steps": 18000,
                              "data_fraction": 1.0})
            for name, text in {
                "PPO hashes": eval_toml(folder, hashes={"model": model_config_hash(STUDENT), "ppo": "0" * 64}),
                "distill hash": eval_toml(folder, hashes={"model": model_config_hash(STUDENT), "distill": "0" * 64}),
                "a [ppo] table": eval_toml(folder, ppo=True),
                "a stale model hash": eval_toml(folder, hashes={"model": FF_HASH}),
            }.items():
                with self.subTest(refusal=name), self.assertRaises(ValueError):
                    load_neural_config(_write(folder, "bad.toml", text))
            # A PPO eval config keeps [ppo]; dropping it under the win/loss objective is refused.
            with self.assertRaises(ValueError):
                load_neural_config(_write(folder, "ppo.toml", eval_toml(
                    folder, objective="spire-learn-combat-winloss-v1")))

            # The seam itself, with a stand-in parser (the real [transformer] table: test_transformer_config).
            transformer_table = "[transformer]\nlayers = 2\nwidth = 128\nheads = 4\n\n"
            text = eval_toml(folder, model_table=transformer_table, hashes={"model": model_config_hash(
                FakeTransformerConfig(2, 128, 4))}, learner=config_module.TRANSFORMER_LEARNER_IDENTITY_IDS)
            with patch.dict(config_module.MODEL_FAMILY_TABLES, {"model": config_module._parse_ff_model}, clear=True), \
                    self.assertRaisesRegex(ValueError, "no registered model family"):
                load_neural_config(_write(folder, "transformer.toml", text))
            with patch.dict(config_module.MODEL_FAMILY_TABLES,
                            {"transformer": lambda table: FakeTransformerConfig(**table)}):
                loaded = load_neural_config(_write(folder, "transformer.toml", text))
                both = eval_toml(folder, model_table="[model]\nembedding_width = 16\n\n" + transformer_table)
                with self.assertRaisesRegex(ValueError, "exactly one model table"):
                    load_neural_config(_write(folder, "both.toml", both))
            self.assertEqual(loaded.model_table, "transformer")
            self.assertEqual(loaded.model, FakeTransformerConfig(2, 128, 4))
            projection = _config_projection(loaded)
            self.assertNotIn("model", projection)
            self.assertNotIn("ppo", projection)
            self.assertEqual(projection["transformer"], asdict(FakeTransformerConfig(2, 128, 4)))
            self.assertEqual(list(projection)[list(projection).index("transformer") + 1], "rollout")


def _distill_identity(model_hash="a" * 64, distill_hash="b" * 64, **extra) -> NeuralIdentity:
    model_id, features_id, action_id = STANDARD_LEARNER_IDENTITY_IDS
    return NeuralIdentity(model_id=model_id, features_id=features_id, action_id=action_id,
                          objective_id=DISTILL_OBJECTIVE_ID,
                          accepted_raw_mode_mappings={"standard-standalone": MAPPING},
                          config_hashes={"model": model_hash, "distill": distill_hash, **extra})


def _checkpoint_metadata(identity: NeuralIdentity, *, step=4, hyperparameter_id="b" * 64, **state) -> dict:
    return {
        "schema_id": "spire-learn-neural-checkpoint-v1",
        "identity": identity_payload(identity),
        "payload_inventory": {"model.pt": {"size_bytes": 128, "sha256": "c" * 64},
                              "training.pt": {"size_bytes": 256, "sha256": "d" * 64}},
        "model_tensors": [{"name": "layer.weight", "dtype": "float32", "shape": [2, 3], "sha256": "e" * 64}],
        "training_state": {
            "optimizer_step": step, "hyperparameter_id": hyperparameter_id,
            "rng_domains": ["policy", "minibatch", "curriculum", "controller"],
            "counters": {name: step if name == "optimizer_steps" else 10 for name in DISTILL_COUNTER_FIELDS},
            "update_id": f"step-{step}", "behavior_policy_version": step,
            "next_environment_seed_ordinal": 0, "next_decision_sequence": 0, "saved_inflight_record_count": 0,
            **state,
        },
        "run_metadata": {"operation": "distill-train"},
        "parent_checkpoint": None,
    }


class DistillCheckpointTests(unittest.TestCase):
    def test_distill_checkpoint_refused_as_ppo(self):
        distill = _distill_identity()
        metadata = _checkpoint_metadata(distill)
        # The distillation identity accepts its own checkpoint, resuming against the distill hash.
        for purpose in ("evaluate", "warm_start", "resume"):
            validate_checkpoint_metadata(metadata, distill, purpose=purpose)
        # An evaluation identity declares only the model hash.
        evaluation = replace(distill, config_hashes={"model": "a" * 64})
        validate_checkpoint_metadata(metadata, evaluation, purpose="evaluate")
        # A PPO identity (same learner, same model hash) never loads it, for any purpose.
        ppo = replace(distill, objective_id="spire-learn-combat-winloss-v1",
                      config_hashes={"model": "a" * 64, "ppo": "b" * 64})
        for purpose in ("evaluate", "warm_start", "resume"):
            with self.subTest(purpose=purpose), self.assertRaises(CheckpointCompatibilityError):
                validate_checkpoint_metadata(metadata, ppo, purpose=purpose)
        # Nor does a PPO checkpoint load as a distillation one.
        ppo_metadata = _checkpoint_metadata(ppo)
        ppo_metadata["training_state"]["counters"] = {"environment_actions": 1}
        for purpose in ("evaluate", "warm_start", "resume"):
            with self.subTest(ppo_purpose=purpose), self.assertRaises(CheckpointCompatibilityError):
                validate_checkpoint_metadata(ppo_metadata, distill, purpose=purpose)
        # Resume checks the distill hash, not a PPO one.
        with self.assertRaisesRegex(CheckpointCompatibilityError, "hyperparameter"):
            validate_checkpoint_metadata(_checkpoint_metadata(distill, hyperparameter_id="f" * 64), distill,
                                         purpose="resume")
        # Mixed hash keys and environment progress are malformed.
        with self.assertRaisesRegex(ValueError, "PPO config hash"):
            validate_checkpoint_metadata(_checkpoint_metadata(_distill_identity(ppo="b" * 64)), distill,
                                         purpose="evaluate")
        with self.assertRaisesRegex(ValueError, "distill config hash"):
            validate_checkpoint_metadata(_checkpoint_metadata(replace(ppo, config_hashes={
                "model": "a" * 64, "ppo": "b" * 64, "distill": "b" * 64})), ppo, purpose="evaluate")
        for state in ({"next_environment_seed_ordinal": 3}, {"next_decision_sequence": 3},
                      {"counters": {"optimizer_steps": 4}}, {"optimizer_step": 5}):
            with self.subTest(state=state), self.assertRaises(ValueError):
                validate_checkpoint_metadata(_checkpoint_metadata(distill, **state), distill, purpose="evaluate")


class CheckpointChoiceTests(unittest.TestCase):
    def test_run_level_se_formula(self):
        differences = [0.2, 0.0, 0.1, 0.3, 0.0]
        runs = [1, 1, 2, 3, 3]
        d_bar, se = run_level_se(differences, runs)
        sums, counts = {1: 0.2, 2: 0.1, 3: 0.3}, {1: 2, 2: 1, 3: 2}
        expected_bar = 0.6 / 5
        spread = sum((sums[run] - counts[run] * expected_bar) ** 2 for run in sums)
        self.assertAlmostEqual(d_bar, expected_bar)
        self.assertAlmostEqual(se, (3 / 2 * spread) ** 0.5 / 5)
        self.assertEqual(run_level_se([0.5, 0.1], [9, 9]), (0.3, 0.0))
        self.assertEqual(run_level_se([], []), (0.0, 0.0))

    def test_best_regret_checkpoint_with_value_tiebreak(self):
        runs = [1, 1, 2, 2, 3, 3, 4, 4]
        best = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]
        # Slightly worse everywhere but noisy across runs: within one SE of the best.
        near = [0.4, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1]
        # Clearly worse in every run: outside one SE.
        far = [value + 0.5 for value in best]
        mean = lambda values: sum(values) / len(values)  # noqa: E731
        records = [
            EvalRecord(1000, mean(far), 0.01, far, runs),
            EvalRecord(2000, mean(best), 0.30, best, runs),
            EvalRecord(3000, mean(near), 0.10, near, runs),
        ]
        d_bar, se = run_level_se([a - b for a, b in zip(near, best)], runs)
        self.assertLessEqual(d_bar, se)
        chosen, tied = choose_checkpoint(records)
        # The far checkpoint has the lowest value loss but is not tied; the near one wins the tie.
        self.assertEqual(tied, [2000, 3000])
        self.assertEqual(chosen.step, 3000)
        # Without a near tie the lowest regret wins regardless of value loss.
        chosen, tied = choose_checkpoint([records[0], records[1]])
        self.assertEqual((chosen.step, tied), (2000, [2000]))
        # Equal keys go to the earlier step.
        same = [EvalRecord(5000, mean(best), 0.30, best, runs), records[1]]
        self.assertEqual(choose_checkpoint(same)[0].step, 2000)
        with self.assertRaises(ValueError):
            choose_checkpoint([records[1], EvalRecord(4000, 0.0, 0.0, best[:4], runs[:4])])


if __name__ == "__main__":
    unittest.main()
