"""The card-metadata entity transformer (S3) in torch: features, the S2 equivalence, checkpoints, trainer, runtime.

Needs numpy and torch (runtime container only) and skips cleanly on the host. Run it in the container with
``gated_tests_entry.py --transformer-cardmeta`` (``SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1``). The tests join the
synthetic metadata fixture (``tests/fixtures/neural/card-metadata-synthetic.json``, the real schema shape, the
pinned fingerprint) by patching the snapshot path; the real snapshot is checked in
``tests/integration/test_card_metadata_live.py``.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from spire_learn.neural.config import (
    DISTILL_OBJECTIVE_ID, STANDARD_LEARNER_IDENTITY_IDS, TRANSFORMER_CARDMETA_FEATURES_ID,
    TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS, TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_FEATURES_ID,
    TRANSFORMER_MODEL_ID, TRANSFORMER_PARAMETER_CAP, DistillConfig,
    TransformerCardMetaConfig, TransformerConfig, distill_config_hash, model_config_hash, transformer_cardmeta_size,
)
from spire_learn.neural.contracts import NeuralIdentity
from tests.neural.test_transformer import (
    FF_TINY, HYPER, MAPPING, TINY, _load_modules, ff_identity, permute_arrays, rich_arrays, s2_identity,
    save_student, to_frame,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/neural/card-metadata-synthetic.json"
TINY3 = TransformerCardMetaConfig(layers=1, width=16, heads=2, number_encoding="bucket")
TINY3_TABLE = ('[transformer_cardmeta]\nlayers = 1\nwidth = 16\nheads = 2\nnumber_encoding = "bucket"\n'
               'value_heads = 3\ncard_metadata_fingerprint = '
               '"effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406"\n\n')
CARD_GROUPS = ("combat_cards", "permanent_cards", "combat_card_offers")
# Rows of the rich frame (tests/neural/test_transformer.py) that are occupied in each card group.
CARD_ROWS = {"permanent_cards": [0, 3, 7], "combat_cards": [2, 5, 9], "combat_card_offers": [0, 2]}
# Cards (id, upgrades) put on those rows: every metadata shape (int, x, unplayable, unresolved, no faces).
CARDS = {"permanent_cards": [("strike", 0), ("searing_blow", 1), ("defend", 2)],
         "combat_cards": [("whirlwind", 0), ("apparition", 1), ("wound", 3)],
         "combat_card_offers": [("bash", 0), ("wish", 0)]}


def s3_identity(config: TransformerCardMetaConfig = TINY3, **changes) -> NeuralIdentity:
    model_id, features_id, action_id = TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS
    fields = dict(model_id=model_id, features_id=features_id, action_id=action_id, objective_id=DISTILL_OBJECTIVE_ID,
                  accepted_raw_mode_mappings={"standard-standalone": MAPPING},
                  config_hashes={"model": model_config_hash(config), "distill": distill_config_hash(HYPER)})
    fields.update(changes)
    return NeuralIdentity(**fields)


class CardMetaTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np, cls.torch, cls.shards = _load_modules()
        cls.torch.set_num_threads(1)
        from spire_learn.neural import transformer

        cls.transformer = transformer
        # Every build below joins the synthetic fixture instead of the committed snapshot.
        cls.snapshot = mock.patch.object(transformer, "CARD_METADATA_SNAPSHOT", FIXTURE)
        cls.snapshot.start()
        cls.schema = cls.shards._schema()
        cls.plan = transformer.transformer_plan(cls.schema)
        cls.vocabulary = cls.schema.vocabularies["card"]
        cls.metadata = json.loads(FIXTURE.read_text())
        cls.tables = transformer.card_metadata_tables(cls.metadata, cls.vocabulary)
        cls.base_arrays = rich_arrays(cls.schema, cls.shards, cls.np)
        cls.arrays = cls.with_cards(cls.base_arrays, CARDS)
        cls.frame = to_frame(cls.arrays, cls.schema)
        cls.plain = to_frame(cls.base_arrays, cls.schema)
        cls.frames = [cls.frame, *cls.shards.frames(cls.schema)]
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.snapshot.stop()
        cls.temp.cleanup()

    @classmethod
    def with_cards(cls, arrays: dict, cards: dict) -> dict:
        """A copy of ``arrays`` with the named cards (id, upgrades) on the card rows of each group."""
        out = {key: value.copy() for key, value in arrays.items()}
        for group, edits in cards.items():
            spec = cls.schema.tensors[group]
            card_index = next(i for i, c in enumerate(spec.columns) if c.vocabulary == "card")
            upgrades = next((i for i, c in enumerate(spec.columns) if c.field == "upgrades"), None)
            for row, (card, level) in zip(CARD_ROWS[group], edits):
                out[group][row, card_index] = cls.vocabulary.index(card)
                out[spec.known_key][row, card_index] = 1
                if upgrades is not None:
                    out[group][row, upgrades] = level
                    out[spec.known_key][row, upgrades] = 1
        return out

    def build(self, config=TINY3, *, seed: int = 7):
        from spire_learn.neural.model_families import build_family_policy

        policy = build_family_policy(TRANSFORMER_CARDMETA_MODEL_ID, self.schema, config, seed=seed, device="cpu",
                                     objective_id=DISTILL_OBJECTIVE_ID, model_id=TRANSFORMER_CARDMETA_MODEL_ID,
                                     features_id=TRANSFORMER_CARDMETA_FEATURES_ID)
        policy.eval()
        return policy

    def build_s2(self, config: TransformerConfig = TINY, *, seed: int = 7):
        from spire_learn.neural.model_families import build_family_policy

        policy = build_family_policy(TRANSFORMER_MODEL_ID, self.schema, config, seed=seed, device="cpu",
                                     objective_id=DISTILL_OBJECTIVE_ID, model_id=TRANSFORMER_MODEL_ID,
                                     features_id=TRANSFORMER_FEATURES_ID)
        policy.eval()
        return policy

    def encode(self, frames, *, trim: bool = False):
        from spire_learn.neural.features import encode_frames

        return encode_frames(list(frames), self.schema, device="cpu", trim_padding=trim)

    def run_policy(self, policy, batch):
        with self.torch.no_grad():
            return policy(batch)

    def assert_same(self, left, right, *, exact: bool = True):
        torch = self.torch
        for name in ("logits", "values", "aux_values"):
            a, b = getattr(left, name), getattr(right, name)
            if exact:
                self.assertTrue(torch.equal(a, b), name)
            else:
                self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4), (name, (a - b).abs().max().item()))

    def zeroed(self, policy, groups):
        """Context manager: the metadata projections of ``groups`` are zero (weight and bias), then restored."""
        import contextlib

        @contextlib.contextmanager
        def manager():
            saved = {name: {key: tensor.clone() for key, tensor in policy.card_projections[name].state_dict().items()}
                     for name in groups}
            try:
                with self.torch.no_grad():
                    for name in groups:
                        for tensor in policy.card_projections[name].parameters():
                            tensor.zero_()
                yield
            finally:
                for name in groups:
                    policy.card_projections[name].load_state_dict(saved[name])

        return manager()

    # --- build -----------------------------------------------------------------------------

    def test_parameter_counts(self):
        from spire_learn.neural import model_families as families
        from spire_learn.contracts import ContractError

        transformer = self.transformer
        for size, encoding in (("small", "bucket"), ("medium", "scaled"), ("large", "bucket")):
            config = transformer_cardmeta_size(size, encoding)
            with self.subTest(size=size, encoding=encoding):
                policy = self.build(config)
                actual = sum(int(parameter.numel()) for parameter in policy.parameters())
                self.assertEqual(actual, transformer.transformer_cardmeta_parameter_count(self.plan, config))
                self.assertLessEqual(actual, TRANSFORMER_PARAMETER_CAP)
                s2 = transformer.transformer_parameter_count(self.plan, config.transformer())
                self.assertEqual(actual - s2, 3 * (transformer.CARD_METADATA_FEATURES * config.width + config.width))
                # Parameters only: the metadata tables and the sinusoid frequencies are not state.
                self.assertEqual(set(policy.state_dict()), {name for name, _ in policy.named_parameters()})
                self.assertEqual({name for name, _ in policy.named_buffers()} - {"_frequencies"},
                                 {"_card_static", "_card_faces"})
                del policy
        capped = replace(families.TRANSFORMER_CARDMETA_FAMILY, parameter_cap=1000)
        with mock.patch.dict(families._FAMILIES, {TRANSFORMER_CARDMETA_MODEL_ID: capped}), \
                self.assertRaisesRegex(ContractError, "over its cap"):
            self.build(TINY3)

    # --- the features ----------------------------------------------------------------------

    def test_card_features_follow_the_tables(self):
        torch, transformer = self.torch, self.transformer
        policy = self.build()
        groups = {item.group: item for item in transformer.card_metadata_groups(self.schema)}
        index = self.vocabulary.index
        # (card, upgrades, card known, upgrades known) per row, then the level the face choice must use.
        cases = [("strike", 0, True, True, 0), ("strike", 1, True, True, 1), ("strike", 2, True, True, 1),
                 ("strike", 9, True, True, 1), ("strike", -1, True, True, 0), ("strike", 1, True, False, 0),
                 ("strike", 1, False, True, None), ("wound", 3, True, True, 1), ("wish", 1, True, True, 1),
                 ("padding", 0, True, True, 1), ("burn", 1, True, True, 1), ("apparition", 1, True, True, 1)]
        for name, item in groups.items():
            with self.subTest(group=name):
                fields = len(self.schema.tensors[name].columns)
                values = torch.zeros((len(cases), fields), dtype=torch.int64)
                known = torch.ones((len(cases), fields), dtype=torch.bool)
                expected = []
                for row, (card, upgrades, card_known, upgrades_known, level) in enumerate(cases):
                    values[row, item.card_field_index] = index(card)
                    known[row, item.card_field_index] = card_known
                    if item.upgrades_field_index is not None:
                        values[row, item.upgrades_field_index] = upgrades
                        known[row, item.upgrades_field_index] = upgrades_known
                    elif card_known:
                        level = 0  # offers have no upgrades: level 0
                    if level is None:
                        expected.append([0.0] * transformer.CARD_METADATA_FEATURES)
                        continue
                    code = index(card)
                    expected.append([*self.tables.static[code], *self.tables.faces[level][code]])
                actual = policy._card_features(item, values, known)
                self.assertEqual(tuple(actual.shape), (len(cases), transformer.CARD_METADATA_FEATURES))
                self.assertTrue(torch.equal(actual, torch.tensor(expected, dtype=torch.float32)))
        self.assertEqual(sorted(groups), sorted(CARD_GROUPS))
        self.assertIsNone(groups["combat_card_offers"].upgrades_field_index)

    def test_s3_is_s2_plus_the_metadata_term(self):
        torch = self.torch
        s2, s3 = self.build_s2(TINY), self.build(TINY3)
        result = s3.load_state_dict(dict(s2.state_dict()), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(sorted(result.missing_keys), sorted(
            f"card_projections.{group}.{kind}" for group in CARD_GROUPS for kind in ("weight", "bias")))
        batch = self.encode(self.frames)
        reference = self.run_policy(s2, batch)
        with self.zeroed(s3, CARD_GROUPS):
            self.assert_same(self.run_policy(s3, batch), reference)
        # With its projections S3 differs on a frame that carries cards with metadata...
        full = self.run_policy(s3, batch)
        self.assertFalse(torch.equal(full.values[:1], reference.values[:1]))
        # ... through every group on its own.
        for group in CARD_GROUPS:
            with self.subTest(group=group):
                with self.zeroed(s3, [group]):
                    only = self.run_policy(s3, batch)
                self.assertFalse(torch.equal(only.values[:1], full.values[:1]))
                with self.zeroed(s3, [name for name in CARD_GROUPS if name != group]):
                    alone = self.run_policy(s3, batch)
                self.assertFalse(torch.equal(alone.values[:1], reference.values[:1]))

    def test_every_parameter_reaches_the_outputs(self):
        torch = self.torch
        for encoding in ("bucket", "scaled", "sinusoidal"):
            with self.subTest(encoding=encoding):
                policy = self.build(replace(TINY3, number_encoding=encoding))
                policy.train()
                batch = self.encode(self.frames)
                output = policy(batch)
                self.assertTrue(torch.isfinite(output.logits).all().item())
                (output.logits.masked_select(batch.action_mask).sum() + output.values.sum()
                 + output.aux_values.sum()).backward()
                self.assertEqual([name for name, p in policy.named_parameters() if p.grad is None], [])
                for group in CARD_GROUPS:
                    self.assertGreater(float(policy.card_projections[group].weight.grad.abs().sum()), 0.0, group)

    def test_trimmed_and_permuted_card_rows_agree(self):
        np = self.np
        policy = self.build(replace(TINY3, number_encoding="scaled"))
        trimmed, full = self.encode([self.frame], trim=True), self.encode([self.frame])
        self.assert_same(self.run_policy(policy, trimmed), self.run_policy(policy, full))
        # Moving rows of the card groups (references remapped) moves their metadata with them.
        rng = np.random.default_rng(3)
        permutations = {name: rng.permutation(self.arrays[name].shape[0]) for name in CARD_GROUPS}
        moved = to_frame(permute_arrays(self.arrays, self.schema, permutations, np), self.schema)
        self.assert_same(self.run_policy(policy, self.encode([moved], trim=True)), self.run_policy(policy, full),
                         exact=False)

    def test_tables_are_static_and_not_state(self):
        torch = self.torch
        policy = self.build()
        before = (policy._card_static.clone(), policy._card_faces.clone())
        self.run_policy(policy, self.encode(self.frames))
        self.assertTrue(torch.equal(policy._card_static, before[0]) and torch.equal(policy._card_faces, before[1]))
        self.assertEqual(tuple(policy._card_static.shape), (len(self.vocabulary), 25))
        self.assertEqual(tuple(policy._card_faces.shape), (2, len(self.vocabulary), 19))
        # The tables are the stdlib tables: padding and missing ids are zero rows.
        self.assertTrue(torch.equal(policy._card_static, torch.tensor(self.tables.static, dtype=torch.float32)))
        self.assertEqual(float(policy._card_static[self.vocabulary.index("padding")].abs().sum()), 0.0)

    # --- checkpoints -----------------------------------------------------------------------

    def test_s2_s3_and_ff_checkpoints_never_cross(self):
        from spire_learn.neural.checkpoint import CheckpointCompatibilityError, load_checkpoint
        from spire_learn.neural.model import build_policy

        torch = self.torch
        root = self.root / "refusals"
        root.mkdir()
        s2, s3 = self.build_s2(TINY), self.build(TINY3, seed=3)
        s2_path = save_student(root / "s2", s2, s2_identity())
        s3_path = save_student(root / "s3", s3, s3_identity())
        ff = build_policy(self.schema, FF_TINY, seed=3, device="cpu", objective_id=DISTILL_OBJECTIVE_ID,
                          model_id=STANDARD_LEARNER_IDENTITY_IDS[0], features_id=STANDARD_LEARNER_IDENTITY_IDS[1])
        ff_path = save_student(root / "ff", ff, ff_identity())
        # S3 loads under its own identity, and its tensors are the saved ones.
        loaded = load_checkpoint(s3_path, s3_identity(), purpose="evaluate", device="cpu")
        fresh = self.build(TINY3, seed=99)
        fresh.load_state_dict(dict(loaded.model_tensors), strict=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), s3.state_dict().values())))
        # S3 weights under S2's or FF's identity, S2 or FF weights under S3's: never, in any purpose.
        crossings = (
            ("S3 as S2", s3_path, s2_identity()), ("S3 as FF", s3_path, ff_identity()),
            ("S2 as S3", s2_path, s3_identity()), ("FF as S3", ff_path, s3_identity()),
            # S3 weights under a mixed identity: S2's features id or S2's model id.
            ("S3 with S2 features", save_student(root / "mixed-features", s3, s3_identity(
                features_id=TRANSFORMER_FEATURES_ID)), s3_identity()),
            ("S3 with S2 model", save_student(root / "mixed-model", s3, s3_identity(
                model_id=TRANSFORMER_MODEL_ID)), s3_identity()),
            ("S3 with another config", s3_path, s3_identity(replace(TINY3, number_encoding="scaled"))),
        )
        for name, path, identity in crossings:
            for purpose in ("evaluate", "warm_start", "resume"):
                with self.subTest(crossing=name, purpose=purpose), self.assertRaises(CheckpointCompatibilityError):
                    load_checkpoint(path, identity, purpose=purpose, device="cpu")
        # The tensors could not cross either.
        with self.assertRaises(RuntimeError):
            s2.load_state_dict(dict(s3.state_dict()), strict=True)
        with self.assertRaises(RuntimeError):
            fresh.load_state_dict(dict(s2.state_dict()), strict=True)

    # --- the trainer and the runtime -------------------------------------------------------

    def test_distill_trainer_trains_the_card_metadata_transformer(self):
        from spire_learn.neural import distill_data, distill_train
        from spire_learn.neural.checkpoint import load_checkpoint

        np = self.np
        frames = [self.frame, self.frames[1], self.frames[3], to_frame(permute_arrays(
            self.arrays, self.schema, {"combat_cards": np.random.default_rng(5).permutation(1024)}, np), self.schema)]

        def write(series, index, order, run):
            path = self.root / f"trainer-{series}-{index}"
            writer = distill_data.ShardWriter(path, self.schema, {"fixture": series})
            for position in order:
                frame = frames[position]
                legal = [int(a) for a in np.flatnonzero(np.frombuffer(frame.tensors["action_mask"].data, np.int8))]
                values = [0.1 * ((action * 7 + position) % 5) - 0.2 for action in legal]
                if len(set(values)) == 1:
                    values[-1] += 0.5
                writer.add(frame, distill_data.DecisionLabels(
                    legal=legal, values=values, samples=32, choice=legal[int(np.argmax(values))], played=legal[0],
                    root="searched", outcome=("win", "death", "escaped")[position % 3], hp_kept=0.1 * position,
                    realised_v2=0.2 - 0.05 * position, value_ok=True, policy_ok=True, research_seed=position,
                    case=(1000, position), run_id=f"{run}-{position % 2}", turn=position))
            manifest = writer.close()
            return distill_train.ShardRef(series, index, path, manifest["rows"], "f" * 64)

        dataset = distill_train.Dataset(chunks=(), chunk_manifest_sha256={},
                                        train=(write("train", 1000, [0, 1, 2, 3], "run-a"),),
                                        validation=(write("validation", 1000, [3, 0, 2], "run-v"),))
        config = DistillConfig(
            source=Path("fixture.toml"), config_sha256="0" * 64, mode="standard-standalone",
            environment_revision="fixture", standard_view_layout="compact",
            standard_view_preset="standard-combat-compact-v1", standard_view_fingerprint="fixture",
            output_root=Path("."), device="cpu", runtime_gpu_uuid=None, seed=3, eval_every=2,
            model=TINY3, distill=HYPER, identity=s3_identity(), model_table="transformer_cardmeta")
        output = self.root / "trainer-run"
        summary = distill_train.train_student(config, self.schema, dataset, output, log_every=1)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["parameter_count"],
                         self.transformer.transformer_cardmeta_parameter_count(self.plan, TINY3))
        self.assertEqual([item["step"] for item in summary["evaluations"]], [2, 3])
        loaded = load_checkpoint(output / "checkpoint-final", config.identity, purpose="resume", device="cpu")
        student = distill_train.build_student(config, self.schema, device="cpu")
        self.assertEqual((student.model_id, student.features_id),
                         (TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_CARDMETA_FEATURES_ID))
        student.load_state_dict(dict(loaded.model_tensors), strict=True)
        initial = distill_train.build_student(config, self.schema, device="cpu")
        self.assertFalse(all(self.torch.equal(a, b) for a, b in zip(initial.state_dict().values(),
                                                                    student.state_dict().values())))

    def test_card_metadata_transformer_through_create_runtime_evaluate(self):
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.features import encode_frames
        from spire_learn.neural.model_families import reflex_argmax
        from spire_learn.neural.runner import _DeterministicPolicy, create_runtime
        from tests.unit.test_distill_config import _write, eval_toml

        torch = self.torch
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            text = eval_toml(folder, model_table=TINY3_TABLE, hashes={"model": model_config_hash(TINY3)},
                             learner=TRANSFORMER_CARDMETA_LEARNER_IDENTITY_IDS)
            config = load_neural_config(_write(folder, "eval.toml", text))
            self.assertEqual(config.model_table, "transformer_cardmeta")
            identity = config.accepted_identities[0]
            saved = self.build(seed=11)
            checkpoint = save_student(folder / "checkpoint", saved, s3_identity(
                accepted_raw_mode_mappings=dict(identity.accepted_raw_mode_mappings)), step=3)
            runtime = create_runtime(config, purpose="evaluate", checkpoint=checkpoint, policy_kind="checkpoint",
                                     action_selection="deterministic")
            try:
                policy = runtime.policy
                self.assertEqual((policy.model_id, policy.features_id, policy.objective_id),
                                 (TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_CARDMETA_FEATURES_ID,
                                  DISTILL_OBJECTIVE_ID))
                self.assertIsInstance(runtime.collector.policy, _DeterministicPolicy)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(policy.state_dict().values(),
                                                                      saved.state_dict().values())))
                rollout = runtime.collector.collect(max_learned_decisions=config.budgets.max_training_decisions,
                                                    max_env_actions=config.budgets.max_environment_actions,
                                                    policy_version=policy._behavior_policy_version)
                self.assertGreaterEqual(len(rollout.records), 1)
                record = rollout.records[0]
                schema = runtime.collector.driver.schema
                with torch.no_grad():
                    batch = encode_frames([record.source], schema, device="cpu", trim_padding=True)
                    output = policy(batch)
                self.assertEqual(record.choice.original_action_index,
                                 int(reflex_argmax(output.logits, batch.action_mask)[0]))
                self.assertAlmostEqual(record.choice.value, float(output.values[0]), places=5)
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
