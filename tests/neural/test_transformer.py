"""The entity transformer (S2) in torch: build and caps, tokens, references, scoring, checkpoints, runtime.

Needs numpy and torch (runtime container only) and skips cleanly on the host. Run it in the container
with ``gated_tests_entry.py --transformer`` (``SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1``); the create_runtime
test plays one real decision on the pinned Gym. Frames are built on the real compact fixture schema,
with every row group occupied and every non-action reference known.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from spire_learn.neural.config import (
    DISTILL_OBJECTIVE_ID, NUMBER_ENCODINGS, STANDARD_LEARNER_IDENTITY_IDS, TRANSFORMER_FEATURES_ID,
    TRANSFORMER_LEARNER_IDENTITY_IDS, TRANSFORMER_MODEL_ID, TRANSFORMER_PARAMETER_CAP, TRANSFORMER_SIZES,
    DistillConfig, DistillHyperparameters, TransformerConfig, distill_config_hash, model_config_hash,
    transformer_size,
)
from spire_learn.neural.contracts import ModelConfig, NeuralIdentity

TINY = TransformerConfig(layers=1, width=16, heads=2, number_encoding="bucket")
TINY_TABLE = ('[transformer]\nlayers = 1\nwidth = 16\nheads = 2\nnumber_encoding = "bucket"\n'
              "value_heads = 3\n\n")
FF_TINY = ModelConfig(embedding_width=4, hidden_width=8, max_parameters=10_000_000, value_heads=3)
HYPER = DistillHyperparameters(tau=0.1, learning_rate=1e-3, batch_rows=2, gradient_steps=3, data_fraction=1.0)
MAPPING = "spire-learn-standard-combat-compact-v1@fixture+test"
RNG_DOMAINS = ["policy", "minibatch", "curriculum", "controller"]
# Occupied rows of every row group of the rich frame (potions/orbs also publish known empty slots).
ROWS = {"permanent_cards": [0, 3, 7], "combat_cards": [2, 5, 9], "enemies": [1, 4], "powers": [0, 6, 10],
        "relics": [2, 8], "potions": [1, 3], "selection": [1, 3], "orbs": [0, 2], "known_draw": [0, 2],
        "stance_choices": [0, 1], "combat_card_offers": [0, 2], "actions": [0, 2, 5, 8, 11]}
EMPTY_KNOWN = {"potions": [0], "orbs": [1]}
# Known references: (group, row, field, target row, selector group or None).
REFERENCES = [
    ("combat_cards", 2, "permanent_row", 3, None), ("combat_cards", 5, "permanent_row", 7, None),
    ("combat_cards", 5, "stasis_enemy_row", 4, None),
    ("powers", 0, "owner_row", 0, "player"), ("powers", 6, "owner_row", 1, "enemies"),
    ("powers", 10, "owner_row", 4, "enemies"),
    ("known_draw", 0, "combat_card_row", 9, None), ("known_draw", 2, "combat_card_row", 2, None),
    ("selection", 1, "source_row", 5, "combat_cards"), ("selection", 3, "source_row", 0, "permanent_cards"),
    ("actions", 0, "source_row", 2, "combat_cards"), ("actions", 0, "target_row", 4, "enemies"),
    ("actions", 2, "source_row", 3, "potions"), ("actions", 2, "target_row", 0, "player"),
    ("actions", 5, "stance_choice_ref", 1, None), ("actions", 8, "source_row", 3, "selection"),
]


def _load_modules():
    try:
        import numpy
        import torch
    except ImportError:
        raise unittest.SkipTest("numpy and torch are only importable in the runtime container")
    from tests.neural import test_distill_data as shards

    shards._load_numpy_modules()
    return numpy, torch, shards


def _field(spec, name: str) -> int:
    return next(index for index, column in enumerate(spec.columns) if column.field == name)


def frame_arrays(frame, np) -> dict:
    return {key: np.frombuffer(buffer.data, dtype=np.dtype(buffer.dtype)).reshape(buffer.shape).copy()
            for key, buffer in frame.tensors.items()}


def to_frame(arrays: dict, schema):
    from spire_learn.neural.contracts import BufferArray, CombatFrame
    from spire_learn.neural.schema import validate_frame

    frame = CombatFrame({key: BufferArray(spec.dtype, spec.shape, arrays[key].tobytes())
                         for key, spec in schema.tensors.items()}, schema.schema_fingerprint)
    validate_frame(frame, schema)
    return frame


def set_reference(arrays: dict, schema, group: str, row: int, field: str, target: int, selector: str | None) -> None:
    spec = schema.tensors[group]
    index = _field(spec, field)
    if selector is not None:
        selector_index = _field(spec, spec.columns[index].reference)
        vocabulary = schema.vocabularies[spec.columns[selector_index].vocabulary]
        arrays[group][row, selector_index] = vocabulary.index(selector)
        arrays[spec.known_key][row, selector_index] = 1
    arrays[group][row, index] = target
    arrays[spec.known_key][row, index] = 1


def rich_arrays(schema, shards, np, *, seed: int = 21) -> dict:
    arrays = frame_arrays(shards.build_frame(schema, seed, ROWS, EMPTY_KNOWN), np)
    for group, row, field, target, selector in REFERENCES:
        set_reference(arrays, schema, group, row, field, target, selector)
    return arrays


def reference_target(schema, group: str, values, known, row, column_index: int) -> str | None:
    """The group a known reference cell names (literal or through its selector)."""
    spec = schema.tensors[group]
    column = spec.columns[column_index]
    selector = next((index for index, item in enumerate(spec.columns) if item.field == column.reference), None)
    if selector is None:
        return column.reference
    if not known[row][selector]:
        return None
    return schema.vocabularies[spec.columns[selector].vocabulary][int(values[row][selector])]


def permute_arrays(arrays: dict, schema, permutations: dict, np) -> dict:
    """Move rows of each permuted group (new row = permutation[old row]) and remap every reference to it."""
    out = {key: value.copy() for key, value in arrays.items()}
    for group, permutation in permutations.items():
        spec = schema.tensors[group]
        keys = [group, spec.known_key, spec.occupied_key] + (["action_mask"] if group == "actions" else [])
        for key in keys:
            moved = np.empty_like(arrays[key])
            moved[permutation] = arrays[key]
            out[key] = moved
    for group, spec in schema.tensors.items():
        if not spec.columns or spec.occupied_key is None:
            continue
        values, known = out[group], out[spec.known_key]
        for column_index, column in enumerate(spec.columns):
            if column.reference is None:
                continue
            for row in np.flatnonzero(out[spec.occupied_key]):
                if not known[row][column_index]:
                    continue
                target = reference_target(schema, group, values, known, row, column_index)
                if target in permutations:
                    values[row, column_index] = permutations[target][int(values[row, column_index])]
    return out


def s2_identity(config: TransformerConfig = TINY, *, mapping: str = MAPPING, **changes) -> NeuralIdentity:
    model_id, features_id, action_id = TRANSFORMER_LEARNER_IDENTITY_IDS
    fields = dict(model_id=model_id, features_id=features_id, action_id=action_id, objective_id=DISTILL_OBJECTIVE_ID,
                  accepted_raw_mode_mappings={"standard-standalone": mapping},
                  config_hashes={"model": model_config_hash(config), "distill": distill_config_hash(HYPER)})
    fields.update(changes)
    return NeuralIdentity(**fields)


def ff_identity(config: ModelConfig = FF_TINY) -> NeuralIdentity:
    model_id, features_id, action_id = STANDARD_LEARNER_IDENTITY_IDS
    return NeuralIdentity(model_id=model_id, features_id=features_id, action_id=action_id,
                          objective_id=DISTILL_OBJECTIVE_ID, accepted_raw_mode_mappings={"standard-standalone": MAPPING},
                          config_hashes={"model": model_config_hash(config), "distill": distill_config_hash(HYPER)})


def save_student(path: Path, policy, identity: NeuralIdentity, *, step: int = 1) -> Path:
    """Save ``policy`` as a distillation checkpoint (the trainer's metadata layout)."""
    import torch

    from spire_learn.neural.checkpoint import identity_payload, save_checkpoint

    save_checkpoint(
        Path(path), policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3),
        rng_state={"policy": {"torch": torch.Generator(device="cpu").manual_seed(1).get_state()},
                   "minibatch": {"torch": torch.Generator(device="cpu").manual_seed(2).get_state()},
                   "curriculum": {"python": random.Random(3).getstate()},
                   "controller": {"python": random.Random(4).getstate()}},
        metadata={
            "identity": identity_payload(identity),
            "run_metadata": {"fixture": "tests.neural.test_transformer", "step": step},
            "training_state": {
                "optimizer_step": step, "hyperparameter_id": identity.config_hashes["distill"],
                "rng_domains": RNG_DOMAINS,
                "counters": {"optimizer_steps": step, "rows_seen": 0, "policy_rows": 0, "value_rows": 0},
                "update_id": f"step-{step}", "behavior_policy_version": step,
                "next_environment_seed_ordinal": 0, "next_decision_sequence": 0, "saved_inflight_record_count": 0,
            },
            "parent_checkpoint": None,
        })
    return Path(path)


class EntityTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np, cls.torch, cls.shards = _load_modules()
        cls.torch.set_num_threads(1)
        from spire_learn.neural.transformer import transformer_plan

        cls.schema = cls.shards._schema()
        cls.plan = transformer_plan(cls.schema)
        cls.arrays = rich_arrays(cls.schema, cls.shards, cls.np)
        cls.rich = to_frame(cls.arrays, cls.schema)
        cls.frames = [cls.rich, *cls.shards.frames(cls.schema)]
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, config: TransformerConfig = TINY, *, seed: int = 7):
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

    # --- build -----------------------------------------------------------------------------

    def test_parameter_counts(self):
        from spire_learn.neural import model_families as families
        from spire_learn.neural.transformer import transformer_parameter_count

        for size in TRANSFORMER_SIZES:
            for encoding in NUMBER_ENCODINGS:
                config = transformer_size(size, encoding)
                with self.subTest(size=size, encoding=encoding):
                    policy = self.build(config)
                    actual = sum(int(parameter.numel()) for parameter in policy.parameters())
                    self.assertEqual(actual, transformer_parameter_count(self.plan, config))
                    self.assertLessEqual(actual, TRANSFORMER_PARAMETER_CAP)
                    # Only parameters are saved: the sinusoid frequencies are not state.
                    self.assertEqual(set(policy.state_dict()), {name for name, _ in policy.named_parameters()})
                    del policy
        # The registry enforces the family cap on the built policy too.
        from spire_learn.contracts import ContractError

        capped = replace(families.TRANSFORMER_FAMILY, parameter_cap=1000)
        with mock.patch.dict(families._FAMILIES, {TRANSFORMER_MODEL_ID: capped}), \
                self.assertRaisesRegex(ContractError, "over its cap"):
            self.build(TINY)

    def test_number_encodings_build(self):
        from spire_learn.neural.transformer import _bucket_tensor, bucket_index

        torch = self.torch
        batch = self.encode(self.frames)
        mask = batch.action_mask
        outputs = {}
        for encoding in NUMBER_ENCODINGS:
            with self.subTest(encoding=encoding):
                policy = self.build(replace(TINY, number_encoding=encoding))
                policy.train()
                output = policy(batch)
                self.assertEqual(tuple(output.logits.shape), tuple(mask.shape))
                self.assertEqual(tuple(output.values.shape), (len(self.frames),))
                self.assertEqual(tuple(output.aux_values.shape), (len(self.frames), 2))
                self.assertTrue(torch.isfinite(output.logits).all().item())
                self.assertTrue(torch.isfinite(output.values).all().item())
                # Every parameter is reached (the rich frame occupies every group and resolves every reference).
                (output.logits.masked_select(mask).sum() + output.values.sum() + output.aux_values.sum()).backward()
                unreached = [name for name, parameter in policy.named_parameters() if parameter.grad is None]
                self.assertEqual(unreached, [])
                self.assertTrue(all(torch.isfinite(parameter.grad).all().item() for parameter in policy.parameters()))
                outputs[encoding] = output.values.detach()
        self.assertFalse(torch.equal(outputs["bucket"], outputs["scaled"]))
        self.assertFalse(torch.equal(outputs["scaled"], outputs["sinusoidal"]))
        # The tensor buckets are the stdlib reference buckets.
        samples = [0, 1, -1, 15, -15, 16, 31, 32, 1000, 2**31 - 1, -(2**31), 2**40]
        self.assertEqual(_bucket_tensor(torch, torch.tensor(samples, dtype=torch.int64)).tolist(),
                         [bucket_index(value) for value in samples])
        # The sinusoidal features: the scaled value, then sin and cos at each frequency.
        policy = self.build(replace(TINY, number_encoding="sinusoidal"))
        group = self.plan.groups["player"]
        features = policy._number(group, group.numeric[0], torch.tensor([0, 7, -7], dtype=torch.int64))
        self.assertEqual(tuple(features.shape), (3, 17))
        self.assertAlmostEqual(float(features[1, 0]), math.log1p(7), places=5)
        self.assertAlmostEqual(float(features[2, 0]), -math.log1p(7), places=5)
        self.assertAlmostEqual(float(features[1, 1]), math.sin(7.0), places=5)
        self.assertAlmostEqual(float(features[1, 9]), math.cos(7.0), places=5)

    # --- tokens ----------------------------------------------------------------------------

    def test_every_row_group_tokenised(self):
        torch, np = self.torch, self.np
        policy = self.build(replace(TINY, number_encoding="scaled"))
        batch = self.encode([self.rich])
        layout = policy.tokenise(batch)
        token_group = layout.token_group[0]
        self.assertEqual(int(token_group[0]), 0)  # the context token
        occupied_total = 0
        for index, name in enumerate(self.plan.token_groups):
            spec = self.schema.tensors[name]
            occupied = int(self.arrays[spec.occupied_key].sum())
            self.assertGreater(occupied, 0, name)
            self.assertEqual(int((token_group == index + 1).sum()), occupied, name)
            occupied_total += occupied
        self.assertEqual(int(layout.valid.sum()), 1 + occupied_total)
        self.assertEqual(set(self.plan.token_groups), set(ROWS) - {"actions"})
        # Every token group and every context group reaches the outputs.
        base = self.run_policy(policy, batch)
        for name in (*self.plan.token_groups, *self.plan.context_groups):
            with self.subTest(group=name):
                group = self.plan.groups[name]
                values = {key: tensor.clone() for key, tensor in batch.values.items()}
                row = int(np.flatnonzero(self.arrays[self.schema.tensors[name].occupied_key])[0]) \
                    if name in self.plan.token_groups else 0
                if group.numeric:
                    values[name][0, row, group.numeric[0]] += 3
                else:
                    field, size = group.categorical[0]
                    values[name][0, row, field] = (values[name][0, row, field] + 1) % size
                changed = self.run_policy(policy, replace(batch, values=values))
                self.assertFalse(torch.equal(changed.values, base.values))

    def test_padding_rows_never_tokens(self):
        torch = self.torch
        policy = self.build()
        full, trimmed = self.encode(self.frames), self.encode(self.frames, trim=True)
        self.assertTrue(full.values["combat_cards"].shape[1] > trimmed.values["combat_cards"].shape[1])
        expected = self.run_policy(policy, full)
        # A trimmed batch has the same tokens, so exactly the same outputs.
        self.assert_same(expected, self.run_policy(policy, trimmed))
        layout = policy.tokenise(full)
        occupied = sum(full.occupied[name].sum(dim=1) for name in self.plan.token_groups)
        self.assertTrue(torch.equal(layout.valid.sum(dim=1), 1 + occupied))
        self.assertTrue(torch.equal(layout.counts, 1 + occupied))
        for name in self.plan.token_groups:
            batch_index, row_index = layout.rows[name]
            self.assertTrue(torch.equal(torch.stack((batch_index, row_index), dim=1), full.occupied[name].nonzero()))
        # Garbage in every unoccupied row (values and known flags) never reaches an output.
        values = {key: tensor.clone() for key, tensor in full.values.items()}
        known = {key: tensor.clone() for key, tensor in full.known.items()}
        for name in (*self.plan.token_groups, "actions"):
            free = ~full.occupied[name].unsqueeze(-1)
            garbage = (torch.arange(values[name].numel(), dtype=torch.int64).reshape(values[name].shape) * 7919) \
                % 2001 - 1000
            values[name] = torch.where(free, garbage, values[name])
            known[name] = known[name] | free
        noisy = replace(full, values=values, known=known)
        self.assert_same(expected, self.run_policy(policy, noisy))
        # The public contract: an ordinary group's padding rows are known-0, or the frame is refused.
        from spire_learn.contracts import ContractError

        known_padding = {key: value.copy() for key, value in self.arrays.items()}
        known_padding[self.schema.tensors["combat_cards"].known_key][0, 0] = 1  # row 0 is not occupied
        with self.assertRaises(ContractError):
            to_frame(known_padding, self.schema)
        # potions and orbs may publish known empty slots. Those rows are still never tokens; they reach the
        # outputs only through the public known_coverage (here perturbed), so with the coverage map silenced
        # the outputs are exactly equal.
        slots = {key: value.copy() for key, value in self.arrays.items()}
        for name, rows in (("potions", [2, 4]), ("orbs", [3, 5])):
            spec = self.schema.tensors[name]
            self.assertFalse(slots[spec.occupied_key][rows].any())
            slots[spec.known_key][rows] = 1
        plain, published = self.encode([self.rich]), self.encode([to_frame(slots, self.schema)])
        self.assertFalse(torch.equal(plain.known_coverage["potions"], published.known_coverage["potions"]))
        self.assertFalse(torch.equal(plain.known_coverage["orbs"], published.known_coverage["orbs"]))
        for name in ("valid", "token_group", "counts"):
            self.assertTrue(torch.equal(getattr(policy.tokenise(plain), name), getattr(policy.tokenise(published), name)))
        with torch.no_grad():
            policy.coverage.weight.zero_()
        self.assert_same(self.run_policy(policy, plain), self.run_policy(policy, published))

    def test_permutation_invariance_with_reference_remap(self):
        torch, np = self.torch, self.np
        policy = self.build()
        generator = np.random.default_rng(1103)
        permutations = {name: generator.permutation(self.schema.tensors[name].shape[0]) for name in ROWS}
        moved = permute_arrays(self.arrays, self.schema, permutations, np)
        permuted = to_frame(moved, self.schema)  # still a valid public frame: every reference still resolves
        self.assertFalse(all(np.array_equal(moved[key], self.arrays[key]) for key in ROWS))
        original = self.run_policy(policy, self.encode([self.rich]))
        # Trimmed: the permuted rows leave holes inside each group's trimmed extent, and references point past them.
        result = self.run_policy(policy, self.encode([permuted], trim=True))
        for name in ("values", "aux_values"):
            self.assertTrue(torch.allclose(getattr(original, name), getattr(result, name), atol=1e-5, rtol=1e-4))
        legal = [int(action) for action in np.flatnonzero(self.arrays["action_mask"])]
        moved_legal = [int(permutations["actions"][action]) for action in legal]
        self.assertTrue(torch.allclose(original.logits[0, legal], result.logits[0, moved_legal], atol=1e-5, rtol=1e-4))
        self.assertEqual(int((result.logits[0] != 0).sum()), len(legal))
        # The references themselves matter: pointing one at another occupied row changes the outputs.
        redirected = {key: value.copy() for key, value in self.arrays.items()}
        set_reference(redirected, self.schema, "combat_cards", 2, "permanent_row", 7, None)
        other = self.run_policy(policy, self.encode([to_frame(redirected, self.schema)]))
        self.assertFalse(torch.equal(other.values, original.values))
        redirected = {key: value.copy() for key, value in self.arrays.items()}
        set_reference(redirected, self.schema, "actions", 0, "target_row", 1, "enemies")
        other = self.run_policy(policy, self.encode([to_frame(redirected, self.schema)]))
        self.assertTrue(torch.equal(other.values, original.values))  # actions are not tokens
        self.assertNotEqual(float(other.logits[0, 0]), float(original.logits[0, 0]))

    def edited(self, batch, group: str, row: int, field: str, value: int, *, known: bool, selector="keep"):
        """``batch`` (one frame) with one reference cell set; ``selector`` "keep", None (unknown) or a group."""
        spec = self.schema.tensors[group]
        index = _field(spec, field)
        values = {key: tensor.clone() for key, tensor in batch.values.items()}
        flags = {key: tensor.clone() for key, tensor in batch.known.items()}
        values[group][0, row, index] = value
        flags[group][0, row, index] = known
        if selector != "keep":
            selector_index = _field(spec, spec.columns[index].reference)
            flags[group][0, row, selector_index] = selector is not None
            if selector is not None:
                values[group][0, row, selector_index] = self.schema.vocabularies["row_group"].index(selector)
        return replace(batch, values=values, known=flags)

    def silence_known_flag(self, policy, group_name: str, field: str) -> None:
        """Zero the group-input weights of one reference's "index known" flag (the only documented difference)."""
        from spire_learn.neural.transformer import _ENCODING_FEATURES

        group = self.plan.groups[group_name]
        position = [ref.field for ref in group.references].index(field)
        column = len(group.numeric) * (_ENCODING_FEATURES[policy._encoding] + 1) + 2 * position
        with self.torch.no_grad():
            policy.group_inputs[f"g{group.index}"].weight[:, column] = 0.0

    def test_unresolved_references_gather_nothing(self):
        torch = self.torch
        policy = self.build()
        references = (("combat_cards", "permanent_row"), ("powers", "owner_row"), ("selection", "source_row"),
                      ("known_draw", "combat_card_row"), ("actions", "target_row"), ("actions", "source_row"))
        for group, field in references:
            self.silence_known_flag(policy, group, field)
        full, trimmed = self.encode([self.rich]), self.encode([self.rich], trim=True)
        # frames[3] occupies powers, known_draw and actions only: enemies and combat_cards are empty groups,
        # trimmed to one unoccupied row.
        sparse = self.encode([self.frames[3]], trim=True)
        self.assertFalse(sparse.occupied["combat_cards"].any() or sparse.occupied["enemies"].any())
        self.assertEqual(sparse.values["combat_cards"].shape[1], 1)
        self.assertLess(trimmed.values["permanent_cards"].shape[1], 500)
        # (batch, group, row, field, value, selector); every row is occupied, every reference is unresolved,
        # and a row value of 0 is what a reference to the context token would carry.
        cases = {
            "-1": (full, "combat_cards", 2, "permanent_row", -1, "keep"),
            "-1 with the player (context) selector": (full, "powers", 0, "owner_row", -1, "keep"),
            "an unoccupied target row": (full, "combat_cards", 2, "permanent_row", 1, "keep"),
            "an unoccupied row 0 of a token group": (full, "powers", 6, "owner_row", 0, "enemies"),
            "out of range": (full, "combat_cards", 2, "permanent_row", 5000, "keep"),
            "past the context's single row": (full, "powers", 0, "owner_row", 1, "player"),
            "past a trimmed extent": (trimmed, "combat_cards", 2, "permanent_row", 500, "keep"),
            "an unknown selector": (full, "powers", 6, "owner_row", 0, None),
            "a non-token selector (map_nodes)": (full, "powers", 6, "owner_row", 0, "map_nodes"),
            "a non-token selector (none)": (full, "selection", 1, "source_row", 0, "none"),
            "a non-token selector (offers)": (full, "selection", 1, "source_row", 0, "offers"),
            "an action target in an unoccupied row": (full, "actions", 0, "target_row", 0, "enemies"),
            "an action source with a non-token selector": (full, "actions", 8, "source_row", 0, "map_nodes"),
            "into an empty trimmed group": (sparse, "known_draw", 9, "combat_card_row", 0, "keep"),
            "an owner in an empty trimmed group": (sparse, "powers", 3, "owner_row", 0, "enemies"),
            "an action target in an empty trimmed group": (sparse, "actions", 1000, "target_row", 0, "enemies"),
        }
        for name, (batch, group, row, field, value, selector) in cases.items():
            with self.subTest(case=name):
                self.assertTrue(bool(batch.occupied[group][0, row]))
                on = self.run_policy(policy, self.edited(batch, group, row, field, value, known=True, selector=selector))
                off = self.run_policy(policy, self.edited(batch, group, row, field, value, known=False,
                                                          selector=selector))
                self.assert_same(on, off)
        # The control: a reference that resolves does gather (known on and off differ), including one to the
        # context token, so the equalities above are not vacuous.
        for group, row, field, value, selector in (("combat_cards", 2, "permanent_row", 3, "keep"),
                                                   ("powers", 0, "owner_row", 0, "player"),
                                                   ("actions", 0, "target_row", 4, "enemies")):
            with self.subTest(control=f"{group}.{field}"):
                on = self.run_policy(policy, self.edited(full, group, row, field, value, known=True, selector=selector))
                off = self.run_policy(policy, self.edited(full, group, row, field, value, known=False,
                                                          selector=selector))
                self.assertFalse(torch.equal(on.logits, off.logits))
        # Independent of the flags: an unresolved reference adds exactly zero. With that reference column
        # known in the edited row only, zeroing its whole projection (weight and bias) changes nothing; an
        # unmasked projection would add the projection of token 0 and differ.
        twin = self.build()
        twin.load_state_dict(policy.state_dict(), strict=True)
        alone_cases = {
            "-1": (full, "combat_cards", 2, "permanent_row", -1, "keep"),
            "an action target in an unoccupied row": (full, "actions", 0, "target_row", 0, "enemies"),
            "a non-token selector (map_nodes)": (full, "powers", 6, "owner_row", 0, "map_nodes"),
            "into an empty trimmed group": (sparse, "known_draw", 9, "combat_card_row", 0, "keep"),
        }
        for name, (batch, group, row, field, value, selector) in alone_cases.items():
            with self.subTest(zeroed_projection=name):
                edited = self.alone(self.edited(batch, group, row, field, value, known=True, selector=selector),
                                    group, row, field)
                with self.zeroed_projection(twin, group, field):
                    self.assert_same(self.run_policy(policy, edited), self.run_policy(twin, edited))
        # The control: for a reference that resolves, zeroing its projection does change the outputs.
        for group, row, field, value, selector in (("combat_cards", 2, "permanent_row", 3, "keep"),
                                                   ("actions", 0, "target_row", 4, "enemies")):
            with self.subTest(zeroed_control=f"{group}.{field}"):
                edited = self.alone(self.edited(full, group, row, field, value, known=True, selector=selector),
                                    group, row, field)
                with self.zeroed_projection(twin, group, field):
                    self.assertFalse(torch.equal(self.run_policy(policy, edited).logits,
                                                 self.run_policy(twin, edited).logits))

    def alone(self, batch, group: str, row: int, field: str):
        """``batch`` with one reference column unknown in every row but ``row`` (of frame 0)."""
        index = _field(self.schema.tensors[group], field)
        flags = {key: tensor.clone() for key, tensor in batch.known.items()}
        keep = flags[group][0, row, index].clone()
        flags[group][0, :, index] = False
        flags[group][0, row, index] = keep
        return replace(batch, known=flags)

    @contextlib.contextmanager
    def zeroed_projection(self, policy, group_name: str, field: str):
        """Zero one reference column's projection (weight and bias) for the duration, then restore it."""
        group = self.plan.groups[group_name]
        projection = policy.reference_projections[f"g{group.index}f{_field(self.schema.tensors[group_name], field)}"]
        saved = {name: tensor.detach().clone() for name, tensor in projection.state_dict().items()}
        with self.torch.no_grad():
            projection.weight.zero_()
            projection.bias.zero_()
        try:
            yield
        finally:
            projection.load_state_dict(saved)

    # --- actions ---------------------------------------------------------------------------

    def test_only_legal_scored(self):
        torch = self.torch
        policy = self.build()
        batch = self.encode(self.frames)
        mask = batch.action_mask
        widths = []
        hook = policy.action_block.register_forward_hook(lambda module, inputs, output: widths.append(inputs[0].shape))
        self.addCleanup(hook.remove)
        output = self.run_policy(policy, batch)
        self.assertEqual(widths[-1][1], int(mask.sum(dim=1).max()))  # one query per legal action, no more
        self.assertTrue(torch.equal(output.logits.masked_select(~mask), torch.zeros(int((~mask).sum()))))
        # Drop every other legal action: the remaining logits are unchanged (queries never attend to one
        # another) and the values do not depend on the actions at all.
        fewer = mask.clone()
        for row in range(mask.shape[0]):
            legal = mask[row].nonzero().flatten().tolist()
            for action in legal[1::2]:
                fewer[row, action] = False
        self.assertLess(int(fewer.sum()), int(mask.sum()))
        smaller = self.run_policy(policy, replace(batch, action_mask=fewer))
        self.assertTrue(torch.allclose(smaller.logits[fewer], output.logits[fewer], atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.equal(smaller.logits[~fewer], torch.zeros(int((~fewer).sum()))))
        self.assertTrue(torch.equal(smaller.values, output.values))
        self.assertTrue(torch.equal(smaller.aux_values, output.aux_values))
        # An occupied action that is not legal is never read: garbage in its row changes nothing.
        values = {key: tensor.clone() for key, tensor in batch.values.items()}
        dropped = (mask & ~fewer).unsqueeze(-1)
        values["actions"] = torch.where(dropped, torch.full_like(values["actions"], 3), values["actions"])
        self.assert_same(smaller, self.run_policy(policy, replace(batch, action_mask=fewer, values=values)))
        # act picks the reflex argmax among legal actions and reports its log-probability and value.
        from spire_learn.neural.model_families import reflex_argmax

        choices = policy.act(self.frames, generator=None, deterministic=True)
        expected = reflex_argmax(output.logits, mask).tolist()
        self.assertEqual([choice.original_action_index for choice in choices], expected)
        log_probs = torch.log_softmax(output.logits.masked_fill(~mask, -torch.inf), dim=-1)
        for row, choice in enumerate(choices):
            self.assertAlmostEqual(choice.log_probability, float(log_probs[row, expected[row]]), places=5)
            self.assertAlmostEqual(choice.value, float(output.values[row]), places=5)
        self.assertEqual(policy.values(self.frames), tuple(float(value) for value in output.values))

    # --- checkpoints -----------------------------------------------------------------------

    def test_refuses_ff_and_s3_checkpoints(self):
        from spire_learn.neural.checkpoint import CheckpointCompatibilityError, load_checkpoint
        from spire_learn.neural.model import build_policy
        from spire_learn.neural.policy_player import CheckpointPlayer

        torch = self.torch
        root = self.root / "refusals"
        root.mkdir()
        s2 = self.build(seed=3)
        s2_path = save_student(root / "s2", s2, s2_identity())
        ff = build_policy(self.schema, FF_TINY, seed=3, device="cpu", objective_id=DISTILL_OBJECTIVE_ID,
                          model_id=STANDARD_LEARNER_IDENTITY_IDS[0], features_id=STANDARD_LEARNER_IDENTITY_IDS[1])
        ff_path = save_student(root / "ff", ff, ff_identity())
        # Stand-ins for a later family (S3): the S2 weights under another features id, or another model id.
        s3_features = save_student(root / "s3-features", s2, s2_identity(
            features_id="spire-learn-combat-features-entity-cards-v1"))
        s3_model = save_student(root / "s3-model", s2, s2_identity(model_id="spire-learn-combat-entity-cards-v1"))
        loaded = load_checkpoint(s2_path, s2_identity(), purpose="evaluate", device="cpu")
        fresh = self.build(seed=99)
        fresh.load_state_dict(dict(loaded.model_tensors), strict=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), s2.state_dict().values())))
        for name, path in (("FF", ff_path), ("S3 features", s3_features), ("S3 model", s3_model)):
            for purpose in ("evaluate", "warm_start", "resume"):
                with self.subTest(checkpoint=name, purpose=purpose), self.assertRaises(CheckpointCompatibilityError):
                    load_checkpoint(path, s2_identity(), purpose=purpose, device="cpu")
        # Nor does S2 load as FF, or under another transformer config.
        with self.assertRaises(CheckpointCompatibilityError):
            load_checkpoint(s2_path, ff_identity(), purpose="evaluate", device="cpu")
        scaled = replace(TINY, number_encoding="scaled")
        with self.assertRaises(CheckpointCompatibilityError):
            load_checkpoint(s2_path, s2_identity(scaled), purpose="evaluate", device="cpu")
        # The tensors could not cross either.
        with self.assertRaises(RuntimeError):
            fresh.load_state_dict(dict(ff.state_dict()), strict=True)
        # CheckpointPlayer dispatches to the transformer family and refuses the FF checkpoint.
        with mock.patch("spire_learn.neural.policy_player.pin_torch_threads"):
            with self.assertRaises(CheckpointCompatibilityError):
                CheckpointPlayer(ff_path, self.schema, seed=0, expected_identity=s2_identity(), model_config=TINY)
            player = CheckpointPlayer(s2_path, self.schema, seed=0, expected_identity=s2_identity(),
                                      model_config=TINY, deterministic=True)
        legal = [int(action) for action in self.np.flatnonzero(self.arrays["action_mask"])]
        from spire_learn.neural.model_families import reflex_argmax

        batch = self.encode([self.rich])
        output = self.run_policy(s2, batch)
        self.assertEqual(player.act(self.rich, legal), int(reflex_argmax(output.logits, batch.action_mask)[0]))

    # --- the trainer and the runtime -------------------------------------------------------

    def test_distill_trainer_trains_the_transformer(self):
        from spire_learn.neural import distill_data, distill_train
        from spire_learn.neural.checkpoint import load_checkpoint
        from spire_learn.neural.transformer import transformer_parameter_count

        np = self.np
        frames = [self.rich, self.frames[1], self.frames[3], to_frame(permute_arrays(
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
            model=TINY, distill=HYPER, identity=s2_identity(), model_table="transformer")
        output = self.root / "trainer-run"
        summary = distill_train.train_student(config, self.schema, dataset, output, log_every=1)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["parameter_count"], transformer_parameter_count(self.plan, TINY))
        self.assertEqual([item["step"] for item in summary["evaluations"]], [2, 3])
        for evaluation in summary["evaluations"]:
            for key in ("regret", "policy_loss", "outcome_mse", "hp_kept_mse", "realised_v2_mse", "value_loss"):
                self.assertTrue(math.isfinite(evaluation[key]), key)
        loaded = load_checkpoint(output / "checkpoint-final", config.identity, purpose="resume", device="cpu")
        student = distill_train.build_student(config, self.schema, device="cpu")
        self.assertEqual(student.model_id, TRANSFORMER_MODEL_ID)
        student.load_state_dict(dict(loaded.model_tensors), strict=True)
        # Training moved the weights away from the initialisation.
        initial = distill_train.build_student(config, self.schema, device="cpu")
        self.assertFalse(all(self.torch.equal(a, b) for a, b in zip(initial.state_dict().values(),
                                                                    student.state_dict().values())))

    def test_transformer_through_create_runtime_evaluate(self):
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.features import encode_frames
        from spire_learn.neural.model_families import reflex_argmax
        from spire_learn.neural.runner import _DeterministicPolicy, create_runtime
        from tests.unit.test_distill_config import _write, eval_toml

        torch = self.torch
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            text = eval_toml(folder, model_table=TINY_TABLE, hashes={"model": model_config_hash(TINY)},
                             learner=TRANSFORMER_LEARNER_IDENTITY_IDS)
            config = load_neural_config(_write(folder, "eval.toml", text))
            self.assertEqual(config.model_table, "transformer")
            identity = config.accepted_identities[0]
            saved = self.build(seed=11)
            checkpoint = save_student(folder / "checkpoint", saved, s2_identity(
                mapping=identity.accepted_raw_mode_mappings["standard-standalone"]), step=3)
            runtime = create_runtime(config, purpose="evaluate", checkpoint=checkpoint, policy_kind="checkpoint",
                                     action_selection="deterministic")
            try:
                policy = runtime.policy
                self.assertEqual((policy.model_id, policy.features_id, policy.objective_id),
                                 (TRANSFORMER_MODEL_ID, TRANSFORMER_FEATURES_ID, DISTILL_OBJECTIVE_ID))
                self.assertIsInstance(runtime.collector.policy, _DeterministicPolicy)
                self.assertEqual(policy._behavior_policy_version, 3)
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
