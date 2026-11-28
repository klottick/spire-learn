"""The committed card-metadata snapshot against the pinned Gym (``--card-metadata-live`` route; runtime container only).

``configs/neural/standard/card-metadata-gym-1.2.4.json`` is written by ``card_metadata_snapshot_entry.py`` and
committed by the controller. These tests check that it is exactly what the pinned Gym's
``spire_lab.card_metadata.load()`` returns now, that it joins the Standard compact ``card`` vocabulary with no
unknown card, and that the card-metadata transformer (S3) builds on it and runs a forward pass. The Gym, numpy and
torch are imported lazily, never at discovery.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from spire_learn.neural.config import CARD_METADATA_FINGERPRINT

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "configs/neural/standard/card-metadata-gym-1.2.4.json"


def _live_metadata() -> dict:
    try:
        from spire_lab import card_metadata
    except ImportError:
        raise unittest.SkipTest("the Gym is only importable in the runtime container")
    return card_metadata.load()


class CardMetadataLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live = _live_metadata()
        if not SNAPSHOT.exists():
            raise AssertionError(f"{SNAPSHOT} is missing: run card_metadata_snapshot_entry.py and copy its output")
        cls.snapshot = json.loads(SNAPSHOT.read_text())

    def test_snapshot_is_the_live_descriptor(self):
        self.assertEqual(self.live["fingerprint"], CARD_METADATA_FINGERPRINT)
        self.assertEqual(self.snapshot["fingerprint"], CARD_METADATA_FINGERPRINT)
        self.assertEqual(self.snapshot, self.live)
        from spire_learn.neural import transformer

        self.assertEqual(transformer.load_card_metadata()["fingerprint"], CARD_METADATA_FINGERPRINT)

    def test_snapshot_joins_the_compact_card_vocabulary(self):
        from spire_learn.neural import transformer
        from tests.unit.test_transformer_config import compact_schema

        vocabulary = compact_schema().vocabularies["card"]
        tables = transformer.card_metadata_tables(self.snapshot, vocabulary)
        # Every card id has a row; only ``padding`` has none, and the snapshot has no id outside the vocabulary.
        self.assertEqual(tables.missing, ("padding",))
        self.assertEqual(tables.unused, ())
        self.assertEqual(tables.rows, len(vocabulary) - 1)
        self.assertEqual(tables.levels, 2)
        by_id = {row["id"]: row for row in self.snapshot["cards"]}
        for code, card in enumerate(vocabulary):
            if card == "padding":
                continue
            faces = by_id[card]["faces"]
            for level in range(tables.levels):
                expected = transformer.card_face_features(faces[min(level, len(faces) - 1)] if faces else None)
                self.assertEqual(tables.faces[level][code], expected, (card, level))
            self.assertEqual(tables.static[code], transformer.card_static_features(by_id[card]), card)

    def test_card_metadata_transformer_builds_on_the_snapshot(self):
        try:
            import numpy
            import torch
        except ImportError:
            raise unittest.SkipTest("numpy and torch are only importable in the runtime container")
        from spire_learn.neural.config import (
            DISTILL_OBJECTIVE_ID, TRANSFORMER_CARDMETA_FEATURES_ID, TRANSFORMER_CARDMETA_MODEL_ID,
            TransformerCardMetaConfig,
        )
        from spire_learn.neural.features import encode_frames
        from spire_learn.neural.model_families import build_family_policy
        from spire_learn.neural.transformer import transformer_cardmeta_parameter_count
        from tests.neural import test_distill_data as shards

        shards._load_numpy_modules()
        schema = shards._schema()
        config = TransformerCardMetaConfig(layers=1, width=16, heads=2, number_encoding="bucket")
        policy = build_family_policy(
            TRANSFORMER_CARDMETA_MODEL_ID, schema, config, seed=1, device="cpu", objective_id=DISTILL_OBJECTIVE_ID,
            model_id=TRANSFORMER_CARDMETA_MODEL_ID, features_id=TRANSFORMER_CARDMETA_FEATURES_ID)
        policy.eval()
        self.assertEqual(sum(int(p.numel()) for p in policy.parameters()),
                         transformer_cardmeta_parameter_count(schema, config))
        batch = encode_frames(list(shards.frames(schema)), schema, device="cpu", trim_padding=True)
        with torch.no_grad():
            output = policy(batch)
        self.assertTrue(torch.isfinite(output.values).all().item())
        self.assertTrue(torch.isfinite(output.logits.masked_select(batch.action_mask)).all().item())
        self.assertIsNotNone(numpy)


if __name__ == "__main__":
    unittest.main()
