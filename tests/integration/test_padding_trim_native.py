"""Compare trimmed and full batches on 2,400 real compact Gym decisions."""

import os
from pathlib import Path
import tempfile
import unittest

from tests.unit import test_neural_standard_compact as compact


@unittest.skipUnless(os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1",
                     "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the pinned runtime")
class PaddingTrimNativeTests(unittest.TestCase):
    def test_trimmed_outputs_identical_on_real_v1_2_0_frames(self):
        import torch

        from spire_learn.neural.config import STANDARD_LEARNER_IDENTITY_IDS, load_neural_config
        from spire_learn.neural.distribution import masked_categorical, pad_to_dispatch_width
        from spire_learn.neural.envs import make_driver
        from spire_learn.neural.features import encode_frames
        from spire_learn.neural.model import build_policy

        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        rows, steps = 8, 300
        entries = compact.diverse_entries(compact._table_states())
        by_character = {
            character: [entry["id"] for entry in entries if entry["character"] == character]
            for character in compact.CHARACTERS
        }
        self.assertTrue(all(len(ids) == 8 for ids in by_character.values()))
        work = tempfile.TemporaryDirectory(dir=compact.WORK)
        self.addCleanup(work.cleanup)
        root = Path(work.name)
        manifest = compact._cases_manifest(root, entries)
        config_path = root / "compact.toml"
        config_path.write_text(compact.compact_text(rows=rows).replace(
            f'manifest_path = "{compact.CASES}"', f'manifest_path = "{manifest}"'))
        config = load_neural_config(config_path)
        driver = make_driver(config, config.cases)
        self.addCleanup(driver.close)
        schema = driver.schema
        model_id, features_id = STANDARD_LEARNER_IDENTITY_IDS[:2]
        policy = build_policy(schema, config.model, seed=1103, device="cpu",
                              model_id=model_id, features_id=features_id)
        generators = [torch.Generator().manual_seed(500) for _ in range(2)]
        slot_entries = [by_character[character] for character in compact.CHARACTERS for _ in range(2)]
        entry_indices = [i % 2 for i in range(rows)]
        seeds = list(range(500, 500 + rows))
        driver.reset(seeds=seeds, entry_ids=[ids[i] for ids, i in zip(slot_entries, entry_indices)])
        compared = 0
        trimmed = 0
        with torch.no_grad():
            for step in range(steps):
                frames = [row.frame for row in driver.rows]
                full, short = (encode_frames(frames, schema, device="cpu", trim_padding=flag)
                               for flag in (False, True))
                trimmed += short.values["combat_cards"].shape[1] < full.values["combat_cards"].shape[1]
                self.assertTrue(torch.equal(full.action_mask, short.action_mask), step)
                for group, values in short.values.items():
                    width = values.shape[1]
                    self.assertTrue(torch.equal(full.values[group][:, :width], values), (step, group))
                    self.assertTrue(torch.equal(full.known[group][:, :width], short.known[group]),
                                    (step, group))
                    if group in short.occupied:
                        self.assertTrue(torch.equal(full.occupied[group][:, :width], short.occupied[group]),
                                        (step, group))
                    self.assertTrue(torch.equal(full.known_coverage[group], short.known_coverage[group]),
                                    (step, group))
                expected, actual = policy(full), policy(short)
                for name in ("logits", "values"):
                    self.assertTrue(torch.equal(getattr(expected, name).view(torch.int32),
                                                getattr(actual, name).view(torch.int32)), (step, name))
                distributions = [
                    masked_categorical(*pad_to_dispatch_width(output.logits, batch.action_mask,
                                                              schema.view.dense_action_width))
                    for output, batch in ((expected, full), (actual, short))
                ]
                self.assertTrue(torch.equal(distributions[0].mode(), distributions[1].mode()), step)
                actions, other = (dist.sample(generator=rng)
                                  for dist, rng in zip(distributions, generators, strict=True))
                self.assertTrue(torch.equal(actions, other), step)

                transitions = driver.step(actions.tolist())
                compared += len(transitions)
                ended = [item.terminated or item.truncated for item in transitions]
                if any(ended):
                    reset_ids = [None] * rows
                    for slot, done in enumerate(ended):
                        if done:
                            entry_indices[slot] += 2
                            ids = slot_entries[slot]
                            reset_ids[slot] = ids[entry_indices[slot] % len(ids)]
                    driver.reset(seeds=[1500 + step * rows + i for i in range(rows)],
                                 entry_ids=reset_ids, reset_mask=ended)
        self.assertEqual(compared, 2400)
        self.assertGreater(trimmed, 0)
