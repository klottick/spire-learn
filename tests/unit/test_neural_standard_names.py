"""Run-file names to Spire Gym Standard ids."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import unittest

from spire_learn.neural.standard_names import NameAliases, load_aliases

ALIASES = Path(__file__).resolve().parents[2] / "configs/neural/standard/name-aliases.json"
_RUN_NATIVE = os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1"


class StandardNamesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aliases = load_aliases(ALIASES)

    def test_card_upgrades_parsed(self):
        self.assertEqual(self.aliases.card("Bash+1"), ("bash", 1))
        self.assertEqual(self.aliases.card("Bash"), ("bash", 0))
        self.assertEqual(self.aliases.card("Searing Blow+12"), ("searing_blow", 12))
        self.assertEqual(self.aliases.card("Pommel Strike+1"), ("pommel_strike", 1))

    def test_starter_aliases(self):
        for name, card in (("Strike_R", "strike"), ("Defend_R", "defend"),
                           ("Strike_G", "strike_g"), ("Defend_G", "defend_g"),
                           ("Strike_B", "strike_b"), ("Defend_B", "defend_b"),
                           ("Strike_P", "strike_p"), ("Defend_P", "defend_p")):
            self.assertEqual(self.aliases.card(name), (card, 0), name)
        self.assertEqual(self.aliases.card("Strike_R+1"), ("strike", 1))

    def test_renamed_cards(self):
        self.assertEqual(self.aliases.card("Wireheading"), ("foresight", 0))
        self.assertEqual(self.aliases.card("Ghostly"), ("apparition", 0))
        self.assertEqual(self.aliases.card("Ghostly Armor"), ("ghostly_armor", 0))

    def test_relics_and_potions(self):
        self.assertEqual(self.aliases.relic("Burning Blood"), "burning_blood")
        self.assertEqual(self.aliases.relic("Paper Frog"), "paper_phrog")
        self.assertEqual(self.aliases.potion("Block Potion"), "block_potion")
        self.assertEqual(self.aliases.potion("SteroidPotion"), "flex_potion")

    def test_unknown_returns_none(self):
        self.assertIsNone(self.aliases.card("Not A Card"))
        self.assertIsNone(self.aliases.card("SKIP"))
        self.assertIsNone(self.aliases.card("Bash+x"))
        self.assertIsNone(self.aliases.relic("Not A Relic"))
        self.assertIsNone(self.aliases.potion("Not A Potion"))

    def test_alias_table_hash_stable(self):
        self.assertEqual(self.aliases.sha256, hashlib.sha256(ALIASES.read_bytes()).hexdigest())
        self.assertEqual(load_aliases(ALIASES).sha256, self.aliases.sha256)

    def test_alias_targets_are_known_cards(self):
        document = json.loads(ALIASES.read_text())
        known = set(document["gym_card_ids"])
        self.assertTrue(set(document["cards"].values()) <= known)

    def test_card_types(self):
        aliases = load_aliases(ALIASES)
        self.assertEqual([aliases.card_type(c) for c in ("clothesline", "defend", "inflame", "regret", "nope")],
                         ["attack", "skill", "power", "curse", None])
        import tempfile
        document = json.loads(ALIASES.read_text())
        del document["gym_card_types"]["bash"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_aliases(path)

    def test_conflicting_pairs_rejected(self):
        import tempfile
        document = json.loads(ALIASES.read_text())
        document["relics"]["silent"]["Burning Blood"] = "anchor"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                load_aliases(path)


@unittest.skipUnless(_RUN_NATIVE, "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the approved runtime")
class StandardNamesGymTests(unittest.TestCase):
    def test_alias_targets_exist_in_gym(self):
        from spire_learn.neural.standard_names import gym_card_metadata, gym_rejections
        aliases = load_aliases(ALIASES)
        document = json.loads(ALIASES.read_text())
        metadata = gym_card_metadata()
        self.assertEqual(metadata["fingerprint"], document["gym_card_metadata_fingerprint"])
        self.assertEqual(sorted(row["id"] for row in metadata["cards"]), document["gym_card_ids"])
        self.assertEqual({row["id"]: row["type"] for row in metadata["cards"]}, document["gym_card_types"])
        self.assertEqual(gym_rejections(aliases), [])


if __name__ == "__main__":
    unittest.main()
