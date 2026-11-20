"""Aggregate report arithmetic on hand-written run-state tables."""
from __future__ import annotations

import unittest

from scripts.neural.run_states import aggregate_stats


class RunStateStatsTest(unittest.TestCase):
    def test_aggregates_on_synthetic_table(self):
        states = [
            {
                "character": "ironclad", "act": 1, "room_kind": "monster",
                "deck": [{"card": "strike", "upgrades": 0}, {"card": "bash", "upgrades": 1}],
                "relics": ["burning_blood", "anchor"], "potions": ["fire_potion"],
                "hp": 50, "max_hp": 100, "won": True, "tags": [],
            },
            {
                "character": "ironclad", "act": 1, "room_kind": "monster",
                "deck": [{"card": "strike", "upgrades": 1}, {"card": "strike", "upgrades": 0},
                         {"card": "shrug_it_off", "upgrades": 0}, {"card": "anger", "upgrades": 0}],
                "relics": ["burning_blood", "anchor"], "potions": ["fire_potion", "block_potion"],
                "hp": 90, "max_hp": 100, "won": False, "tags": ["potions_assumed_empty"],
            },
            {
                "character": "ironclad", "act": 1, "room_kind": "monster",
                "deck": [{"card": "strike", "upgrades": 0}, {"card": "bash", "upgrades": 0},
                         {"card": "shrug_it_off", "upgrades": 1}, {"card": "inflame", "upgrades": 0},
                         {"card": "anger", "upgrades": 0}, {"card": "anger", "upgrades": 0}],
                "relics": ["burning_blood", "vajra"], "potions": [],
                "hp": 25, "max_hp": 50, "won": True, "tags": [],
            },
        ]
        result = aggregate_stats(
            {"schema": "spire-learn-run-states-v1", "states": states},
            {"strike": "attack", "bash": "attack", "anger": "attack",
             "shrug_it_off": "skill", "inflame": "power"},
        )
        group = result["strata"][0]
        self.assertEqual((group["character"], group["act"], group["room_kind"]),
                         ("ironclad", 1, "monster"))
        self.assertEqual(group["states"], 3)
        self.assertEqual(group["deck_size"], {"median": 4, "iqr": 2})
        self.assertEqual(group["upgrade_fraction"], 0.25)
        self.assertEqual(group["card_type_mix"], {"attack": 0.75, "power": 0.083333, "skill": 0.166667})
        self.assertEqual(group["top_cards"], [
            {"card": "strike", "share": 1.0},
            {"card": "anger", "share": 0.666667},
            {"card": "bash", "share": 0.666667},
            {"card": "shrug_it_off", "share": 0.666667},
            {"card": "inflame", "share": 0.333333},
        ])
        self.assertEqual(group["top_relics"], [
            {"relic": "burning_blood", "share": 1.0},
            {"relic": "anchor", "share": 0.666667},
            {"relic": "vajra", "share": 0.333333},
        ])
        self.assertEqual(group["mean_potions"], 1.0)
        self.assertEqual(group["hp_fraction"], 0.633333)
        self.assertEqual(group["win_rate"], 0.666667)
        self.assertEqual(group["potions_assumed_empty_share"], 0.333333)


if __name__ == "__main__":
    unittest.main()
