"""Accepted public combat views: identity, tensor groups and lifecycle values.

Dependency-free. Each view is a separate schema; the Standard combat view is
not a superset of the common view.
"""

from __future__ import annotations

from dataclasses import dataclass

from spire_learn.contracts import ContractError
from spire_learn.neural.gym_contracts import CONTRACTS


# Synthetic structural fixtures use this legacy common-view fingerprint.
_LEGACY_SYNTHETIC_FINGERPRINT = "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78"


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    profile_id: str
    fingerprint_domain: bytes
    groups: tuple[str, ...]
    occupied_groups: frozenset[str]
    lifecycle_view_id: str
    routings: frozenset[str]
    outcomes: frozenset[str]

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset(
            set(self.groups)
            | {f"{group}_known" for group in self.groups}
            | {f"{group}_occupied" for group in self.occupied_groups}
            | {"action_mask"}
        )


COMMON_V1 = ViewSpec(
    view_id="spire-gym-combat-policy-view-v1",
    profile_id="IroncladCombatPolicy",
    fingerprint_domain=b"spire-gym/combat-policy-view/v1\0",
    groups=(
        "player", "permanent_cards", "combat_cards", "enemies", "powers", "relics",
        "potions", "selection", "selection_counts", "actions", "combat_state", "known_draw",
    ),
    occupied_groups=frozenset({
        "permanent_cards", "combat_cards", "enemies", "powers", "relics", "potions",
        "selection", "actions", "known_draw",
    }),
    lifecycle_view_id="spire-gym-combat-policy-view-v1",
    routings=frozenset({"combat", "combat_selection", "noncombat"}),
    outcomes=frozenset({"victory", "death"}),
)
STANDARD_COMBAT_V1 = ViewSpec(
    view_id="spire-gym-standard-combat-view-v1",
    profile_id="StandardCombat",
    fingerprint_domain=b"spire-gym/combat-view/standard/v1\0",
    groups=(
        "combat_context", "player", "permanent_cards", "combat_cards", "enemies", "powers",
        "relics", "potions", "orbs", "selection", "selection_counts", "actions",
        "combat_state", "known_draw", "stance_choices", "combat_card_offers",
    ),
    occupied_groups=frozenset({
        "permanent_cards", "combat_cards", "enemies", "powers", "relics", "potions", "orbs",
        "selection", "actions", "known_draw", "stance_choices", "combat_card_offers",
    }),
    lifecycle_view_id="spire-gym-standard-combat-view-v1",
    routings=frozenset({"combat", "combat_selection", "noncombat"}),
    outcomes=frozenset({"victory", "death", "escaped"}),
)
VIEWS = (COMMON_V1, STANDARD_COMBAT_V1)


def view_for_id(view_id: object) -> ViewSpec:
    for view in VIEWS:
        if view_id == view.view_id:
            return view
    raise ContractError("combat view ID is not accepted")


def view_for_fingerprint(fingerprint: object) -> ViewSpec:
    """Return the view an accepted descriptor fingerprint belongs to."""
    if fingerprint == _LEGACY_SYNTHETIC_FINGERPRINT or any(
            fingerprint == contract.common_fingerprint for contract in CONTRACTS):
        return COMMON_V1
    if any(fingerprint == contract.standard_combat_fingerprint for contract in CONTRACTS):
        return STANDARD_COMBAT_V1
    raise ContractError("combat view fingerprint is not accepted")
