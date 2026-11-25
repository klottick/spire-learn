"""Accepted public combat views: identity, tensor groups and lifecycle values.

Dependency-free. Each view is a separate schema; the Standard combat view is
not a superset of the common view.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from spire_learn.contracts import ContractError
from spire_learn.neural.gym_contracts import CONTRACTS_BY_REVISION, STANDARD_COMPACT_SCHEMA_ID


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
    # (tensor, field, value): documented non-padding values an unknown cell may hold.
    unknown_sentinels: frozenset[tuple[str, str, int]] = frozenset()
    # Slot groups whose unoccupied rows are zero values; required: always known empty
    # slots; optional: known empty slots or unknown unused capacity.
    required_known_empty_groups: frozenset[str] = frozenset()
    optional_known_empty_groups: frozenset[str] = frozenset()
    # Value-tensor dtypes the descriptor may declare (masks are always int8).
    value_dtypes: frozenset[str] = frozenset({"int64"})
    # A compact layout of a dense view: the dense view's row capacity per row group and its
    # dispatch width. Learner inputs that depend on capacity (known-coverage means, the
    # sampled dispatch width) use these, as the dense view computes them. Empty: dense.
    dense_view_id: str | None = None
    dense_rows: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}), compare=False)
    dense_action_width: int | None = None

    def accepts_unknown(self, tensor: str, field: str, value: int, padding: int) -> bool:
        return value == padding or (tensor, field, value) in self.unknown_sentinels

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
    required_known_empty_groups=frozenset({"potions"}),
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
    # Descriptor semantics.draw_rows: hidden draw-pile cards use zone_position -1/known 0.
    unknown_sentinels=frozenset({("combat_cards", "zone_position", -1)}),
    # Potion capacity exceeds every character's slots; orbs publish empty slots as known.
    optional_known_empty_groups=frozenset({"potions", "orbs"}),
)
# Gym v1.2.0 compact layout of STANDARD_COMBAT_V1: the same groups, columns, vocabularies,
# row order, flags and dispatch indices in narrower dtypes and fewer rows (a capacity preset).
# Widening each value array to int64 and zero-padding the rows gives the dense observation.
STANDARD_COMBAT_COMPACT_V1 = ViewSpec(
    view_id=STANDARD_COMPACT_SCHEMA_ID,
    profile_id=STANDARD_COMBAT_V1.profile_id,
    fingerprint_domain=b"spire-gym/combat-view/standard/compact/v1\0",
    groups=STANDARD_COMBAT_V1.groups,
    occupied_groups=STANDARD_COMBAT_V1.occupied_groups,
    lifecycle_view_id=STANDARD_COMPACT_SCHEMA_ID,
    routings=STANDARD_COMBAT_V1.routings,
    outcomes=STANDARD_COMBAT_V1.outcomes,
    unknown_sentinels=STANDARD_COMBAT_V1.unknown_sentinels,
    optional_known_empty_groups=STANDARD_COMBAT_V1.optional_known_empty_groups,
    value_dtypes=frozenset({"int16", "int32", "int64"}),
    dense_view_id=STANDARD_COMBAT_V1.view_id,
    # The dense descriptor's row capacities (fingerprint 5d8147e1...; unchanged since Gym 1.0.0).
    dense_rows=MappingProxyType({
        "permanent_cards": 1024, "combat_cards": 1024, "enemies": 12, "powers": 2048, "relics": 256,
        "potions": 16, "orbs": 10, "selection": 1024, "actions": 2048, "known_draw": 1024,
        "stance_choices": 2, "combat_card_offers": 512,
    }),
    dense_action_width=2048,
)
VIEWS = (COMMON_V1, STANDARD_COMBAT_V1, STANDARD_COMBAT_COMPACT_V1)


def view_for_id(view_id: object) -> ViewSpec:
    for view in VIEWS:
        if view_id == view.view_id:
            return view
    raise ContractError("combat view ID is not accepted")


def view_for_fingerprint(fingerprint: object) -> ViewSpec:
    """Return the view an accepted descriptor fingerprint belongs to."""
    contracts = CONTRACTS_BY_REVISION.values()
    if fingerprint == _LEGACY_SYNTHETIC_FINGERPRINT or any(
            fingerprint == contract.common_fingerprint for contract in contracts):
        return COMMON_V1
    if any(fingerprint == contract.standard_combat_fingerprint for contract in contracts):
        return STANDARD_COMBAT_V1
    if any(fingerprint in contract.standard_compact_fingerprints.values() for contract in contracts):
        return STANDARD_COMBAT_COMPACT_V1
    raise ContractError("combat view fingerprint is not accepted")
