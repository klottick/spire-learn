"""Exact public Gym contracts; no environment or optional runtime imports.

Schema names are not compatibility promises. A release selects the complete
descriptor fingerprints and the checkpoint mapping IDs derived from them.
"""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping


@dataclass(frozen=True)
class BuildIdentity:
    """Published ``build_info()`` values a release build must report exactly."""

    revision: str
    source_digest: str
    file_digest: str
    profile: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class GymContract:
    source_revision: str
    runtime_revisions: tuple[str, ...]
    common_fingerprint: str
    run_fingerprint: str
    card_metadata_fingerprint: str | None = None
    build: BuildIdentity | None = None
    standard_combat_fingerprint: str | None = None
    standard_run_fingerprint: str | None = None
    # Gym may change rules without changing descriptor fingerprints; the epoch
    # keeps checkpoint mappings from crossing such a change.
    rules_epoch: str | None = None

    @property
    def raw_mode_mappings(self) -> Mapping[str, str]:
        mappings = {
            "standalone": f"spire-learn-standalone-common-v1@{self.common_fingerprint}",
            "run": f"spire-learn-run-policy-v3-common-v1@{self.run_fingerprint}:{self.common_fingerprint}",
        }
        if self.standard_combat_fingerprint is not None:
            mappings["standard-standalone"] = (
                f"spire-learn-standard-combat-v1@{self.standard_combat_fingerprint}")
        if self.standard_run_fingerprint is not None and self.standard_combat_fingerprint is not None:
            mappings["standard-run"] = (
                f"spire-learn-standard-run-v1@{self.standard_run_fingerprint}"
                f":{self.standard_combat_fingerprint}")
        if self.rules_epoch is not None:
            mappings = {mode: f"{value}+{self.rules_epoch}" for mode, value in mappings.items()}
        return MappingProxyType(mappings)


LEGACY_CONTRACT = GymContract(
    "533dad1c8aec86df624b3742b6b3ad1dbf1d0033",
    ("533dad1c8aec86df624b3742b6b3ad1dbf1d0033", "a86df922a9dc598645820cc6544df36aefb3d2cc"),
    "02009686282c592f971eece3d5853eb54f2a482dde9dae1d0cf380ecdaf8fd78",
    "30905db1b7a520482d3ea8aa2e63ddb7f78b7c081de89d87268151f29cf4c7f8",
)
CURRENT_CONTRACT = GymContract(
    "afb51f5f222b4c4331ed34067c5eebeda606db91",
    ("afb51f5f222b4c4331ed34067c5eebeda606db91",),
    "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64",
    "b60b66c69bce54a6652442e54bc42a2a446d05ec7efa39bb1b11cf4b8c6ff5fb",
)
GYM_V1_CONTRACT = GymContract(
    "1929e18b9f44372e020877bd62f2a5c9dee3e64e",
    ("1929e18b9f44372e020877bd62f2a5c9dee3e64e",),
    "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64",
    "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406",
    BuildIdentity(
        "1929e18b9f44372e020877bd62f2a5c9dee3e64e",
        "d4b3121226cfbb030b89731ed7ea3773f5c2b44988c9ec01c2355d53f2dfa9b2",
        "dce682976f221f5686d16beda7ced03f0221624da0a9b83e8b5969f7300f60ce",
        "release",
        ("spire-python/extension",),
    ),
    standard_combat_fingerprint="5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e",
    standard_run_fingerprint="bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae",
    rules_epoch="gym-1.0.0",
)
CONTRACTS = (LEGACY_CONTRACT, CURRENT_CONTRACT, GYM_V1_CONTRACT)
RUNTIME_REVISIONS = frozenset(revision for item in CONTRACTS for revision in item.runtime_revisions)
COMMON_FINGERPRINTS = frozenset(item.common_fingerprint for item in CONTRACTS)


def contract_for_revision(revision: str) -> GymContract:
    for contract in CONTRACTS:
        if type(revision) is str and revision in contract.runtime_revisions:
            return contract
    raise ValueError("unsupported Gym revision")


def contracts_for_fingerprint(fingerprint: str) -> tuple[GymContract, ...]:
    """Return every contract sharing a common-view fingerprint; may be several."""
    return tuple(
        contract for contract in CONTRACTS
        if type(fingerprint) is str and fingerprint == contract.common_fingerprint
    )


def contract_for_schema(fingerprint: str, revision: str | None) -> GymContract:
    """Select by revision; without one, accept only an unambiguous fingerprint."""
    if revision is not None:
        contract = contract_for_revision(revision)
        if fingerprint != contract.common_fingerprint:
            raise ValueError("common combat view fingerprint does not match the Gym revision")
        return contract
    matches = contracts_for_fingerprint(fingerprint)
    if not matches:
        raise ValueError("unsupported common combat view fingerprint")
    if len(matches) > 1:
        raise ValueError("common combat view fingerprint is shared by several Gym revisions; "
                         "the configured revision is required")
    return matches[0]
