"""Exact public Gym contracts; no environment or optional runtime imports.

Schema names are not compatibility promises. A release selects the complete
descriptor fingerprints and the checkpoint mapping IDs derived from them.
"""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping


@dataclass(frozen=True)
class GymContract:
    source_revision: str
    runtime_revisions: tuple[str, ...]
    common_fingerprint: str
    run_fingerprint: str
    card_metadata_fingerprint: str | None = None

    @property
    def raw_mode_mappings(self) -> Mapping[str, str]:
        return MappingProxyType({
            "standalone": f"spire-learn-standalone-common-v1@{self.common_fingerprint}",
            "run": f"spire-learn-run-policy-v3-common-v1@{self.run_fingerprint}:{self.common_fingerprint}",
        })


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
CONTRACTS = (LEGACY_CONTRACT, CURRENT_CONTRACT)
RUNTIME_REVISIONS = frozenset(revision for item in CONTRACTS for revision in item.runtime_revisions)
COMMON_FINGERPRINTS = frozenset(item.common_fingerprint for item in CONTRACTS)


def contract_for_revision(revision: str) -> GymContract:
    for contract in CONTRACTS:
        if type(revision) is str and revision in contract.runtime_revisions:
            return contract
    raise ValueError("unsupported Gym revision")


def contract_for_fingerprint(fingerprint: str) -> GymContract:
    for contract in CONTRACTS:
        if type(fingerprint) is str and fingerprint == contract.common_fingerprint:
            return contract
    raise ValueError("unsupported common combat view fingerprint")
