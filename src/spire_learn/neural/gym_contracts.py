"""Exact public Gym contracts; no environment or optional runtime imports.

Schema names are not compatibility promises. A release selects the complete
descriptor fingerprints and the checkpoint mapping IDs derived from them.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType
from collections.abc import Mapping


# Opt-in compact Standard combat view (Gym v1.2.0, standard-run-api.md "M39b"). Each capacity
# preset has its own descriptor fingerprint; a compact layout is its own learner input identity.
STANDARD_COMPACT_SCHEMA_ID = "spire-gym-standard-combat-view-compact-v1"
STANDARD_COMPACT_PRESET = "standard-combat-compact-v1"
STANDARD_COMPACT_PLAY_PRESET = "standard-combat-compact-play-v1"
RUN_STATES_TABLE_V1 = "spire-learn-run-states-v1"
_STANDARD_COMPACT_MAPPING_ID = "spire-learn-standard-combat-compact-v1"


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
    m10_rules_id: str = "spire-gym-native-m10-rules-v1"
    standard_rules_id: str | None = None
    # Opt-in compact Standard combat view (schema
    # ``spire-gym-standard-combat-view-compact-v1``), one fingerprint per capacity preset.
    # They never enter ``raw_mode_mappings`` (dense); a compact config's mapping comes from
    # ``raw_mode_mappings_for_view``.
    standard_compact_combat_fingerprint: str | None = None
    standard_compact_combat_play_fingerprint: str | None = None

    entry_resources_fingerprint: str | None = None
    run_states_table_schemas: tuple[str, ...] = ()
    standard_stream_smoke_tests: tuple[str, ...] = ()
    # Native ``playout_many``/``playout_action`` accept ``tiered_verbs`` (rule
    # ``spire-gym-public-playout-tiered-v1``, Gym v1.2.5). A capability, not a rules change.
    tiered_playouts: bool = False

    @property
    def standard_compact_fingerprints(self) -> Mapping[str, str]:
        """Compact capacity preset -> descriptor fingerprint; empty for dense-only releases."""
        presets = {}
        if self.standard_compact_combat_fingerprint is not None:
            presets[STANDARD_COMPACT_PRESET] = self.standard_compact_combat_fingerprint
        if self.standard_compact_combat_play_fingerprint is not None:
            presets[STANDARD_COMPACT_PLAY_PRESET] = self.standard_compact_combat_play_fingerprint
        return MappingProxyType(presets)

    def raw_mode_mappings_for_view(self, standard_view_fingerprint: str | None) -> Mapping[str, str]:
        """The accepted raw-mode mappings for a Standard combat view fingerprint.

        The dense view (or no Standard view) keeps ``raw_mode_mappings`` unchanged; any other
        fingerprint that is not one of the release's compact presets is refused. A compact
        preset fingerprint maps ``standard-standalone`` only (the Run's combat projection is
        dense only), under its own ID naming the compact schema and the preset's fingerprint,
        so dense and compact checkpoints never load into each other.
        """
        if standard_view_fingerprint is None or standard_view_fingerprint == self.standard_combat_fingerprint:
            return self.raw_mode_mappings
        if standard_view_fingerprint not in self.standard_compact_fingerprints.values():
            raise ValueError("Standard combat view fingerprint is not accepted by this Gym release")
        mapping = f"{_STANDARD_COMPACT_MAPPING_ID}@{standard_view_fingerprint}"
        if self.rules_epoch is not None:
            mapping = f"{mapping}+{self.rules_epoch}"
        return MappingProxyType({"standard-standalone": mapping})

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
    standard_rules_id="spire-gym-native-standard-rules-v3",
)
GYM_8726D467_CONTRACT = GymContract(
    "8726d467ad8579730ddf4fa236b24aeb2206b71a",
    ("8726d467ad8579730ddf4fa236b24aeb2206b71a",),
    "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64",
    "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406",
    BuildIdentity(
        "8726d467ad8579730ddf4fa236b24aeb2206b71a",
        "73f554e2a9a2cb20c23efff0daba83e4ff6a6bcd90e0fbd33b48f8534bce1d7a",
        "98db0a9ac92e05c45424fb5658485ed16ee92a0cfcd49063728c2d9a905a6ccf",
        "release",
        ("spire-python/extension",),
    ),
    standard_combat_fingerprint="5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e",
    standard_run_fingerprint="bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae",
    rules_epoch="gym-1.0.0+8726d467",
    m10_rules_id="spire-gym-native-m10-rules-v2",
    standard_rules_id="spire-gym-native-standard-rules-v5",
)
GYM_V1_1_0_CONTRACT = GymContract(
    "26615ef470e420114f29a208103260b517ee4906",
    ("26615ef470e420114f29a208103260b517ee4906",),
    "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64",
    "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406",
    BuildIdentity(
        "26615ef470e420114f29a208103260b517ee4906",
        "35ea001a49233209132ede98c85ffe9943e0e5366bad8e2876757a4865becaef",
        "9e6eab51ad85b7c42917fefbe518796cb3b4517f0d2282c5f6480d2d0abd8867",
        "release",
        ("spire-python/extension",),
    ),
    standard_combat_fingerprint="5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e",
    standard_run_fingerprint="bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae",
    rules_epoch="gym-1.1.0",
    m10_rules_id="spire-gym-native-m10-rules-v2",
    standard_rules_id="spire-gym-native-standard-rules-v5",
)
GYM_V1_1_1_CONTRACT = GymContract(
    "e3c13836bd1458be8a87b93160f5c350f78fcd80",
    ("e3c13836bd1458be8a87b93160f5c350f78fcd80",),
    "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64",
    "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406",
    BuildIdentity(
        "e3c13836bd1458be8a87b93160f5c350f78fcd80",
        "82b4384a1e51e57de7c0243c9e7a6a053b8aad9fdd4bb525610800418a5e5ad0",
        "9ec87b6bf319ac24b4c055a89b87b65c65f1429798eb3cd59fb9ef77fbe1e1f8",
        "release",
        ("spire-python/extension",),
    ),
    standard_combat_fingerprint="5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e",
    standard_run_fingerprint="bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae",
    rules_epoch="gym-1.1.1",
    m10_rules_id="spire-gym-native-m10-rules-v3",
    standard_rules_id="spire-gym-native-standard-rules-v6",
    run_states_table_schemas=(RUN_STATES_TABLE_V1,),
    standard_stream_smoke_tests=(
        "tests.unit.test_neural_standard_stream_runtime.StandardStreamSmokeTest.test_standard_stream_smoke",
    ),
)
GYM_V1_2_0_CONTRACT = GymContract(
    "4d198064b3b568b6f0d08d6e3937d236fbc48e9b",
    ("4d198064b3b568b6f0d08d6e3937d236fbc48e9b",),
    "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64",
    "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406",
    BuildIdentity(
        "4d198064b3b568b6f0d08d6e3937d236fbc48e9b",
        "4cd56ee7f22dd8303f7d5ff293d88fd65f6071088a00dab73e5c1b176dbe7508",
        "888d789820040c3dc9a795850f75580eb5e31bfb6760c3e1809933ad966561bd",
        "release",
        ("spire-python/extension",),
    ),
    standard_combat_fingerprint="5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e",
    standard_run_fingerprint="bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae",
    rules_epoch="gym-1.2.0",
    m10_rules_id="spire-gym-native-m10-rules-v4",
    standard_rules_id="spire-gym-native-standard-rules-v7",
    standard_compact_combat_fingerprint="49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8",
    standard_compact_combat_play_fingerprint="fbfade07003bd64ad6071243ec3a52e23413c88c5758ca702a515945935439ae",
    entry_resources_fingerprint="4eafca3b0089e15735fb6aef24d594a084030438daa9738a1ac8cf6eab75dc5a",
    run_states_table_schemas=(RUN_STATES_TABLE_V1,),
    standard_stream_smoke_tests=(
        "tests.unit.test_neural_standard_stream_runtime.StandardStreamSmokeTest.test_standard_stream_smoke_gym_1_2_0",
        "tests.unit.test_neural_standard_compact.GatedCompactTests.test_compact_and_dense_give_identical_policy_outputs",
        "tests.unit.test_neural_standard_compact.GatedCompactTests.test_capacity_overflow_is_a_driver_fault",
    ),
)
GYM_V1_2_4_CONTRACT = GymContract(
    "673eb8f220023b0c95401bbf96bec814a359457b",
    ("673eb8f220023b0c95401bbf96bec814a359457b",),
    "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463",
    "28d92d89c437eea7fdc260240585bca466dee1fd74b9b3eb38513d2b34b0be64",
    "effbb810d345e906f6e20b47bd972e35d72ef48141378d54bd28b54afac5f406",
    BuildIdentity(
        "673eb8f220023b0c95401bbf96bec814a359457b",
        "194a901c8bff91a5176eb994f5e1b52c70473c3af74ea0cf1c2672cd679f3757",
        "934fcd6a297196d493e63ea8ae0bf5f0e5f021de877e9ee646cc5e181a611dfb",
        "release",
        ("spire-python/extension",),
    ),
    standard_combat_fingerprint="5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e",
    standard_run_fingerprint="bf8e1d111bbf2171594231b99d8d6b701244249abfbb90a11e86c4b4a405f7ae",
    rules_epoch="gym-1.2.4",
    m10_rules_id="spire-gym-native-m10-rules-v6",
    standard_rules_id="spire-gym-native-standard-rules-v9",
    standard_compact_combat_fingerprint="49cd162afb9710bc6a27ed78eb6ef6dd84af83ae20317e0fffb106b5963ec3e8",
    standard_compact_combat_play_fingerprint="fbfade07003bd64ad6071243ec3a52e23413c88c5758ca702a515945935439ae",
    entry_resources_fingerprint="4eafca3b0089e15735fb6aef24d594a084030438daa9738a1ac8cf6eab75dc5a",
    # Same table schema as v1.2.0; the generation is the table's own recorded revision, so a
    # v1.2.0 table is refused under this contract and the reverse (tables are regenerated).
    run_states_table_schemas=(RUN_STATES_TABLE_V1,),
    standard_stream_smoke_tests=(
        "tests.unit.test_neural_standard_stream_runtime.StandardStreamSmokeTest.test_standard_stream_smoke_gym_1_2_4",
        "tests.unit.test_neural_standard_compact.GatedCompactTests.test_compact_and_dense_give_identical_policy_outputs",
        "tests.unit.test_neural_standard_compact.GatedCompactTests.test_capacity_overflow_is_a_driver_fault",
    ),
)
# A drop-in: v1.2.5 keeps v1.2.4's rules IDs, descriptor fingerprints, replay digests and rules
# epoch, so its mode mappings equal v1.2.4's and checkpoints, tables and recipes cross between the
# two. Only the build identity, the smoke tests and the tiered playout capability differ.
GYM_V1_2_5_CONTRACT = replace(
    GYM_V1_2_4_CONTRACT,
    source_revision="731fca0e0341d80e7cf08a8be7a03be1fa32c3f1",
    runtime_revisions=("731fca0e0341d80e7cf08a8be7a03be1fa32c3f1",),
    build=BuildIdentity(
        "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1",
        "19854fcccf6bd09ceef036297d4a6ecf9d068d9bbc37968a38de07482f2ce117",
        "63533371d420314d41b6307169156e5fe715c41920052f4117d225cf758d6a63",
        "release",
        ("spire-python/extension",),
    ),
    standard_stream_smoke_tests=(
        "tests.unit.test_neural_standard_stream_runtime.StandardStreamSmokeTest.test_standard_stream_smoke_gym_1_2_5",
        "tests.unit.test_neural_standard_compact.GatedCompactTests.test_compact_and_dense_give_identical_policy_outputs",
        "tests.unit.test_neural_standard_compact.GatedCompactTests.test_capacity_overflow_is_a_driver_fault",
    ),
    tiered_playouts=True,
)
CONTRACTS_BY_REVISION: dict[str, GymContract] = {
    revision: contract
    for contract in (
        LEGACY_CONTRACT, CURRENT_CONTRACT, GYM_V1_CONTRACT, GYM_8726D467_CONTRACT,
        GYM_V1_1_0_CONTRACT, GYM_V1_1_1_CONTRACT, GYM_V1_2_0_CONTRACT, GYM_V1_2_4_CONTRACT,
        GYM_V1_2_5_CONTRACT,
    )
    for revision in contract.runtime_revisions
}


def supported_contracts() -> tuple[GymContract, ...]:
    """List the registered contracts once, even when a contract has runtime aliases."""
    return tuple(dict.fromkeys(CONTRACTS_BY_REVISION.values()))


CONTRACTS = supported_contracts()
RUNTIME_REVISIONS = frozenset(CONTRACTS_BY_REVISION)
COMMON_FINGERPRINTS = frozenset(item.common_fingerprint for item in CONTRACTS)


def contract_for_revision(revision: str) -> GymContract:
    if type(revision) is str and revision in CONTRACTS_BY_REVISION:
        return CONTRACTS_BY_REVISION[revision]
    raise ValueError("unsupported Gym revision")


def same_rules_epoch(revision_a: str, revision_b: str) -> bool:
    """True only when both revisions are registered and their contracts share a rules epoch.

    A contract without an epoch (``None``) shares none, not even with itself.
    """
    contracts = [CONTRACTS_BY_REVISION.get(revision) if type(revision) is str else None
                 for revision in (revision_a, revision_b)]
    if None in contracts:
        return False
    epoch = contracts[0].rules_epoch
    return epoch is not None and epoch == contracts[1].rules_epoch


def accepts_recorded_revision(live: str, recorded: str) -> bool:
    """A recorded revision (a table's, a recipe's, a plan's) is usable on the live or configured one.

    The same revision, as before, or two registered revisions in the same rules epoch. Records
    from any other epoch fail closed.
    """
    return (type(live) is str and live == recorded) or same_rules_epoch(live, recorded)


def contract_for_run_states(revision: str, table_schema: str) -> GymContract:
    """Select the public contract for a supported run-states table generation."""
    try:
        contract = contract_for_revision(revision)
    except ValueError as error:
        raise ValueError(f"unsupported Gym revision {revision!r}") from error
    if table_schema not in contract.run_states_table_schemas:
        raise ValueError(f"unsupported Gym revision {revision!r} for run-states schema {table_schema!r}")
    return contract


def contracts_for_fingerprint(fingerprint: str) -> tuple[GymContract, ...]:
    """Return every contract sharing a common-view fingerprint; may be several."""
    return tuple(
        contract for contract in supported_contracts()
        if type(fingerprint) is str and fingerprint == contract.common_fingerprint
    )


def contract_for_schema(
    fingerprint: str, revision: str | None, *, field: str = "common_fingerprint",
) -> GymContract:
    """Select by revision; without one, accept only an unambiguous fingerprint.

    ``field`` names the contract fingerprint the descriptor must equal:
    ``common_fingerprint``, ``standard_combat_fingerprint`` or
    ``standard_compact_fingerprints`` (any compact preset of the release).
    """
    if field not in {"common_fingerprint", "standard_combat_fingerprint", "standard_compact_fingerprints"}:
        raise ValueError("unsupported combat view fingerprint field")

    def accepts(contract: GymContract) -> bool:
        if field == "standard_compact_fingerprints":
            return type(fingerprint) is str and fingerprint in contract.standard_compact_fingerprints.values()
        return type(fingerprint) is str and fingerprint == getattr(contract, field)

    if revision is not None:
        contract = contract_for_revision(revision)
        if not accepts(contract):
            raise ValueError("combat view fingerprint does not match the Gym revision")
        return contract
    matches = tuple(contract for contract in supported_contracts() if accepts(contract))
    if not matches:
        raise ValueError("unsupported combat view fingerprint")
    if len(matches) > 1:
        raise ValueError("combat view fingerprint is shared by several Gym revisions; "
                         "the configured revision is required")
    return matches[0]
