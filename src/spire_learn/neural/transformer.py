"""The entity transformer student (S2): one token per occupied public row, references resolved first.

**Tokens.** The plan (``transformer_plan``) is read from the authenticated descriptor, never from a
list of names: every value group with rows (``occupied_key``) other than ``actions`` is a *token
group*, and every single-row group is a *context group*. A batch has one token per occupied row of
every token group, in descriptor group order and then row order, plus one context token at
position 0 built from all context groups and the public known-coverage vector. Padding rows are
never tokens: they are neither embedded nor attended, so their contents cannot reach an output, and
a trimmed batch gives the same result as a full one. There is no positional input; a token knows
its group (a learned group embedding) and its own columns only, so the model is invariant to how
rows are ordered within a group.

**Row embedding.** A row's embedding is its group embedding, plus one learned embedding per
categorical column (an extra index for an unknown value), plus one linear map of its numeric
columns and reference flags. Each numeric column contributes its encoding (zero when unknown)
and its known flag. The ``number_encoding`` is one of:

* ``scaled``: ``sign(x) * log1p(|x|)``;
* ``sinusoidal``: that scaled value, then ``sin`` and ``cos`` of ``x * w_k`` for
  ``SINUSOIDAL_FREQUENCIES`` fixed frequencies ``w_k = 10000 ** (-k / K)``;
* ``bucket``: a learned ``BUCKET_FEATURES``-wide embedding of a signed magnitude bucket
  (``bucket_index``: exact below 16, one per power of two above).

**References are resolved before the trunk.** A reference column never enters as a raw row index.
It contributes two flags (the index is known; it resolves to a token), and a learned projection
(one per reference column) of the referenced token's *pre-trunk* embedding is added to the row's
token. The pre-trunk embedding is the referenced row's own embedding before any reference is
resolved, so there is no ordering between references. The projection is masked (zero) when the
reference is -1 or unknown, its selector names no token group, or the row is out of range or
unoccupied. A reference to a single-row group (``player``, ``combat_state``) resolves to the
context token. With the compact descriptor the resolved columns are exactly
``powers.owner_row``, ``selection.source_row``, ``known_draw.combat_card_row``,
``combat_cards.permanent_row`` and ``combat_cards.stasis_enemy_row``.

**Trunk.** ``layers`` pre-norm blocks (``x + attention(LN(x))``, then ``x + FFN(LN(x))``, GELU,
``FFN_RATIO * width`` hidden, no dropout) over the tokens, with padding masked from every key,
then a final layer norm.

**Action scoring.** Only legal actions are scored. Each legal action's query is its own row
embedding (built like any row) plus learned projections of the post-trunk tokens its reference
columns name (``source_row``, ``target_row`` and ``stance_choice_ref``, masked like any reference).
One pre-norm cross-attention block lets every query attend over the trunk tokens (queries never
attend to each other), and a layer norm and a linear map give its logit. Illegal entries of the
logits are exactly 0 and are never computed from anything.

**Value heads.** Three linear heads over the mean of the post-trunk tokens: ``values`` is the
outcome head, ``aux_values`` the HP-kept and realised-v2 heads, as for the FF student.

**Identity.** ``TRANSFORMER_MODEL_ID``/``TRANSFORMER_FEATURES_ID`` with the shared dispatch action
id, under the distillation objective only. The parameter count is computed from the plan before
torch is imported; a configuration over ``TRANSFORMER_PARAMETER_CAP`` (16,000,000) is refused, and
the built module must have exactly the computed count.

**Card metadata (S3).** ``build_transformer_cardmeta_policy`` is the same network with the Gym's
static public card metadata (``spire-lab-card-metadata-v1``, a committed snapshot pinned by its
fingerprint) added to the card tokens of ``combat_cards``, ``permanent_cards`` and
``combat_card_offers``. It is public, static, run-independent game data: no seed, RNG or hidden
order. A card token's code (its ``card`` vocabulary index) joins to the metadata row of that id;
``padding``, an unknown code and an unknown card give an all-zero vector. The vector is one linear map
(per group) of:

* the card's own fields: ``has_row``, one-hot ``color``, ``rarity`` and ``type``, multi-hot ``tags``;
* the face at the card's upgrade level, ``faces[min(max(upgrades, 0), len(faces) - 1)]`` (offers have
  no upgrades: level 0): ``has_face`` (0 for a card with no faces), multi-hot ``keywords``, one-hot
  cost kind (``int``, ``x``, ``unplayable``, ``unresolved``) and the cost number (``scaled``, 0 for a
  non-integer cost), one-hot target (``enemy``, ``untargeted``, ``unresolved``), and for each of
  ``base_damage``, ``base_block`` and ``magic_number`` a presence bit and the ``scaled`` value
  (``sign(x) * log1p(|x|)``, 0 when null).

The live card row (cost, upgrades, damage and so on) is still embedded as in S2; the metadata is
the printed face, never a replacement for an observation. The tables are built once from the
snapshot into non-persistent buffers, so the state dict holds parameters only; the snapshot's
fingerprint is a field of ``TransformerCardMetaConfig`` and so part of the config hash.

Importing this module imports neither torch nor numpy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import functools
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.neural.config import (
    CARD_METADATA_FINGERPRINT, DISTILL_OBJECTIVE_ID, LEARNER_IDENTITY_IDS, TRANSFORMER_CARDMETA_FEATURES_ID,
    TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_FEATURES_ID, TRANSFORMER_MODEL_ID, TRANSFORMER_PARAMETER_CAP,
    TransformerCardMetaConfig, TransformerConfig, model_config_hash,
)
from spire_learn.neural.schema import CombatSchema

ACTION_GROUP = "actions"
FFN_RATIO = 4
SINUSOIDAL_FREQUENCIES = 8
SINUSOIDAL_BASE = 10000.0
BUCKET_FEATURES = 16
BUCKET_EXACT = 16  # |x| below this has its own bucket
BUCKET_MAX_MAGNITUDE = (BUCKET_EXACT - 4) + 31  # 16 -> 16, ..., 2**31 -> 43
BUCKET_COUNT = 2 * BUCKET_MAX_MAGNITUDE + 1
EMBEDDING_STD = 0.02
_ACTION_ID = LEARNER_IDENTITY_IDS[2]
_STANDARD_VIEW_ID = "spire-gym-standard-combat-view-v1"
_ENCODING_FEATURES = {"scaled": 1, "sinusoidal": 1 + 2 * SINUSOIDAL_FREQUENCIES, "bucket": BUCKET_FEATURES}


# --- the descriptor plan (stdlib) -----------------------------------------------------------

@dataclass(frozen=True)
class ReferencePlan:
    """One reference column: its selector (None for a literal target) and resolvable targets."""

    group: str
    field: str
    field_index: int
    selector_index: int | None
    # (selector vocabulary code, target group); the code is None for a literal target.
    choices: tuple[tuple[int | None, str], ...]


@dataclass(frozen=True)
class GroupPlan:
    """One value group's columns by role, in descriptor column order."""

    name: str
    index: int  # position among the plan's value groups
    categorical: tuple[tuple[int, int], ...]  # (field index, vocabulary size)
    numeric: tuple[int, ...]
    references: tuple[ReferencePlan, ...]

    def input_width(self, number_encoding: str) -> int:
        """Width of the numeric-and-flags input of this group's linear map (0: no map)."""
        return len(self.numeric) * (_ENCODING_FEATURES[number_encoding] + 1) + 2 * len(self.references)


@dataclass(frozen=True)
class TransformerPlan:
    schema_fingerprint: str
    groups: Mapping[str, GroupPlan]  # every value group, descriptor order
    token_groups: tuple[str, ...]  # one token per occupied row
    context_groups: tuple[str, ...]  # folded into the context token
    coverage_width: int  # the public known-coverage vector (one entry per column of every group)

    @property
    def resolved_references(self) -> tuple[ReferencePlan, ...]:
        """References resolved before the trunk (every non-action reference column)."""
        return tuple(ref for name in (*self.token_groups, *self.context_groups)
                     for ref in self.groups[name].references)

    @property
    def action_references(self) -> tuple[ReferencePlan, ...]:
        """Action references gathered from the post-trunk tokens."""
        return self.groups[ACTION_GROUP].references


def _reference_plan(schema: CombatSchema, group: str, field_index: int, targets: frozenset[str]) -> ReferencePlan:
    spec = schema.tensors[group]
    column = spec.columns[field_index]
    selector_index = next((index for index, item in enumerate(spec.columns) if item.field == column.reference), None)
    if selector_index is None:
        if column.reference not in targets:
            raise ContractError(f"{group}.{column.field} targets {column.reference}, which is no token or context group")
        choices: tuple[tuple[int | None, str], ...] = ((None, column.reference),)
    else:
        selector = spec.columns[selector_index]
        if selector.vocabulary is None or selector.vocabulary not in schema.vocabularies:
            raise ContractError(f"{group}.{column.field} needs a categorical selector")
        choices = tuple((code, name) for code, name in enumerate(schema.vocabularies[selector.vocabulary])
                        if name in targets)
        if not choices:
            raise ContractError(f"{group}.{column.field} selects no token or context group")
    return ReferencePlan(group, column.field, field_index, selector_index, choices)


def transformer_plan(schema: CombatSchema) -> TransformerPlan:
    """The token and reference plan of ``schema``, enumerated from its descriptor (stdlib only)."""
    if not isinstance(schema, CombatSchema):
        raise ContractError("the entity transformer requires an authenticated CombatSchema")
    value_groups = tuple(name for name, spec in schema.tensors.items() if spec.columns)
    actions = schema.tensors.get(ACTION_GROUP)
    if actions is None or not actions.columns or actions.occupied_key is None or "action_mask" not in schema.tensors:
        raise ContractError("the schema has no public actions row group and action mask")
    token_groups = tuple(name for name in value_groups
                         if name != ACTION_GROUP and schema.tensors[name].occupied_key is not None)
    context_groups = tuple(name for name in value_groups if schema.tensors[name].occupied_key is None)
    if not context_groups:
        raise ContractError("the schema has no single-row group for the context token")
    targets = frozenset((*token_groups, *context_groups))
    groups: dict[str, GroupPlan] = {}
    for index, name in enumerate(value_groups):
        categorical, numeric, references = [], [], []
        for field_index, column in enumerate(schema.tensors[name].columns):
            if column.reference is not None:
                references.append(_reference_plan(schema, name, field_index, targets))
            elif column.vocabulary is not None:
                vocabulary = schema.vocabularies.get(column.vocabulary)
                if vocabulary is None:
                    raise ContractError(f"{name}.{column.field} names a missing vocabulary")
                categorical.append((field_index, len(vocabulary)))
            else:
                numeric.append(field_index)
        groups[name] = GroupPlan(name, index, tuple(categorical), tuple(numeric), tuple(references))
    coverage = sum(len(schema.tensors[name].columns) for name in value_groups)
    return TransformerPlan(schema.schema_fingerprint, MappingProxyType(groups), token_groups, context_groups, coverage)


def _linear(inputs: int, outputs: int) -> int:
    return inputs * outputs + outputs


def _block_parameters(width: int) -> int:
    # Two layer norms, attention (in-projection for query/key/value, out-projection), the FFN.
    return 2 * (2 * width) + _linear(width, 3 * width) + _linear(width, width) \
        + _linear(width, FFN_RATIO * width) + _linear(FFN_RATIO * width, width)


def transformer_parameter_count(plan: TransformerPlan | CombatSchema, config: TransformerConfig) -> int:
    """The exact parameter count of the entity transformer for ``config`` (stdlib only)."""
    if isinstance(plan, CombatSchema):
        plan = transformer_plan(plan)
    if not isinstance(config, TransformerConfig):
        raise ContractError("the entity transformer requires a TransformerConfig")
    width = config.width
    count = len(plan.groups) * width  # group embeddings
    for group in plan.groups.values():
        count += sum((size + 1) * width for _field, size in group.categorical)
        if config.number_encoding == "bucket":
            count += len(group.numeric) * BUCKET_COUNT * BUCKET_FEATURES
        inputs = group.input_width(config.number_encoding)
        if inputs:
            count += _linear(inputs, width)
    count += _linear(plan.coverage_width, width)
    count += (len(plan.resolved_references) + len(plan.action_references)) * _linear(width, width)
    count += config.layers * _block_parameters(width) + 2 * width  # the trunk and its final norm
    count += _block_parameters(width) + 2 * width + _linear(width, 1)  # action cross block, norm, score
    count += _linear(width, config.value_heads)
    return count


def bucket_index(value: int) -> int:
    """The ``bucket`` encoding's bucket of one integer (the reference for the tensor version)."""
    magnitude = abs(int(value))
    if magnitude >= BUCKET_EXACT:
        magnitude = (BUCKET_EXACT - 4) + int(math.floor(math.log2(magnitude)))
    magnitude = min(magnitude, BUCKET_MAX_MAGNITUDE)
    sign = (value > 0) - (value < 0)
    return BUCKET_MAX_MAGNITUDE + sign * magnitude



# --- card metadata (stdlib, S3) --------------------------------------------------------------

CARD_METADATA_SCHEMA_ID = "spire-lab-card-metadata-v1"
CARD_METADATA_SNAPSHOT = (Path(__file__).resolve().parents[3] / "configs" / "neural" / "standard"
                          / "card-metadata-gym-1.2.4.json")
CARD_METADATA_GROUPS = ("combat_cards", "permanent_cards", "combat_card_offers")
CARD_VOCABULARY = "card"
CARD_COLORS = ("red", "green", "blue", "purple", "colorless", "status", "curse", "shared")
CARD_RARITIES = ("basic", "common", "uncommon", "rare", "special", "status", "curse")
CARD_TYPES = ("attack", "skill", "power", "status", "curse")
CARD_TAGS = ("strike", "starter_strike", "starter_defend", "healing")
CARD_KEYWORDS = ("ethereal", "exhaust", "innate", "retain")
CARD_COST_KINDS = ("int", "x", "unplayable", "unresolved")
CARD_TARGETS = ("enemy", "untargeted", "unresolved")
CARD_NULLABLE_FIELDS = ("base_damage", "base_block", "magic_number")
# [has_row, color, rarity, type, tags]
CARD_STATIC_FEATURES = 1 + len(CARD_COLORS) + len(CARD_RARITIES) + len(CARD_TYPES) + len(CARD_TAGS)
# [has_face, keywords, cost kind, cost number, target, (presence, value) per nullable field]
CARD_FACE_FEATURES = 1 + len(CARD_KEYWORDS) + len(CARD_COST_KINDS) + 1 + len(CARD_TARGETS) + 2 * len(CARD_NULLABLE_FIELDS)
CARD_METADATA_FEATURES = CARD_STATIC_FEATURES + CARD_FACE_FEATURES


def load_card_metadata(path: Path | str | None = None) -> dict[str, Any]:
    """The committed card-metadata snapshot, refused unless it is the pinned revision.

    The default is ``CARD_METADATA_SNAPSHOT`` (``card_metadata_snapshot_entry.py`` writes it in the
    runtime container). A missing file, another schema or a fingerprint other than
    ``CARD_METADATA_FINGERPRINT`` is a ``ContractError``.
    """
    source = Path(CARD_METADATA_SNAPSHOT if path is None else path)
    try:
        document = json.loads(source.read_text())
    except OSError as error:
        raise ContractError(f"the card metadata snapshot {source} cannot be read: {error}") from error
    except ValueError as error:
        raise ContractError(f"the card metadata snapshot {source} is not JSON: {error}") from error
    if not isinstance(document, dict) or document.get("schema_id") != CARD_METADATA_SCHEMA_ID:
        raise ContractError(f"the card metadata snapshot is not {CARD_METADATA_SCHEMA_ID}")
    if document.get("fingerprint") != CARD_METADATA_FINGERPRINT:
        raise ContractError("the card metadata snapshot's fingerprint is not the pinned "
                            f"{CARD_METADATA_FINGERPRINT}: {document.get('fingerprint')!r}")
    if not isinstance(document.get("cards"), list):
        raise ContractError("the card metadata snapshot has no cards list")
    return document


def _one_hot(value: object, choices: tuple[str, ...], what: str) -> list[float]:
    if not isinstance(value, str) or value not in choices:
        raise ContractError(f"card metadata {what} {value!r} is not one of {', '.join(choices)}")
    return [1.0 if value == choice else 0.0 for choice in choices]


def _multi_hot(values: object, choices: tuple[str, ...], what: str) -> list[float]:
    if not isinstance(values, list) or any(not isinstance(item, str) or item not in choices for item in values):
        raise ContractError(f"card metadata {what} {values!r} must be a list drawn from {', '.join(choices)}")
    return [1.0 if choice in values else 0.0 for choice in choices]


def _scaled(value: int) -> float:
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def _integer(value: object, what: str) -> int:
    if type(value) is not int:
        raise ContractError(f"card metadata {what} {value!r} must be an integer")
    return value


def card_static_features(row: Mapping[str, Any]) -> tuple[float, ...]:
    """A card's face-independent features (``CARD_STATIC_FEATURES`` wide)."""
    return tuple([1.0, *_one_hot(row.get("color"), CARD_COLORS, "color"),
                  *_one_hot(row.get("rarity"), CARD_RARITIES, "rarity"),
                  *_one_hot(row.get("type"), CARD_TYPES, "type"),
                  *_multi_hot(row.get("tags"), CARD_TAGS, "tags")])


def card_face_features(face: Mapping[str, Any] | None) -> tuple[float, ...]:
    """One face's features (``CARD_FACE_FEATURES`` wide); all zero, ``has_face`` 0, for no face."""
    if face is None:
        return (0.0,) * CARD_FACE_FEATURES
    cost = face.get("cost")
    if type(cost) is int:
        kind, number = "int", _scaled(cost)
    elif cost in ("x", "unplayable", "unresolved"):
        kind, number = cost, 0.0
    else:
        raise ContractError(f"card metadata cost {cost!r} is not an integer, x, unplayable or unresolved")
    nullable: list[float] = []
    for name in CARD_NULLABLE_FIELDS:
        value = face.get(name)
        nullable += [0.0, 0.0] if value is None else [1.0, _scaled(_integer(value, name))]
    return tuple([1.0, *_multi_hot(face.get("keywords"), CARD_KEYWORDS, "keywords"),
                  *_one_hot(kind, CARD_COST_KINDS, "cost kind"), number,
                  *_one_hot(face.get("target"), CARD_TARGETS, "target"), *nullable])


@dataclass(frozen=True)
class CardMetadataTables:
    """Per card code (an index of the ``card`` vocabulary), the metadata features at each face level.

    ``static[code]`` is ``CARD_STATIC_FEATURES`` wide; ``faces[level][code]`` is ``CARD_FACE_FEATURES``
    wide. ``levels`` is the most faces any card has; a card with fewer faces uses its last face at a
    higher level (``faces[min(level, len(faces) - 1)]``), and a card with none has ``has_face`` 0.
    A code without a metadata row (``padding``, an id the snapshot lacks) is all zero.
    """

    fingerprint: str
    static: tuple[tuple[float, ...], ...]
    faces: tuple[tuple[tuple[float, ...], ...], ...]
    levels: int
    rows: int  # codes with a metadata row
    missing: tuple[str, ...]  # vocabulary ids without a row (``padding`` first, when present)
    unused: tuple[str, ...]  # metadata ids outside the vocabulary


def card_metadata_tables(metadata: Mapping[str, Any], vocabulary: Sequence[str]) -> CardMetadataTables:
    """Join the metadata rows to ``vocabulary`` (the schema's ``card`` vocabulary, id by code)."""
    if metadata.get("fingerprint") != CARD_METADATA_FINGERPRINT:
        raise ContractError(f"card metadata fingerprint is not the pinned {CARD_METADATA_FINGERPRINT}")
    if metadata.get("schema_id") != CARD_METADATA_SCHEMA_ID:
        raise ContractError(f"card metadata is not {CARD_METADATA_SCHEMA_ID}")
    cards = metadata.get("cards")
    if not isinstance(cards, list):
        raise ContractError("card metadata has no cards list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in cards:
        card_id = row.get("id") if isinstance(row, Mapping) else None
        if not isinstance(card_id, str) or card_id in by_id:
            raise ContractError(f"card metadata has a missing or repeated card id {card_id!r}")
        row_faces = row.get("faces")
        if not isinstance(row_faces, list):
            raise ContractError(f"card metadata {card_id} faces must be a list")
        for position, face in enumerate(row_faces):
            if not isinstance(face, Mapping) or face.get("upgrades") != position or type(face.get("upgrades")) is not int:
                raise ContractError(f"card metadata {card_id} face {position} must have upgrades == {position}")
        by_id[card_id] = row
    levels = max([len(row["faces"]) for row in by_id.values()] + [1])
    zero_static = (0.0,) * CARD_STATIC_FEATURES
    zero_face = card_face_features(None)
    static, faces = [], [[] for _ in range(levels)]
    missing = []
    for card_id in vocabulary:
        row = by_id.get(card_id)
        if row is None:
            missing.append(card_id)
            static.append(zero_static)
            for level in range(levels):
                faces[level].append(zero_face)
            continue
        static.append(card_static_features(row))
        row_faces = row["faces"]
        for level in range(levels):
            face = row_faces[min(level, len(row_faces) - 1)] if row_faces else None
            faces[level].append(card_face_features(face))
    known = set(vocabulary)
    return CardMetadataTables(
        fingerprint=metadata["fingerprint"], static=tuple(static),
        faces=tuple(tuple(level) for level in faces), levels=levels, rows=len(vocabulary) - len(missing),
        missing=tuple(missing), unused=tuple(sorted(set(by_id) - known)))


@dataclass(frozen=True)
class CardMetadataGroup:
    """One card-token group: its ``card`` column and its ``upgrades`` column (None: level 0 always)."""

    group: str
    card_field_index: int
    upgrades_field_index: int | None


def card_metadata_groups(schema: CombatSchema) -> tuple[CardMetadataGroup, ...]:
    """The groups that carry card metadata, read from the descriptor (stdlib only)."""
    if not isinstance(schema, CombatSchema):
        raise ContractError("card metadata requires an authenticated CombatSchema")
    if CARD_VOCABULARY not in schema.vocabularies:
        raise ContractError("the schema has no card vocabulary")
    groups = []
    for name in CARD_METADATA_GROUPS:
        spec = schema.tensors.get(name)
        if spec is None or not spec.columns or spec.occupied_key is None:
            raise ContractError(f"the schema has no {name} row group for card metadata")
        cards = [index for index, column in enumerate(spec.columns)
                 if column.vocabulary == CARD_VOCABULARY and column.reference is None]
        if len(cards) != 1:
            raise ContractError(f"{name} must have exactly one card column; it has {len(cards)}")
        upgrades = [index for index, column in enumerate(spec.columns)
                    if column.field == "upgrades" and column.vocabulary is None and column.reference is None]
        groups.append(CardMetadataGroup(name, cards[0], upgrades[0] if upgrades else None))
    return tuple(groups)


def transformer_cardmeta_parameter_count(plan: TransformerPlan | CombatSchema,
                                         config: TransformerCardMetaConfig) -> int:
    """The exact parameter count of S3: S2's plus one metadata projection per card group (stdlib only)."""
    if not isinstance(config, TransformerCardMetaConfig):
        raise ContractError("the card-metadata transformer requires a TransformerCardMetaConfig")
    return (transformer_parameter_count(plan, config.transformer())
            + len(CARD_METADATA_GROUPS) * _linear(CARD_METADATA_FEATURES, config.width))

# --- the policy (torch, imported lazily) ------------------------------------------------------

@dataclass
class TokenLayout:
    """Where each occupied row sits in the packed token sequence of one batch.

    ``positions[group]`` is ``[batch, rows]`` (meaningful where the row is occupied); ``rows[group]``
    the ``(batch index, row index)`` of every occupied row, in token order; ``valid`` ``[batch, T]``
    marks real tokens; ``token_group`` is 0 for the context token, ``k + 1`` for token group ``k``
    and -1 for padding.
    """

    positions: Mapping[str, Any]
    rows: Mapping[str, tuple[Any, Any]]
    flat_positions: Mapping[str, Any]
    valid: Any
    token_group: Any
    counts: Any


@functools.lru_cache(maxsize=1)
def _policy_class():
    import torch
    from torch import nn
    from torch.nn import functional

    from spire_learn.neural.features import EncodedBatch, encode_frames
    from spire_learn.neural.model import PolicyOutput

    class PreNormBlock(nn.Module):
        """``x + attention(LN(x), keys)``, then ``x + FFN(LN(x))``; self-attention when ``keys`` is None."""

        def __init__(self, width: int, heads: int) -> None:
            super().__init__()
            self.attention_norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
            self.feedforward_norm = nn.LayerNorm(width)
            self.feedforward_in = nn.Linear(width, FFN_RATIO * width)
            self.feedforward_out = nn.Linear(FFN_RATIO * width, width)

        def forward(self, x, keys, key_padding):
            normed = self.attention_norm(x)
            source = normed if keys is None else keys
            attended, _ = self.attention(normed, source, source, key_padding_mask=key_padding, need_weights=False)
            x = x + attended
            return x + self.feedforward_out(functional.gelu(self.feedforward_in(self.feedforward_norm(x))))

    class EntityTransformerPolicy(nn.Module):
        def __init__(self, plan: TransformerPlan, schema: CombatSchema, config: TransformerConfig, *,
                     model_id: str, features_id: str, objective_id: str, trim_padding: bool) -> None:
            super().__init__()
            self._plan = plan
            self._schema = schema
            self._config = config
            self._width = config.width
            self._encoding = config.number_encoding
            self._trim_padding = trim_padding
            self._value_heads = config.value_heads
            self._model_config_hash = model_config_hash(config)
            self._behavior_policy_version: int = 0
            self.model_id = model_id
            self.features_id = features_id
            self.action_id = _ACTION_ID
            self.objective_id = objective_id
            width = config.width

            self.category_embeddings = nn.ModuleDict()
            self.bucket_embeddings = nn.ModuleDict()
            self.group_inputs = nn.ModuleDict()
            for group in plan.groups.values():
                for field_index, size in group.categorical:
                    embedding = nn.Embedding(size + 1, width)  # the last index is "unknown"
                    nn.init.normal_(embedding.weight, std=EMBEDDING_STD)
                    self.category_embeddings[_field_key(group, field_index)] = embedding
                if self._encoding == "bucket":
                    for field_index in group.numeric:
                        self.bucket_embeddings[_field_key(group, field_index)] = nn.Embedding(
                            BUCKET_COUNT, BUCKET_FEATURES)
                inputs = group.input_width(self._encoding)
                if inputs:
                    self.group_inputs[_group_key(group)] = nn.Linear(inputs, width)
            self.group_embeddings = nn.Parameter(torch.empty(len(plan.groups), width))
            nn.init.normal_(self.group_embeddings, std=EMBEDDING_STD)
            self.coverage = nn.Linear(plan.coverage_width, width)
            self.reference_projections = nn.ModuleDict({
                _field_key(plan.groups[ref.group], ref.field_index): nn.Linear(width, width)
                for ref in (*plan.resolved_references, *plan.action_references)})
            self.blocks = nn.ModuleList(PreNormBlock(width, config.heads) for _ in range(config.layers))
            self.trunk_norm = nn.LayerNorm(width)
            self.action_block = PreNormBlock(width, config.heads)
            self.action_norm = nn.LayerNorm(width)
            self.action_score = nn.Linear(width, 1)
            self.value_heads = nn.Linear(width, config.value_heads)
            frequencies = [SINUSOIDAL_BASE ** (-index / SINUSOIDAL_FREQUENCIES)
                           for index in range(SINUSOIDAL_FREQUENCIES)]
            self.register_buffer("_frequencies", torch.tensor(frequencies, dtype=torch.float64), persistent=False)

        # --- layout ------------------------------------------------------------------------

        def tokenise(self, batch: EncodedBatch) -> TokenLayout:
            """The packed token layout: context at 0, then each token group's occupied rows in row order."""
            plan = self._plan
            device = batch.action_mask.device
            size = batch.action_mask.shape[0]
            offset = torch.ones(size, dtype=torch.int64, device=device)
            positions, rows, flat = {}, {}, {}
            for name in plan.token_groups:
                occupied = batch.occupied[name].to(dtype=torch.bool)
                positions[name] = offset.unsqueeze(1) + occupied.to(torch.int64).cumsum(dim=1) - 1
                batch_index, row_index = occupied.nonzero(as_tuple=True)
                rows[name] = (batch_index, row_index)
                flat[name] = positions[name][batch_index, row_index]
                offset = offset + occupied.to(torch.int64).sum(dim=1)
            length = int(offset.max().item())
            valid = torch.arange(length, device=device).unsqueeze(0) < offset.unsqueeze(1)
            token_group = torch.full((size, length), -1, dtype=torch.int64, device=device)
            token_group[:, 0] = 0
            for index, name in enumerate(plan.token_groups):
                batch_index, _ = rows[name]
                token_group[batch_index, flat[name]] = index + 1
            return TokenLayout(positions, rows, flat, valid, token_group, offset)

        def _pack(self, layout: TokenLayout, context, embedded: Mapping[str, Any]):
            size = context.shape[0]
            batch_index = [torch.arange(size, device=context.device)]
            position = [torch.zeros(size, dtype=torch.int64, device=context.device)]
            pieces = [context]
            for name in self._plan.token_groups:
                batch_index.append(layout.rows[name][0])
                position.append(layout.flat_positions[name])
                pieces.append(embedded[name])
            packed = context.new_zeros((size, layout.valid.shape[1], self._width))
            return packed.index_put((torch.cat(batch_index), torch.cat(position)), torch.cat(pieces))

        # --- rows ----------------------------------------------------------------------------

        def _number(self, group: GroupPlan, field_index: int, values):
            if self._encoding == "bucket":
                return self.bucket_embeddings[_field_key(group, field_index)](_bucket_tensor(torch, values))
            if self._encoding == "scaled":
                as_float = values.to(dtype=torch.float32)
                return (torch.sign(as_float) * torch.log1p(torch.abs(as_float))).unsqueeze(-1)
            as_double = values.to(dtype=torch.float64)
            scaled = torch.sign(as_double) * torch.log1p(torch.abs(as_double))
            angles = as_double.unsqueeze(-1) * self._frequencies
            return torch.cat((scaled.unsqueeze(-1), torch.sin(angles), torch.cos(angles)), dim=-1).to(torch.float32)

        def _resolve(self, batch, layout: TokenLayout, group: GroupPlan, values, known, batch_index):
            """Per reference column: (token position, resolves to a token, index known), per row."""
            result = {}
            for ref in group.references:
                raw = values[:, ref.field_index]
                ref_known = known[:, ref.field_index]
                position = torch.zeros_like(raw)
                resolved = torch.zeros_like(ref_known)
                for code, target in ref.choices:
                    matches = ref_known
                    if code is not None:
                        matches = matches & known[:, ref.selector_index] & values[:, ref.selector_index].eq(code)
                    if target in self._plan.context_groups:
                        hit = matches & raw.eq(0)
                        target_position = torch.zeros_like(raw)
                    else:
                        occupied = batch.occupied[target].to(dtype=torch.bool)
                        capacity = occupied.shape[1]
                        safe = raw.clamp(min=0, max=capacity - 1)
                        hit = matches & (raw >= 0) & (raw < capacity) & occupied[batch_index, safe]
                        target_position = layout.positions[target][batch_index, safe]
                    position = torch.where(hit, target_position, position)
                    resolved = resolved | hit
                result[ref.field_index] = (position, resolved, ref_known)
            return result

        def _embed(self, group: GroupPlan, values, known, references):
            """Row embeddings before any reference is resolved: ``[rows, width]``."""
            rows = values.shape[0]
            out = self.group_embeddings[group.index].expand(rows, self._width)
            for field_index, size in group.categorical:
                code = torch.where(known[:, field_index], values[:, field_index],
                                   torch.full_like(values[:, field_index], size))
                out = out + self.category_embeddings[_field_key(group, field_index)](code)
            pieces = []
            for field_index in group.numeric:
                flag = known[:, field_index].unsqueeze(-1)
                encoded = self._number(group, field_index, values[:, field_index])
                pieces.append(torch.where(flag, encoded, torch.zeros_like(encoded)))
                pieces.append(flag.to(dtype=torch.float32))
            for ref in group.references:
                _position, resolved, ref_known = references[ref.field_index]
                pieces.append(ref_known.unsqueeze(-1).to(dtype=torch.float32))
                pieces.append(resolved.unsqueeze(-1).to(dtype=torch.float32))
            if pieces:
                out = out + self.group_inputs[_group_key(group)](torch.cat(pieces, dim=-1))
            return out

        def _referenced(self, group: GroupPlan, references, batch_index, tokens):
            """The sum of each reference column's projection of the token it names (zero if none)."""
            total = None
            for ref in group.references:
                position, resolved, _known = references[ref.field_index]
                projected = self.reference_projections[_field_key(group, ref.field_index)](
                    tokens[batch_index, position])
                term = torch.where(resolved.unsqueeze(-1), projected, torch.zeros_like(projected))
                total = term if total is None else total + term
            return total

        def _pre_trunk(self, batch, layout: TokenLayout):
            plan = self._plan
            size = batch.action_mask.shape[0]
            everyone = torch.arange(size, device=batch.action_mask.device)
            rows: dict[str, tuple[Any, Any, Any]] = {}
            for name in plan.token_groups:
                batch_index, row_index = layout.rows[name]
                rows[name] = (batch.values[name][batch_index, row_index],
                              batch.known[name][batch_index, row_index].to(dtype=torch.bool), batch_index)
            for name in plan.context_groups:
                rows[name] = (batch.values[name][:, 0], batch.known[name][:, 0].to(dtype=torch.bool), everyone)
            references = {name: self._resolve(batch, layout, plan.groups[name], *rows[name]) for name in rows}
            base = {name: self._embed(plan.groups[name], rows[name][0], rows[name][1], references[name])
                    for name in rows}
            coverage = torch.cat([batch.known_coverage[name].to(dtype=torch.float32) for name in plan.groups], dim=-1)
            context = self.coverage(coverage)
            for name in plan.context_groups:
                context = context + base[name]
            packed = self._pack(layout, context, base)
            # Every reference reads the pre-trunk (unresolved) embedding of the token it names.
            resolved = {}
            for name in rows:
                extra = self._referenced(plan.groups[name], references[name], rows[name][2], packed)
                resolved[name] = base[name] if extra is None else base[name] + extra
            for name in plan.context_groups:
                if plan.groups[name].references:
                    context = context + (resolved[name] - base[name])
            return self._pack(layout, context, resolved)

        # --- actions ----------------------------------------------------------------------

        def _score_actions(self, batch, layout: TokenLayout, hidden, padding):
            mask = batch.action_mask.to(dtype=torch.bool)
            logits = hidden.new_zeros(mask.shape)
            batch_index, action_index = mask.nonzero(as_tuple=True)
            if batch_index.numel() == 0:
                return logits
            group = self._plan.groups[ACTION_GROUP]
            occupied = batch.occupied[ACTION_GROUP].to(dtype=torch.bool)
            capacity = occupied.shape[1]
            unrowed = (action_index >= capacity) | ~occupied[batch_index, action_index.clamp(max=capacity - 1)]
            if unrowed.any().item():  # one host sync for both conditions
                raise ContractError("a legal action has no occupied action row")
            values = batch.values[ACTION_GROUP][batch_index, action_index]
            known = batch.known[ACTION_GROUP][batch_index, action_index].to(dtype=torch.bool)
            references = self._resolve(batch, layout, group, values, known, batch_index)
            query = self._embed(group, values, known, references)
            extra = self._referenced(group, references, batch_index, hidden)
            if extra is not None:
                query = query + extra
            rank = mask.to(torch.int64).cumsum(dim=1)[batch_index, action_index] - 1
            queries = query.new_zeros((mask.shape[0], int(mask.sum(dim=1).max().item()), self._width))
            queries = queries.index_put((batch_index, rank), query)
            scored = self.action_block(queries, hidden, padding)
            scores = self.action_score(self.action_norm(scored)).squeeze(-1)
            return logits.index_put((batch_index, action_index), scores[batch_index, rank])

        # --- the policy interface -----------------------------------------------------------

        def forward(self, batch: EncodedBatch) -> PolicyOutput:
            if not isinstance(batch, EncodedBatch):
                raise ContractError("EntityTransformerPolicy.forward requires an EncodedBatch")
            if batch.schema_fingerprint != self._schema.schema_fingerprint:
                raise ContractError("EncodedBatch schema fingerprint does not match this policy")
            layout = self.tokenise(batch)
            padding = ~layout.valid
            empty = padding.unsqueeze(-1)  # padding positions are held at exactly 0
            hidden = self._pre_trunk(batch, layout)
            for block in self.blocks:
                hidden = block(hidden, None, padding).masked_fill(empty, 0.0)
            hidden = self.trunk_norm(hidden).masked_fill(empty, 0.0)
            pooled = hidden.sum(dim=1) / layout.counts.unsqueeze(-1).to(dtype=hidden.dtype)
            heads = self.value_heads(pooled)
            values, aux_values = heads[:, 0], heads[:, 1:]
            if not torch.isfinite(heads).all().item():
                raise ContractError("value heads must be finite")
            logits = self._score_actions(batch, layout, hidden, padding)
            legal = logits.masked_select(batch.action_mask.to(dtype=torch.bool))
            if legal.numel() and not torch.isfinite(legal).all().item():
                raise ContractError("legal action logits must be finite")
            return PolicyOutput(logits, values, aux_values)

        def _encode(self, frames: Sequence[Any]) -> EncodedBatch:
            # The packed tokens are the same for a trimmed and a full batch; trimming is cheaper.
            return encode_frames(frames, self._schema, device=str(self.group_embeddings.device), trim_padding=True)

        def act(self, frames: Sequence[Any], *, generator: Any, deterministic: bool = False) -> tuple[Any, ...]:
            if type(deterministic) is not bool:
                raise ContractError("deterministic must be a boolean")
            from spire_learn.neural.contracts import PolicyChoice
            from spire_learn.neural.distribution import masked_categorical, pad_to_dispatch_width
            from spire_learn.neural.model_families import reflex_argmax

            with torch.no_grad():
                batch = self._encode(frames)
                output = self.forward(batch)
                # A compact layout samples at the dense dispatch width, as the FF student does.
                logits, mask = pad_to_dispatch_width(output.logits, batch.action_mask,
                                                     self._schema.view.dense_action_width)
                distribution = masked_categorical(logits, mask)
                actions = (reflex_argmax(logits, mask) if deterministic
                           else distribution.sample(generator=generator))
                log_probs = distribution.log_prob(actions)
                return tuple(
                    PolicyChoice(original_action_index=int(actions[index].item()),
                                 log_probability=float(log_probs[index].item()),
                                 value=float(output.values[index].item()),
                                 behavior_policy_version=self._behavior_policy_version)
                    for index in range(len(frames)))

        def values(self, frames: Sequence[Any]) -> tuple[float, ...]:
            with torch.no_grad():
                output = self.forward(self._encode(frames))
                return tuple(float(value.item()) for value in output.values)

    return EntityTransformerPolicy


@functools.lru_cache(maxsize=1)
def _cardmeta_policy_class():
    import torch
    from torch import nn

    base = _policy_class()

    class CardMetaTransformerPolicy(base):
        """S2 plus a linear map of the card's static metadata added to its row embedding."""

        def __init__(self, plan: TransformerPlan, schema: CombatSchema, config: TransformerCardMetaConfig, *,
                     groups: tuple[CardMetadataGroup, ...], tables: CardMetadataTables, **identity) -> None:
            super().__init__(plan, schema, config, **identity)  # the size fields are TransformerConfig's
            self._card_groups = {item.group: item for item in groups}
            self.card_projections = nn.ModuleDict({
                item.group: nn.Linear(CARD_METADATA_FEATURES, config.width) for item in groups})
            self.register_buffer("_card_static", torch.tensor(tables.static, dtype=torch.float32), persistent=False)
            self.register_buffer("_card_faces", torch.tensor(tables.faces, dtype=torch.float32), persistent=False)

        def _card_features(self, item: CardMetadataGroup, values, known):
            """``[rows, CARD_METADATA_FEATURES]``: zero where the card is unknown or has no row."""
            flag = known[:, item.card_field_index]
            code = torch.where(flag, values[:, item.card_field_index], torch.zeros_like(flag, dtype=values.dtype))
            if item.upgrades_field_index is None:
                level = torch.zeros_like(code)
            else:
                upgrades = torch.where(known[:, item.upgrades_field_index], values[:, item.upgrades_field_index],
                                       torch.zeros_like(code))
                level = upgrades.clamp(min=0, max=self._card_faces.shape[0] - 1)
            features = torch.cat((self._card_static[code], self._card_faces[level, code]), dim=-1)
            return features * flag.unsqueeze(-1).to(dtype=features.dtype)

        def _embed(self, group: GroupPlan, values, known, references):
            out = super()._embed(group, values, known, references)
            item = self._card_groups.get(group.name)
            if item is None:
                return out
            return out + self.card_projections[group.name](self._card_features(item, values, known))

    return CardMetaTransformerPolicy


def _group_key(group: GroupPlan) -> str:
    return f"g{group.index}"


def _field_key(group: GroupPlan, field_index: int) -> str:
    return f"g{group.index}f{field_index}"


def _bucket_tensor(torch: Any, values: Any) -> Any:
    """``bucket_index`` on an int64 tensor."""
    magnitude = values.abs()
    logs = torch.floor(torch.log2(magnitude.clamp(min=1).to(dtype=torch.float64))).to(dtype=torch.int64)
    magnitude = torch.where(magnitude < BUCKET_EXACT, magnitude, (BUCKET_EXACT - 4) + logs)
    magnitude = magnitude.clamp(max=BUCKET_MAX_MAGNITUDE)
    return BUCKET_MAX_MAGNITUDE + torch.sign(values) * magnitude


def build_transformer_policy(
    schema: CombatSchema,
    model_config: TransformerConfig,
    *,
    seed: int,
    device: str,
    objective_id: str = DISTILL_OBJECTIVE_ID,
    model_id: str = TRANSFORMER_MODEL_ID,
    features_id: str = TRANSFORMER_FEATURES_ID,
    trim_padding: bool = False,
    experimental_trim_compute: bool = False,
) -> Any:
    """Build the entity transformer; everything but the module itself is checked before torch loads.

    ``trim_padding`` is accepted for the family interface; the outputs are the same either way.
    """
    if not isinstance(model_config, TransformerConfig):
        raise ContractError("the entity transformer requires a TransformerConfig")
    plan = transformer_plan(schema)
    if (model_id, features_id) != (TRANSFORMER_MODEL_ID, TRANSFORMER_FEATURES_ID):
        raise ContractError("the entity transformer's model_id/features_id must be its own identity pair")
    if objective_id != DISTILL_OBJECTIVE_ID:
        raise ContractError(f"the entity transformer is a {DISTILL_OBJECTIVE_ID} student only")
    if _STANDARD_VIEW_ID not in (schema.view.view_id, schema.view.dense_view_id):
        raise ContractError("the entity transformer requires the Standard combat view")
    if type(seed) is not int:
        raise ContractError("policy seed must be a primitive integer")
    if not isinstance(device, str) or not device:
        raise ContractError("policy device must be a non-empty string")
    if type(trim_padding) is not bool or type(experimental_trim_compute) is not bool:
        raise ContractError("trim_padding and experimental_trim_compute must be booleans")
    if experimental_trim_compute:
        raise ContractError("the entity transformer has no experimental_trim_compute mode")
    count = transformer_parameter_count(plan, model_config)
    if count > TRANSFORMER_PARAMETER_CAP:
        raise ContractError(f"the entity transformer needs {count} parameters; its cap is {TRANSFORMER_PARAMETER_CAP}")
    import torch

    policy_class = _policy_class()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        policy = policy_class(plan, schema, model_config, model_id=model_id, features_id=features_id,
                              objective_id=objective_id, trim_padding=trim_padding)
    policy = policy.to(device=device)
    actual = sum(int(parameter.numel()) for parameter in policy.parameters())
    if actual != count:
        raise ContractError(f"entity transformer parameter count mismatch: expected {count}, got {actual}")
    return policy


def build_transformer_cardmeta_policy(
    schema: CombatSchema,
    model_config: TransformerCardMetaConfig,
    *,
    seed: int,
    device: str,
    objective_id: str = DISTILL_OBJECTIVE_ID,
    model_id: str = TRANSFORMER_CARDMETA_MODEL_ID,
    features_id: str = TRANSFORMER_CARDMETA_FEATURES_ID,
    trim_padding: bool = False,
    experimental_trim_compute: bool = False,
    card_metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Build S3, the entity transformer with card metadata; everything but the module is checked first.

    ``card_metadata`` is the descriptor to join (default: the committed snapshot, ``load_card_metadata()``);
    either way its fingerprint must be the pinned one and the config's. ``trim_padding`` is accepted
    for the family interface; the outputs are the same either way.
    """
    if not isinstance(model_config, TransformerCardMetaConfig):
        raise ContractError("the card-metadata transformer requires a TransformerCardMetaConfig")
    plan = transformer_plan(schema)
    if (model_id, features_id) != (TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_CARDMETA_FEATURES_ID):
        raise ContractError("the card-metadata transformer's model_id/features_id must be its own identity pair")
    if objective_id != DISTILL_OBJECTIVE_ID:
        raise ContractError(f"the card-metadata transformer is a {DISTILL_OBJECTIVE_ID} student only")
    if _STANDARD_VIEW_ID not in (schema.view.view_id, schema.view.dense_view_id):
        raise ContractError("the card-metadata transformer requires the Standard combat view")
    if type(seed) is not int:
        raise ContractError("policy seed must be a primitive integer")
    if not isinstance(device, str) or not device:
        raise ContractError("policy device must be a non-empty string")
    if type(trim_padding) is not bool or type(experimental_trim_compute) is not bool:
        raise ContractError("trim_padding and experimental_trim_compute must be booleans")
    if experimental_trim_compute:
        raise ContractError("the card-metadata transformer has no experimental_trim_compute mode")
    groups = card_metadata_groups(schema)
    count = transformer_cardmeta_parameter_count(plan, model_config)
    if count > TRANSFORMER_PARAMETER_CAP:
        raise ContractError(
            f"the card-metadata transformer needs {count} parameters; its cap is {TRANSFORMER_PARAMETER_CAP}")
    metadata = load_card_metadata() if card_metadata is None else card_metadata
    if metadata.get("fingerprint") != model_config.card_metadata_fingerprint:
        raise ContractError("the card metadata's fingerprint is not the config's card_metadata_fingerprint")
    tables = card_metadata_tables(metadata, schema.vocabularies[CARD_VOCABULARY])
    import torch

    policy_class = _cardmeta_policy_class()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        policy = policy_class(plan, schema, model_config, groups=groups, tables=tables, model_id=model_id,
                              features_id=features_id, objective_id=objective_id, trim_padding=trim_padding)
    policy = policy.to(device=device)
    actual = sum(int(parameter.numel()) for parameter in policy.parameters())
    if actual != count:
        raise ContractError(f"card-metadata transformer parameter count mismatch: expected {count}, got {actual}")
    return policy


__all__ = [
    "ACTION_GROUP", "CARD_COLORS", "CARD_COST_KINDS", "CARD_FACE_FEATURES", "CARD_KEYWORDS", "CARD_METADATA_FEATURES",
    "CARD_METADATA_GROUPS", "CARD_METADATA_SCHEMA_ID", "CARD_METADATA_SNAPSHOT", "CARD_NULLABLE_FIELDS", "CARD_RARITIES",
    "CARD_STATIC_FEATURES", "CARD_TAGS", "CARD_TARGETS", "CARD_TYPES", "CARD_VOCABULARY", "CardMetadataGroup",
    "CardMetadataTables", "build_transformer_cardmeta_policy", "card_face_features", "card_metadata_groups",
    "card_metadata_tables", "card_static_features", "load_card_metadata", "transformer_cardmeta_parameter_count",
    "BUCKET_COUNT", "BUCKET_EXACT", "BUCKET_FEATURES", "BUCKET_MAX_MAGNITUDE", "FFN_RATIO",
    "GroupPlan", "ReferencePlan", "SINUSOIDAL_FREQUENCIES", "TokenLayout", "TransformerPlan", "bucket_index",
    "build_transformer_policy", "transformer_parameter_count", "transformer_plan",
]
