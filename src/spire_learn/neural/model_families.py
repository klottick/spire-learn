"""The model-family registry: how each learner ``model_id`` builds, encodes and caps its policy.

A ``ModelFamily`` gives one family's ``build`` (schema, config, keyword options; the FF family's
is ``model.build_policy`` exactly), its ``config_type`` (the parsed model table), its
``parameter_cap`` and ``encode`` (public frames to the batch its ``forward`` takes). Families are
keyed by every ``model_id`` they own. The FF family owns the M10 and Standard FF learners; the
entity transformer family (S2, ``transformer.build_transformer_policy``, ``TransformerConfig``, a
16,000,000 parameter cap) owns ``spire-learn-combat-entity-transformer-v1``, and the card-metadata
transformer family (S3, ``transformer.build_transformer_cardmeta_policy``, ``TransformerCardMetaConfig``,
the same cap) owns ``spire-learn-combat-entity-transformer-cardmeta-v1``. All are registered here; a later family registers with ``register_model_family``.

``build_family_policy`` builds through the registry and refuses a policy that lacks what the
``Collector`` and the evaluators use: ``act``, ``values`` and ``forward``, a
``_behavior_policy_version``, and identity attributes ``model_id``, ``features_id``,
``action_id`` and ``objective_id`` that match the request. An unregistered ``model_id`` is refused,
never defaulted to FF.

``reflex_argmax`` is the student evaluation's action rule, FF's rule exactly: the argmax over legal
actions, lowest index on ties, and a ``ContractError`` (a fault, not a loss) when a legal set is
empty or holds any non-finite logit.

Importing this module imports neither torch nor numpy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import statistics
import time
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.neural.config import (
    LEARNER_IDENTITY_IDS, STANDARD_LEARNER_IDENTITY_IDS, TRANSFORMER_CARDMETA_MODEL_ID, TRANSFORMER_MODEL_ID,
    TRANSFORMER_PARAMETER_CAP, TransformerCardMetaConfig, TransformerConfig,
)
from spire_learn.neural.contracts import ModelConfig

FF_PARAMETER_CAP = 10_000_000
# What every built policy exposes (the Collector checks act/values and the identity attributes).
POLICY_METHODS = ("act", "values", "forward")
POLICY_IDENTITY_ATTRIBUTES = ("model_id", "features_id", "action_id", "objective_id")


@dataclass(frozen=True)
class ModelFamily:
    family_id: str
    model_ids: tuple[str, ...]
    config_type: type
    parameter_cap: int
    build: Callable[..., Any]
    encode: Callable[..., Any]
    # Check every built policy's interface and parameter count (see build_family_policy).
    check_built: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ContractError("a model family needs a non-empty family_id")
        if (not isinstance(self.model_ids, tuple) or not self.model_ids
                or any(not isinstance(item, str) or not item for item in self.model_ids)
                or len(set(self.model_ids)) != len(self.model_ids)):
            raise ContractError("a model family owns a non-empty tuple of distinct model ids")
        if not isinstance(self.config_type, type):
            raise ContractError("a model family's config_type must be a type")
        if type(self.parameter_cap) is not int or self.parameter_cap < 1:
            raise ContractError("a model family's parameter_cap must be a positive integer")
        if not callable(self.build) or not callable(self.encode):
            raise ContractError("a model family's build and encode must be callable")


def _ff_build(schema, model_config, **options):
    from spire_learn.neural.model import build_policy

    return build_policy(schema, model_config, **options)


def _ff_encode(frames, schema, *, device: str, trim_padding: bool = False):
    from spire_learn.neural.features import encode_frames

    return encode_frames(frames, schema, device=device, trim_padding=trim_padding)


FF_FAMILY = ModelFamily(
    family_id="FF",
    model_ids=(LEARNER_IDENTITY_IDS[0], STANDARD_LEARNER_IDENTITY_IDS[0]),
    config_type=ModelConfig,
    parameter_cap=FF_PARAMETER_CAP,
    build=_ff_build,
    encode=_ff_encode,
    # build_policy enforces its exact parameter count and the 10M cap itself, and its interface is
    # pinned by tests/neural/test_model.py, so the FF path through the registry adds no step.
    check_built=False,
)

_FAMILIES: dict[str, ModelFamily] = {model_id: FF_FAMILY for model_id in FF_FAMILY.model_ids}


def _transformer_build(schema, model_config, **options):
    from spire_learn.neural.transformer import build_transformer_policy

    return build_transformer_policy(schema, model_config, **options)


# The entity transformer reads the same public EncodedBatch as FF (encode_frames); it tokenises the
# occupied rows itself, so trimmed and full batches give the same outputs.
TRANSFORMER_FAMILY = ModelFamily(
    family_id="transformer",
    model_ids=(TRANSFORMER_MODEL_ID,),
    config_type=TransformerConfig,
    parameter_cap=TRANSFORMER_PARAMETER_CAP,
    build=_transformer_build,
    encode=_ff_encode,
)


def _transformer_cardmeta_build(schema, model_config, **options):
    from spire_learn.neural.transformer import build_transformer_cardmeta_policy

    return build_transformer_cardmeta_policy(schema, model_config, **options)


# S3 reads the same public EncodedBatch; its card metadata is static public data joined inside the model.
TRANSFORMER_CARDMETA_FAMILY = ModelFamily(
    family_id="transformer-cardmeta",
    model_ids=(TRANSFORMER_CARDMETA_MODEL_ID,),
    config_type=TransformerCardMetaConfig,
    parameter_cap=TRANSFORMER_PARAMETER_CAP,
    build=_transformer_cardmeta_build,
    encode=_ff_encode,
)


def registered_model_ids() -> tuple[str, ...]:
    return tuple(sorted(_FAMILIES))


def register_model_family(family: ModelFamily) -> None:
    """Register a new family under each of its model ids; an id that is already taken is refused."""
    if not isinstance(family, ModelFamily):
        raise TypeError("family must be a ModelFamily")
    taken = sorted(model_id for model_id in family.model_ids if model_id in _FAMILIES)
    if taken:
        raise ContractError(f"model ids already have a registered family: {', '.join(taken)}")
    for model_id in family.model_ids:
        _FAMILIES[model_id] = family


register_model_family(TRANSFORMER_FAMILY)
register_model_family(TRANSFORMER_CARDMETA_FAMILY)


def model_family(model_id: str) -> ModelFamily:
    """The registered family of ``model_id``; an unknown id is refused, never defaulted."""
    family = _FAMILIES.get(model_id) if isinstance(model_id, str) else None
    if family is None:
        known = ", ".join(sorted({item.family_id for item in _FAMILIES.values()}))
        raise ContractError(f"no registered model family for model_id {model_id!r} (registered families: {known})")
    return family


def check_family_config(model_id: str, model_config: object) -> ModelFamily:
    """The family of ``model_id``, after checking ``model_config`` is that family's config type."""
    family = model_family(model_id)
    if not isinstance(model_config, family.config_type):
        raise TypeError(f"model_config must be a {family.config_type.__name__} for the {family.family_id} family")
    return family


def check_policy(policy: Any, *, model_id: str, objective_id: str | None = None) -> Any:
    """Refuse a built policy missing the interface the Collector and the evaluators use."""
    missing = [name for name in POLICY_METHODS if not callable(getattr(policy, name, None))]
    missing += [name for name in ("_behavior_policy_version", *POLICY_IDENTITY_ATTRIBUTES)
                if not hasattr(policy, name)]
    if missing:
        raise ContractError(f"the built policy lacks {', '.join(missing)}")
    if policy.model_id != model_id:
        raise ContractError(f"the built policy declares model_id {policy.model_id!r}, not {model_id!r}")
    if objective_id is not None and policy.objective_id != objective_id:
        raise ContractError(f"the built policy declares objective_id {policy.objective_id!r}, not {objective_id!r}")
    return policy


def build_family_policy(model_id: str, schema, model_config, /, **options) -> Any:
    """Build ``model_id``'s policy through its family; ``options`` pass to the family's build unchanged.

    For the FF family this is exactly ``build_policy(schema, model_config, **options)``. Any other
    family's policy is checked against its parameter cap and the required interface.
    """
    family = check_family_config(model_id, model_config)
    policy = family.build(schema, model_config, **options)
    if not family.check_built:
        return policy
    parameters = getattr(policy, "parameters", None)
    if callable(parameters):
        count = sum(int(parameter.numel()) for parameter in parameters())
        if count > family.parameter_cap:
            raise ContractError(f"{family.family_id} policy has {count} parameters, over its cap {family.parameter_cap}")
    return check_policy(policy, model_id=model_id, objective_id=options.get("objective_id"))


def reflex_argmax(logits: Any, action_mask: Any) -> Any:
    """Per row, the legal action with the largest logit (lowest index on ties); int64.

    FF's rule exactly: illegal actions never win, and a row with an empty legal set or any
    non-finite legal logit (NaN, +inf or -inf) raises a ``ContractError``. A student that cannot
    act is a fault, never a loss, and a legal ``-inf`` is never silently picked around.
    """
    import torch

    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ContractError("reflex_argmax takes [batch, action] logits")
    mask = action_mask.to(dtype=torch.bool)
    if tuple(mask.shape) != tuple(logits.shape):
        raise ContractError("action_mask shape must match the logits")
    if not mask.any(dim=-1).all().item():
        raise ContractError("a legal set is empty: the student cannot act")
    if not torch.isfinite(logits.masked_select(mask)).all().item():
        raise ContractError("a legal action logit is not finite: the student cannot act")
    return logits.masked_fill(~mask, -torch.inf).argmax(dim=-1).to(dtype=torch.int64)


def median_forward_seconds(forward: Callable[[], Any], *, passes: int = 1000, warmup: int = 50,
                           synchronize: Callable[[], None] | None = None,
                           timer: Callable[[], float] = time.perf_counter) -> dict[str, object]:
    """Median wall time of ``passes`` calls of ``forward`` after ``warmup`` untimed calls.

    ``synchronize`` (``torch.cuda.synchronize`` on a GPU) runs before each timer read, so each
    sample covers the whole pass.
    """
    if type(passes) is not int or passes < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("passes must be a positive integer and warmup a non-negative integer")
    sync = synchronize or (lambda: None)
    for _ in range(warmup):
        forward()
    sync()
    samples = []
    for _ in range(passes):
        started = timer()
        forward()
        sync()
        samples.append(timer() - started)
    return {"median_seconds": statistics.median(samples), "passes": passes, "warmup": warmup,
            "min_seconds": min(samples), "max_seconds": max(samples)}


def time_batch1_forward(policy: Any, family: ModelFamily, frame, schema, *, device: str,
                        trim_padding: bool = False, passes: int = 1000, warmup: int = 50,
                        torch: Any = None) -> dict[str, object]:
    """Warmed batch-1 ``forward`` timing of ``policy`` on ``device`` (the frame is encoded once)."""
    if torch is None:
        import torch
    batch = family.encode([frame], schema, device=device, trim_padding=trim_padding)
    synchronize = torch.cuda.synchronize if str(device).startswith("cuda") else None

    def forward():
        with torch.no_grad():
            return policy.forward(batch)

    return {"device": device, **median_forward_seconds(forward, passes=passes, warmup=warmup,
                                                       synchronize=synchronize)}


__all__ = [
    "FF_FAMILY", "FF_PARAMETER_CAP", "ModelFamily", "TRANSFORMER_CARDMETA_FAMILY", "TRANSFORMER_FAMILY", "POLICY_IDENTITY_ATTRIBUTES", "POLICY_METHODS",
    "build_family_policy", "check_family_config", "check_policy", "median_forward_seconds", "model_family",
    "reflex_argmax", "register_model_family", "registered_model_ids", "time_batch1_forward",
]
