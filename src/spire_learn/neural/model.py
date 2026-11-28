"""Lazy CPU/GPU-capable typed actor-critic construction and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.neural.config import (
    DISTILL_OBJECTIVE_ID, LEARNER_IDENTITY_IDS, OBJECTIVE_IDS, STANDARD_LEARNER_IDENTITY_IDS,
    WINLOSS_OBJECTIVE_ID, model_config_hash,
)
from spire_learn.neural.contracts import CombatFrame, ModelConfig
from spire_learn.neural.features import EncodedBatch, FeatureContract, encode_frames, feature_contract
from spire_learn.neural.schema import CombatSchema


MODEL_ID = "spire-learn-combat-ff-v1"
_ACTION_ID = "spire-learn-combat-dispatch-v1"
# The default objective; callers pass the config's accepted objective ID otherwise.
_OBJECTIVE_ID = WINLOSS_OBJECTIVE_ID
_HARD_PARAMETER_CAP = 10_000_000
# A policy is built for a PPO objective or for supervised distillation.
_POLICY_OBJECTIVE_IDS = (*OBJECTIVE_IDS, DISTILL_OBJECTIVE_ID)


@dataclass(frozen=True)
class PolicyOutput:
    """Actor logits in original dispatch order and one critic value per frame.

    ``values`` is always the outcome (PPO critic) head. ``aux_values`` holds the other value
    heads of a ``value_heads = 3`` student, ``(batch, 2)``: HP kept, then realised v2. It is
    None for a one-head policy.
    """

    logits: Any
    values: Any
    aux_values: Any = None


@dataclass(frozen=True)
class _ReferencePlan:
    group: str
    field_index: int
    selector_index: int | None
    choices: tuple[tuple[int | None, str], ...]


@dataclass(frozen=True)
class _ModelPlan:
    schema: CombatSchema
    feature_contract: FeatureContract
    group_names: tuple[str, ...]
    group_indices: Mapping[str, int]
    field_indices: Mapping[str, Mapping[str, int]]
    category_sizes: Mapping[tuple[str, int], int]
    reference_plans: tuple[_ReferencePlan, ...]
    reference_by_field: Mapping[tuple[str, int], _ReferencePlan]
    row_input_widths: Mapping[str, int]
    global_groups: tuple[str, ...]
    coverage_groups: tuple[str, ...]
    global_input_width: int
    power_targets: tuple[str, ...]
    parameter_count: int


def _linear_parameter_count(input_width: int, output_width: int) -> int:
    return input_width * output_width + output_width


def _field_key(group_index: int, field_index: int) -> str:
    return f"g{group_index}f{field_index}"


def _group_key(group_index: int) -> str:
    return f"g{group_index}"


def _reference_plan(
    schema: CombatSchema,
    group: str,
    field_index: int,
) -> _ReferencePlan:
    spec = schema.tensors[group]
    column = spec.columns[field_index]
    assert column.reference is not None
    selector_index = next(
        (index for index, item in enumerate(spec.columns) if item.field == column.reference),
        None,
    )
    if selector_index is None:
        target = column.reference
        target_spec = schema.tensors.get(target)
        if target_spec is None or not target_spec.columns:
            raise ContractError(f"{group}.{column.field} targets an unavailable public group")
        choices = ((None, target),)
    else:
        selector = spec.columns[selector_index]
        if selector.vocabulary is None:
            raise ContractError(f"{group}.{column.field} selector must be categorical")
        vocabulary = schema.vocabularies[selector.vocabulary]
        choices = tuple(
            (index, name)
            for index, name in enumerate(vocabulary)
            if name != "none" and name in schema.tensors and schema.tensors[name].columns
        )
    return _ReferencePlan(group, field_index, selector_index, choices)


def _compile_model_plan(schema: CombatSchema, config: ModelConfig) -> _ModelPlan:
    if not isinstance(schema, CombatSchema):
        raise ContractError("build_policy requires an authenticated CombatSchema")
    if not isinstance(config, ModelConfig):
        raise ContractError("build_policy requires a ModelConfig")
    plan = feature_contract(schema)
    group_names = tuple(plan.groups)
    group_indices = {name: index for index, name in enumerate(group_names)}
    field_indices = {
        group: {field.field: index for index, field in enumerate(fields)}
        for group, fields in plan.groups.items()
    }
    category_sizes: dict[tuple[str, int], int] = {}
    row_input_widths: dict[str, int] = {}
    reference_plans: list[_ReferencePlan] = []
    reference_by_field: dict[tuple[str, int], _ReferencePlan] = {}
    embedding_width = config.embedding_width
    hidden_width = config.hidden_width

    for group, fields in plan.groups.items():
        input_width = 0
        for index, item in enumerate(fields):
            if item.kind == "categorical":
                assert item.vocabulary is not None
                vocabulary = schema.vocabularies.get(item.vocabulary)
                if vocabulary is None:
                    raise ContractError(f"{group}.{item.field} names a missing vocabulary")
                category_sizes[(group, index)] = len(vocabulary)
                input_width += embedding_width + 1
            elif item.kind == "numeric":
                input_width += 2
            else:
                relation = _reference_plan(schema, group, index)
                reference_plans.append(relation)
                reference_by_field[(group, index)] = relation
                input_width += 2
        row_input_widths[group] = input_width

    if "actions" not in plan.groups:
        raise ContractError("CombatSchema has no public actions value group")
    global_groups = tuple(group for group in group_names if group != "actions")
    coverage_groups = group_names
    global_input_width = sum(
        hidden_width if len(schema.tensors[group].shape) == 1 else 2 * hidden_width
        for group in global_groups
    ) + sum(len(plan.groups[group]) for group in coverage_groups)

    power_targets = tuple(sorted({
        target
        for relation in reference_plans
        if relation.group == "powers"
        for _selector, target in relation.choices
        if target in {"player", "enemies"}
    }))

    count = 0
    for (group, _field_index), vocabulary_size in category_sizes.items():
        del group
        count += (vocabulary_size + 1) * embedding_width
    for group in group_names:
        input_width = row_input_widths[group]
        count += _linear_parameter_count(input_width, hidden_width)
        count += _linear_parameter_count(hidden_width, hidden_width)
    for _relation in reference_plans:
        count += _linear_parameter_count(2 * hidden_width + 2, hidden_width)
    for _target in power_targets:
        count += _linear_parameter_count(hidden_width, hidden_width)
    count += _linear_parameter_count(global_input_width, hidden_width)
    count += _linear_parameter_count(hidden_width, hidden_width)
    count += _linear_parameter_count(4 * hidden_width + 4, hidden_width)
    count += _linear_parameter_count(hidden_width, hidden_width)
    count += _linear_parameter_count(hidden_width, 1)
    count += _linear_parameter_count(hidden_width, 1)
    if config.value_heads > 1:
        count += _linear_parameter_count(hidden_width, config.value_heads - 1)

    limit = min(config.max_parameters, _HARD_PARAMETER_CAP)
    if count > limit:
        raise ContractError(
            f"compiled combat model requires {count} parameters; effective cap is {limit}"
        )
    return _ModelPlan(
        schema=schema,
        feature_contract=plan,
        group_names=group_names,
        group_indices=group_indices,
        field_indices=field_indices,
        category_sizes=category_sizes,
        reference_plans=tuple(reference_plans),
        reference_by_field=reference_by_field,
        row_input_widths=row_input_widths,
        global_groups=global_groups,
        coverage_groups=coverage_groups,
        global_input_width=global_input_width,
        power_targets=power_targets,
        parameter_count=count,
    )


def _reference_state(batch: EncodedBatch, relation: _ReferencePlan, torch: Any) -> tuple[Any, Any]:
    group = relation.group
    known = batch.known[group][..., relation.field_index]
    if relation.selector_index is None:
        return torch.ones_like(known), known
    selector_known = batch.known[group][..., relation.selector_index]
    selector_values = batch.values[group][..., relation.selector_index]
    applicable = torch.zeros_like(known)
    for selector_value, _target in relation.choices:
        assert selector_value is not None
        applicable = applicable | (selector_known & selector_values.eq(selector_value))
    return applicable, known


def _gather_reference(
    batch: EncodedBatch,
    relation: _ReferencePlan,
    target_embeddings: Mapping[str, Any],
    hidden_width: int,
    torch: Any,
) -> tuple[Any, Any, Any]:
    group = relation.group
    raw = batch.values[group][..., relation.field_index].to(dtype=torch.int64)
    applicable, known = _reference_state(batch, relation, torch)
    source_shape = (*raw.shape, hidden_width)
    gathered_total = torch.zeros(source_shape, dtype=torch.float32, device=raw.device)
    if relation.selector_index is None:
        selector_values = None
        selector_known = None
    else:
        selector_values = batch.values[group][..., relation.selector_index]
        selector_known = batch.known[group][..., relation.selector_index]

    for selector_value, target in relation.choices:
        target_rows = target_embeddings[target]
        capacity = target_rows.shape[1]
        if selector_value is None:
            matches = applicable
        else:
            assert selector_values is not None and selector_known is not None
            matches = selector_known & selector_values.eq(selector_value)
        in_bounds = (raw >= 0) & (raw < capacity)
        safe_rows = raw.clamp(min=0, max=capacity - 1)
        valid = matches & known & in_bounds
        target_spec = batch.occupied.get(target)
        if target_spec is not None:
            row_occupied = target_spec.gather(1, safe_rows)
            valid = valid & row_occupied
        selected = target_rows.gather(
            1,
            safe_rows.unsqueeze(-1).expand(*safe_rows.shape, hidden_width),
        )
        gathered_total = gathered_total + torch.where(
            valid.unsqueeze(-1), selected, torch.zeros_like(selected)
        )
    return gathered_total, applicable, known


def build_policy(
    schema: CombatSchema,
    model_config: ModelConfig,
    *,
    seed: int,
    device: str,
    objective_id: str = _OBJECTIVE_ID,
    model_id: str = MODEL_ID,
    features_id: str | None = None,
    trim_padding: bool = False,
    experimental_trim_compute: bool = False,
) -> Any:
    """Compile a capped shared candidate actor and public-state critic lazily.

    ``objective_id`` labels the reward the policy is trained under; it does not
    change the architecture, but checkpoints and rollouts carry and check it.
    ``model_id``/``features_id`` select the learner identity: the default v1 pair, or
    the Standard pair, which requires the Standard combat view. Neither changes the
    descriptor-derived architecture.
    """
    plan = _compile_model_plan(schema, model_config)
    if features_id is None:
        features_id = plan.feature_contract.features_id
    if (model_id, features_id) not in (LEARNER_IDENTITY_IDS[:2], STANDARD_LEARNER_IDENTITY_IDS[:2]):
        raise ContractError("policy model_id/features_id must be a supported learner identity pair")
    if ((model_id, features_id) == STANDARD_LEARNER_IDENTITY_IDS[:2]
            and "spire-gym-standard-combat-view-v1" not in (schema.view.view_id, schema.view.dense_view_id)):
        raise ContractError("the Standard learner identity requires the Standard combat view")
    if type(seed) is not int:
        raise ContractError("policy seed must be a primitive integer")
    if not isinstance(objective_id, str) or objective_id not in _POLICY_OBJECTIVE_IDS:
        raise ContractError("policy objective_id must be a supported combat objective")
    if not isinstance(device, str) or not device:
        raise ContractError("policy device must be a non-empty string")
    if type(trim_padding) is not bool:
        raise ContractError("trim_padding must be a boolean")
    if type(experimental_trim_compute) is not bool:
        raise ContractError("experimental_trim_compute must be a boolean")
    if experimental_trim_compute and not trim_padding:
        raise ContractError("experimental_trim_compute requires trim_padding")
    import torch

    class CombatPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._schema = plan.schema
            self._trim_padding = trim_padding
            self._feature_contract = plan.feature_contract
            self._group_names = plan.group_names
            self._group_indices = dict(plan.group_indices)
            self._field_indices = {name: dict(items) for name, items in plan.field_indices.items()}
            self._category_sizes = dict(plan.category_sizes)
            self._reference_plans = plan.reference_plans
            self._reference_by_field = dict(plan.reference_by_field)
            self._row_input_widths = dict(plan.row_input_widths)
            self._global_groups = plan.global_groups
            self._coverage_groups = plan.coverage_groups
            self._power_targets = plan.power_targets
            self._hidden_width = model_config.hidden_width
            self._embedding_width = model_config.embedding_width
            self._parameter_count = plan.parameter_count
            # The shared hash payload: value_heads = 1 is left out, as before it existed.
            self._model_config_hash = model_config_hash(model_config)
            self._value_heads = model_config.value_heads
            self._behavior_policy_version: int = 0
            self._consumed_decision_ids: set[tuple[str, str]] = set()
            self._consumed_decision_limit = 65_536
            self.model_id = model_id
            self.features_id = features_id
            self.action_id = _ACTION_ID
            self.objective_id = objective_id

            self.category_embeddings = torch.nn.ModuleDict()
            for (group, field_index), vocabulary_size in plan.category_sizes.items():
                key = _field_key(plan.group_indices[group], field_index)
                self.category_embeddings[key] = torch.nn.Embedding(
                    vocabulary_size + 1, model_config.embedding_width
                )

            self.group_first = torch.nn.ModuleDict()
            self.group_second = torch.nn.ModuleDict()
            for group in plan.group_names:
                key = _group_key(plan.group_indices[group])
                self.group_first[key] = torch.nn.Linear(
                    plan.row_input_widths[group], model_config.hidden_width
                )
                self.group_second[key] = torch.nn.Linear(
                    model_config.hidden_width, model_config.hidden_width
                )

            self.reference_layers = torch.nn.ModuleDict()
            for relation in plan.reference_plans:
                key = _field_key(plan.group_indices[relation.group], relation.field_index)
                self.reference_layers[key] = torch.nn.Linear(
                    2 * model_config.hidden_width + 2, model_config.hidden_width
                )

            self.power_to_owner = torch.nn.ModuleDict()
            for group in plan.power_targets:
                key = _group_key(plan.group_indices[group])
                self.power_to_owner[key] = torch.nn.Linear(
                    model_config.hidden_width, model_config.hidden_width
                )

            self.global_first = torch.nn.Linear(
                plan.global_input_width, model_config.hidden_width
            )
            self.global_second = torch.nn.Linear(
                model_config.hidden_width, model_config.hidden_width
            )
            self.candidate_first = torch.nn.Linear(
                4 * model_config.hidden_width + 4, model_config.hidden_width
            )
            self.candidate_second = torch.nn.Linear(
                model_config.hidden_width, model_config.hidden_width
            )
            self.candidate_output = torch.nn.Linear(model_config.hidden_width, 1)
            self.value_head = torch.nn.Linear(model_config.hidden_width, 1)
            if model_config.value_heads > 1:
                # Created last, so a one-head policy draws exactly its historical initial weights.
                self.aux_value_head = torch.nn.Linear(
                    model_config.hidden_width, model_config.value_heads - 1
                )

        def _base_group_embeddings(self, batch: EncodedBatch) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for group in self._group_names:
                values = batch.values[group]
                known = batch.known[group]
                pieces: list[Any] = []
                group_index = self._group_indices[group]
                for field_index, item in enumerate(self._feature_contract.groups[group]):
                    field_known = known[..., field_index]
                    if item.kind == "categorical":
                        assert item.vocabulary is not None
                        vocabulary_size = self._category_sizes[(group, field_index)]
                        token = values[..., field_index].to(dtype=torch.int64)
                        token = torch.where(field_known, token, torch.full_like(token, vocabulary_size))
                        embedded = self.category_embeddings[
                            _field_key(group_index, field_index)
                        ](token)
                        pieces.extend((embedded, field_known.to(dtype=torch.float32).unsqueeze(-1)))
                    elif item.kind == "numeric":
                        numeric = values[..., field_index].to(dtype=torch.float32)
                        transformed = torch.sign(numeric) * torch.log1p(torch.abs(numeric))
                        transformed = torch.where(field_known, transformed, torch.zeros_like(transformed))
                        pieces.extend((transformed.unsqueeze(-1), field_known.to(dtype=torch.float32).unsqueeze(-1)))
                    else:
                        relation = self._reference_by_field[(group, field_index)]
                        applicable, ref_known = _reference_state(batch, relation, torch)
                        pieces.extend((
                            applicable.to(dtype=torch.float32).unsqueeze(-1),
                            ref_known.to(dtype=torch.float32).unsqueeze(-1),
                        ))
                row_input = torch.cat(pieces, dim=-1)
                key = _group_key(group_index)
                hidden = torch.tanh(self.group_first[key](row_input))
                result[group] = torch.tanh(self.group_second[key](hidden))
            return result

        def _reference_enrichment(self, batch: EncodedBatch, base: Mapping[str, Any]) -> dict[str, Any]:
            enriched: dict[str, Any] = {}
            for group in self._group_names:
                aggregate = torch.zeros_like(base[group])
                group_index = self._group_indices[group]
                for relation in self._reference_plans:
                    if relation.group != group:
                        continue
                    gathered, applicable, ref_known = _gather_reference(
                        batch, relation, base, self._hidden_width, torch
                    )
                    features = torch.cat((
                        base[group], gathered,
                        applicable.to(dtype=torch.float32).unsqueeze(-1),
                        ref_known.to(dtype=torch.float32).unsqueeze(-1),
                    ), dim=-1)
                    key = _field_key(group_index, relation.field_index)
                    aggregate = aggregate + torch.tanh(self.reference_layers[key](features))
                enriched[group] = base[group] + aggregate
            return enriched

        def _power_owner_enrichment(
            self,
            batch: EncodedBatch,
            rows: Mapping[str, Any],
        ) -> dict[str, Any]:
            if "powers" not in rows or not self._power_targets:
                return dict(rows)
            powers = rows["powers"]
            power_occupied = batch.occupied.get("powers")
            if power_occupied is None:
                return dict(rows)
            power_spec = self._schema.tensors["powers"]
            output = dict(rows)
            for target in self._power_targets:
                target_rows = rows[target]
                capacity = target_rows.shape[1]
                sums = torch.zeros_like(target_rows)
                counts = torch.zeros(
                    target_rows.shape[:2], dtype=torch.float32, device=target_rows.device
                )
                for relation in self._reference_plans:
                    if relation.group != "powers" or not any(
                        name == target for _value, name in relation.choices
                    ):
                        continue
                    raw_index = batch.values["powers"][..., relation.field_index].to(torch.int64)
                    applicable, ref_known = _reference_state(batch, relation, torch)
                    if relation.selector_index is None:
                        matches = applicable
                    else:
                        selector_values = batch.values["powers"][..., relation.selector_index]
                        selector_known = batch.known["powers"][..., relation.selector_index]
                        selector_value = next(
                            code for code, name in relation.choices if name == target
                        )
                        matches = selector_known & selector_values.eq(selector_value)
                    in_bounds = (raw_index >= 0) & (raw_index < capacity)
                    safe_index = raw_index.clamp(min=0, max=capacity - 1)
                    valid = power_occupied & matches & ref_known & in_bounds
                    target_occupied = batch.occupied.get(target)
                    if target_occupied is not None:
                        valid = valid & target_occupied.gather(1, safe_index)
                    messages = torch.where(
                        valid.unsqueeze(-1), powers, torch.zeros_like(powers)
                    )
                    scatter_index = safe_index.unsqueeze(-1).expand(
                        -1, -1, self._hidden_width
                    )
                    sums = sums.scatter_add(1, scatter_index, messages)
                    counts = counts.scatter_add(1, safe_index, valid.to(torch.float32))
                mean_power = sums / counts.clamp_min(1.0).unsqueeze(-1)
                projected = torch.tanh(
                    self.power_to_owner[_group_key(self._group_indices[target])](mean_power)
                )
                output[target] = rows[target] + torch.where(
                    counts.unsqueeze(-1) > 0,
                    projected,
                    torch.zeros_like(projected),
                )
            return output

        def _global_context(self, batch: EncodedBatch, rows: Mapping[str, Any]) -> Any:
            pieces: list[Any] = []
            for group in self._global_groups:
                group_rows = rows[group]
                if len(self._schema.tensors[group].shape) == 1:
                    pieces.append(group_rows[:, 0, :])
                    continue
                occupied = batch.occupied[group]
                expanded = occupied.unsqueeze(-1)
                counts = occupied.to(dtype=torch.float32).sum(dim=1, keepdim=True).clamp_min(1.0)
                mean = (group_rows * expanded.to(dtype=group_rows.dtype)).sum(dim=1) / counts
                masked_max = group_rows.masked_fill(~expanded, -torch.inf).max(dim=1).values
                has_rows = occupied.any(dim=1, keepdim=True)
                maximum = torch.where(has_rows, masked_max, torch.zeros_like(masked_max))
                pieces.extend((mean, maximum))
            pieces.extend(batch.known_coverage[group] for group in self._coverage_groups)
            joined = torch.cat(pieces, dim=-1)
            return torch.tanh(self.global_second(torch.tanh(self.global_first(joined))))

        def _candidate_references(
            self,
            batch: EncodedBatch,
            rows: Mapping[str, Any],
        ) -> tuple[Any, Any, Any, Any, Any, Any]:
            actions = batch.values["actions"]
            fields = self._field_indices["actions"]
            source_field = fields.get("source_row")
            target_field = fields.get("target_row")
            if source_field is None or target_field is None:
                raise ContractError("actions must declare typed source_row and target_row references")
            source_plan = self._reference_by_field[("actions", source_field)]
            target_plan = self._reference_by_field[("actions", target_field)]
            source, source_applicable, source_known = _gather_reference(
                batch, source_plan, rows, self._hidden_width, torch
            )
            target, target_applicable, target_known = _gather_reference(
                batch, target_plan, rows, self._hidden_width, torch
            )
            return (
                source,
                source_applicable,
                source_known,
                target,
                target_applicable,
                target_known,
            )

        def _restore_row_capacity(self, batch: EncodedBatch) -> EncodedBatch:
            # Keep GEMM and floating reduction shapes identical, including backward.
            values, known, occupied = dict(batch.values), dict(batch.known), dict(batch.occupied)
            for group, tail_rows in batch._row_padding:
                spec = self._schema.tensors[group]
                padding = (0, 0, 0, tail_rows)
                required_known = group in self._schema.view.required_known_empty_groups
                optional_known = group in self._schema.view.optional_known_empty_groups
                padding_value = 0 if required_known or optional_known else spec.padding
                values[group] = torch.nn.functional.pad(values[group], padding, value=padding_value)
                tail = batch._known_padding.get(group)
                if tail is None:
                    known[group] = torch.nn.functional.pad(
                        known[group], padding,
                        value=required_known,
                    )
                else:
                    known[group] = torch.cat((known[group], tail), dim=1)
                occupied[group] = torch.nn.functional.pad(occupied[group], (0, tail_rows))
            return EncodedBatch(
                batch.schema_fingerprint, values, known, occupied,
                batch.action_mask, batch.known_coverage,
            )

        def forward(self, batch: EncodedBatch) -> PolicyOutput:
            if not isinstance(batch, EncodedBatch):
                raise ContractError("CombatPolicy.forward requires an EncodedBatch")
            if batch.schema_fingerprint != self._schema.schema_fingerprint:
                raise ContractError("EncodedBatch schema fingerprint does not match this policy")
            if batch._row_padding and not experimental_trim_compute:
                batch = self._restore_row_capacity(batch)
            base = self._base_group_embeddings(batch)
            relational = self._reference_enrichment(batch, base)
            entity_rows = self._power_owner_enrichment(batch, relational)
            global_context = self._global_context(batch, entity_rows)
            source, source_app, source_known, target, target_app, target_known = (
                self._candidate_references(batch, entity_rows)
            )
            action_embeddings = entity_rows["actions"]
            broadcast_global = global_context.unsqueeze(1).expand(
                -1, action_embeddings.shape[1], -1
            )
            candidate_input = torch.cat((
                broadcast_global,
                action_embeddings,
                source,
                target,
                source_app.to(torch.float32).unsqueeze(-1),
                source_known.to(torch.float32).unsqueeze(-1),
                target_app.to(torch.float32).unsqueeze(-1),
                target_known.to(torch.float32).unsqueeze(-1),
            ), dim=-1)
            candidate = torch.tanh(self.candidate_first(candidate_input))
            candidate = torch.tanh(self.candidate_second(candidate))
            logits = self.candidate_output(candidate).squeeze(-1)
            if experimental_trim_compute and logits.shape[1] < batch.action_mask.shape[1]:
                # Preserve the public dispatch width after computing only the occupied prefix.
                logits = torch.nn.functional.pad(logits, (0, batch.action_mask.shape[1] - logits.shape[1]))
            values = self.value_head(global_context).squeeze(-1)
            if not torch.isfinite(values).all().item():
                raise ContractError("critic values must be finite")
            aux_values = None
            if self._value_heads > 1:
                aux_values = self.aux_value_head(global_context)
                if not torch.isfinite(aux_values).all().item():
                    raise ContractError("auxiliary value heads must be finite")
            applicable_logits = logits.masked_select(batch.action_mask)
            if applicable_logits.numel() and not torch.isfinite(applicable_logits).all().item():
                raise ContractError("legal action logits must be finite")
            return PolicyOutput(logits, values, aux_values)

        def act(
            self,
            frames: Sequence[CombatFrame],
            *,
            generator: Any,
            deterministic: bool = False,
        ) -> tuple[Any, ...]:
            if type(deterministic) is not bool:
                raise ContractError("deterministic must be a boolean")
            from spire_learn.neural.distribution import masked_categorical, pad_to_dispatch_width
            from spire_learn.neural.contracts import PolicyChoice

            with torch.no_grad():
                batch = encode_frames(
                    frames, self._schema, device=str(next(self.parameters()).device),
                    trim_padding=self._trim_padding,
                )
                output = self.forward(batch)
                if not torch.isfinite(output.values).all().item():
                    raise ContractError("critic values must be finite before action selection")
                # A compact layout samples at the dense dispatch width (illegal padding), so the
                # generator is consumed exactly as for the dense view of the same state.
                logits, mask = pad_to_dispatch_width(
                    output.logits, batch.action_mask, self._schema.view.dense_action_width)
                distribution = masked_categorical(logits, mask)
                actions = distribution.mode() if deterministic else distribution.sample(generator=generator)
                log_probs = distribution.log_prob(actions)
                return tuple(
                    PolicyChoice(
                        original_action_index=int(actions[index].item()),
                        log_probability=float(log_probs[index].item()),
                        value=float(output.values[index].item()),
                        behavior_policy_version=self._behavior_policy_version,
                    )
                    for index in range(len(frames))
                )

        def values(self, frames: Sequence[CombatFrame]) -> tuple[float, ...]:
            with torch.no_grad():
                batch = encode_frames(
                    frames, self._schema, device=str(next(self.parameters()).device),
                    trim_padding=self._trim_padding,
                )
                output = self.forward(batch)
                return tuple(float(value.item()) for value in output.values)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        policy = CombatPolicy()
    policy = policy.to(device=device)
    actual_count = sum(parameter.numel() for parameter in policy.parameters())
    if actual_count != plan.parameter_count:
        raise ContractError(
            f"compiled parameter count mismatch: expected {plan.parameter_count}, got {actual_count}"
        )
    return policy


__all__ = ["MODEL_ID", "PolicyOutput", "build_policy"]
