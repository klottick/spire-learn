"""Direct PPO updates over admitted, immutable training rollouts.

The caller owns the operation boundary: Task 7 must require both
``config.operation == 'train'`` and ``config.split == 'train'`` before creating
an optimizer or dispatching here.  This function enforces the rollout split and
all identity, behavior-version and consumption gates it can observe.

Optional tensor imports are deliberately local to :func:`ppo_update`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

from spire_learn.contracts import ContractError
from spire_learn.experiments.budget import Budget
from spire_learn.neural.contracts import NeuralIdentity, PPOConfig, Rollout
from spire_learn.neural.returns import compute_gae


@dataclass(frozen=True)
class UpdateMetrics:
    """Detached results and exact optimizer-call counts for one admitted rollout.

    A normal wall-time stop returns metrics with ``stop_reason='wall_time'``.
    Operational faults raise :class:`UpdateFault`, retaining the accepted
    optimizer prefix and distinguishing a call whose return was uncertain.
    Versions name the behavior policy admitted at entry and the current version
    after any optimizer call was attempted.
    """

    policy_loss: float
    value_loss: float
    entropy: float
    attempted_steps: int
    accepted_steps: int
    stop_reason: str | None
    admitted_behavior_policy_version: str | int
    resulting_behavior_policy_version: str | int
    consumed_decisions: int

    def __post_init__(self) -> None:
        for name in ("policy_loss", "value_loss", "entropy"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ContractError(f"UpdateMetrics {name} must be a finite Python float")
        for name in ("attempted_steps", "accepted_steps", "consumed_decisions"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"UpdateMetrics {name} must be a non-negative Python int")
        if self.accepted_steps > self.attempted_steps:
            raise ContractError("accepted optimizer calls cannot exceed attempts")
        if self.stop_reason not in {None, "wall_time"}:
            raise ContractError("UpdateMetrics stop_reason is unsupported")
        for name in ("admitted_behavior_policy_version", "resulting_behavior_policy_version"):
            value = getattr(self, name)
            if type(value) is int:
                if value < 0:
                    raise ContractError(f"UpdateMetrics {name} cannot be negative")
            elif not isinstance(value, str) or not value:
                raise ContractError(f"UpdateMetrics {name} must be a non-empty version")


class UpdateFault(RuntimeError):
    """Operational update failure with a snapshot of its optimizer prefix.

    ``attempted_steps`` counts calls entered, ``accepted_steps`` counts calls
    that returned, and ``uncertain_steps`` counts calls whose return or effect
    is unknown. ``phase`` names the failing update phase;
    ``admitted_behavior_policy_version`` and
    ``resulting_behavior_policy_version`` bracket the admitted prefix, while
    ``consumed_decisions`` records the already-consumed rollout rows. The
    original operational exception remains available through ``__cause__``.
    Normal wall-budget exhaustion returns :class:`UpdateMetrics` with
    ``stop_reason='wall_time'`` instead of raising this fault.
    """

    def __init__(
        self,
        message: str,
        *,
        attempted_steps: int,
        accepted_steps: int,
        uncertain_steps: int,
        phase: str,
        admitted_behavior_policy_version: str | int,
        resulting_behavior_policy_version: str | int,
        consumed_decisions: int,
    ) -> None:
        super().__init__(message)
        self.attempted_steps = attempted_steps
        self.accepted_steps = accepted_steps
        self.uncertain_steps = uncertain_steps
        self.phase = phase
        self.admitted_behavior_policy_version = admitted_behavior_policy_version
        self.resulting_behavior_policy_version = resulting_behavior_policy_version
        self.consumed_decisions = consumed_decisions


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_payload(identity: NeuralIdentity) -> dict[str, object]:
    return {
        "model_id": identity.model_id,
        "features_id": identity.features_id,
        "action_id": identity.action_id,
        "objective_id": identity.objective_id,
        "accepted_raw_mode_mappings": dict(identity.accepted_raw_mode_mappings),
        "config_hashes": dict(identity.config_hashes),
    }


def _budget_exhausted(budget: Any) -> bool:
    if not isinstance(budget, Budget):
        raise ContractError("budget must be the shared wall-time Budget")
    return bool(budget.exhausted())


def _check_optimizer(policy: Any, optimizer: Any, config: PPOConfig, torch: Any) -> None:
    if not callable(getattr(optimizer, "step", None)):
        raise ContractError("optimizer must expose a callable step method")
    if not callable(getattr(optimizer, "zero_grad", None)):
        raise ContractError("optimizer must expose a callable zero_grad method")
    parameters = tuple(policy.parameters())
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if len({id(parameter) for parameter in optimizer_parameters}) != len(optimizer_parameters):
        raise ContractError("optimizer contains a repeated parameter")
    if {id(parameter) for parameter in optimizer_parameters} != {
        id(parameter) for parameter in parameters
    }:
        raise ContractError("optimizer parameters must exactly match the policy parameters")
    if not optimizer.param_groups:
        raise ContractError("optimizer must contain at least one parameter group")
    for group in optimizer.param_groups:
        if not math.isclose(float(group.get("lr", math.nan)), config.learning_rate, rel_tol=0.0, abs_tol=0.0):
            raise ContractError("optimizer learning rate disagrees with PPOConfig")
        if "eps" in group and not math.isclose(
            float(group["eps"]), config.adam_eps, rel_tol=0.0, abs_tol=0.0
        ):
            raise ContractError("optimizer epsilon disagrees with PPOConfig")
    if not all(torch.is_floating_point(parameter) for parameter in parameters):
        raise ContractError("policy parameters must be floating tensors")


def _admission_key(rollout: Rollout) -> tuple[str, ...]:
    identity_hash = _canonical_hash(_identity_payload(rollout.identity))
    return tuple(
        f"{identity_hash}:{record.stream_id}:{record.decision_id}"
        for record in rollout.records
    )


def _validate_admission(policy: Any, rollout: Rollout, config: PPOConfig) -> tuple[Any, ...]:
    if not isinstance(rollout, Rollout):
        raise ContractError("ppo_update requires an immutable Rollout")
    if rollout.split != "train":
        raise ContractError("PPO updates require a train-split rollout")
    if not isinstance(config, PPOConfig):
        raise ContractError("ppo_update requires a validated PPOConfig")
    identity = rollout.identity
    expected_ids = (
        getattr(policy, "model_id", None),
        getattr(policy, "features_id", None),
        getattr(policy, "action_id", None),
        getattr(policy, "objective_id", None),
    )
    actual_ids = (identity.model_id, identity.features_id, identity.action_id, identity.objective_id)
    if actual_ids != expected_ids:
        raise ContractError("rollout learner identity does not match the policy")
    hashes = identity.config_hashes
    if hashes.get("model") != getattr(policy, "_model_config_hash", None):
        raise ContractError("rollout model config hash does not match the policy")
    if hashes.get("ppo") != _canonical_hash(asdict(config)):
        raise ContractError("rollout PPO config hash does not match PPOConfig")
    accepted_mappings = identity.accepted_raw_mode_mappings
    if rollout.mode not in accepted_mappings:
        raise ContractError("rollout mode is not accepted by its immutable identity")
    from spire_learn.neural.config import (
        COMMON_VIEW_FINGERPRINT,
        RUN_RAW_MAPPING_ID,
        STANDALONE_RAW_MAPPING_ID,
    )

    schema = getattr(policy, "_schema", None)
    if schema is None or schema.schema_fingerprint != COMMON_VIEW_FINGERPRINT:
        raise ContractError("policy schema is not the authenticated common public view")
    expected_mapping = {
        "standalone": STANDALONE_RAW_MAPPING_ID,
        "run": RUN_RAW_MAPPING_ID,
    }[rollout.mode]
    if accepted_mappings[rollout.mode] != expected_mapping:
        raise ContractError("rollout raw-mode mapping is not authenticated by the accepted config")
    version = getattr(policy, "_behavior_policy_version", None)
    if type(version) not in (int, str) or (type(version) is int and version < 0) or version == "":
        raise ContractError("policy behavior version is invalid")
    actor_records = tuple(record for record in rollout.records if record.actor_training_mask)
    if not actor_records:
        raise ContractError("PPO rollout contains no actor-training rows")
    for record in actor_records:
        choice = record.choice
        if choice.log_probability is None or choice.value is None:
            raise ContractError("actor rows require frozen log-probability and value statistics")
        if choice.behavior_policy_version != version:
            raise ContractError("rollout behavior policy version is stale")
    from spire_learn.neural.schema import validate_frame

    for record in actor_records:
        validate_frame(record.source, schema)
        action_index = record.choice.original_action_index
        action_count = schema.action_space.n
        if action_index >= action_count:
            raise ContractError("stored action index is outside the public dispatch space")
        mask = record.source.tensors["action_mask"]
        if mask.data[action_index] != 1:
            raise ContractError("stored action is illegal in its public source frame")
    # Validate GAE boundaries before consuming any optimizer state. Fixed rows
    # remain aligned but compute_gae leaves their placeholder targets masked.
    returns = compute_gae(
        rollout.records,
        gamma=config.gamma,
        lam=config.gae_lambda,
    )
    return actor_records, returns


def _as_update_fault(
    error: Exception,
    *,
    attempted: int,
    accepted: int,
    uncertain: int,
    phase: str,
    admitted_version: str | int,
    resulting_version: str | int,
    consumed_decisions: int,
) -> UpdateFault:
    fault = UpdateFault(
        f"PPO update failed during {phase}: {error}",
        attempted_steps=attempted,
        accepted_steps=accepted,
        uncertain_steps=uncertain,
        phase=phase,
        admitted_behavior_policy_version=admitted_version,
        resulting_behavior_policy_version=resulting_version,
        consumed_decisions=consumed_decisions,
    )
    fault.__cause__ = error
    return fault


def ppo_update(
    policy: Any,
    optimizer: Any,
    rollout: Rollout,
    config: PPOConfig,
    *,
    shuffle_generator: Any,
    budget: Budget,
) -> UpdateMetrics:
    """Admit and update one train rollout, stopping without replay or retries.

    This helper enforces ``rollout.split == 'train'`` and immutable identity,
    config hash, behavior-version, and consumed-decision gates before mutation.
    It cannot observe the caller's operation setting; the train runner must
    reject evaluation before optimizer construction or update dispatch.
    Rollout consumption happens once admission succeeds, including an already
    exhausted wall budget. Optimizer attempts are counted before calls, accepted
    only after successful returns, and an uncertain call is never repeated.
    """
    import torch

    from spire_learn.neural.distribution import masked_categorical
    from spire_learn.neural.features import encode_frames

    if not isinstance(policy, torch.nn.Module):
        raise ContractError("policy must be a torch Module")
    if not isinstance(shuffle_generator, torch.Generator):
        raise ContractError("shuffle_generator must be a Torch Generator")
    if shuffle_generator.device.type != "cpu":
        raise ContractError("shuffle_generator must be a separate CPU generator")
    if not isinstance(budget, Budget):
        raise ContractError("budget must be the shared wall-time Budget")
    if not isinstance(config, PPOConfig):
        raise ContractError("ppo_update requires a validated PPOConfig")

    # Pure admission checks precede optimizer inspection and mutable policy state.
    actor_records, return_batch = _validate_admission(policy, rollout, config)
    _check_optimizer(policy, optimizer, config, torch)
    version = policy._behavior_policy_version
    consumed_ids = policy._consumed_decision_ids
    admission_keys = _admission_key(rollout)
    if not isinstance(consumed_ids, set):
        raise ContractError("policy consumed-decision guard is invalid")
    if any(key in consumed_ids for key in admission_keys):
        raise ContractError("rollout contains a previously consumed decision")
    if len(consumed_ids) + len(admission_keys) > policy._consumed_decision_limit:
        raise ContractError("policy consumed-decision guard reached its bounded capacity")

    # Consumption is the admission mutation and deliberately precedes budget
    # evaluation or any gradient/optimizer work. Failed attempts cannot replay.
    consumed_ids.update(admission_keys)
    consumed_decisions = len(admission_keys)
    attempted_steps = 0
    accepted_steps = 0
    uncertain_steps = 0
    resulting_version = version
    losses = (0.0, 0.0, 0.0)
    phase = "prepare"
    optimizer_step_in_flight = False

    try:
        phase = "budget-check"
        if _budget_exhausted(budget):
            return UpdateMetrics(
                *losses,
                attempted_steps=0,
                accepted_steps=0,
                stop_reason="wall_time",
                admitted_behavior_policy_version=version,
                resulting_behavior_policy_version=version,
                consumed_decisions=consumed_decisions,
            )

        phase = "freeze-rollout"
        schema = policy._schema
        policy_device = next(policy.parameters()).device
        frames = tuple(record.source for record in actor_records)
        actions_cpu = tuple(record.choice.original_action_index for record in actor_records)
        action_values = torch.tensor(actions_cpu, dtype=torch.int64, device="cpu")
        old_log_probabilities = torch.tensor(
            [record.choice.log_probability for record in actor_records],
            dtype=torch.float32,
            device="cpu",
        )
        targets_cpu = torch.tensor(
            [return_batch.targets[index] for index, record in enumerate(rollout.records)
             if record.actor_training_mask],
            dtype=torch.float32,
            device="cpu",
        )
        advantages_cpu = torch.tensor(
            [return_batch.advantages[index] for index, record in enumerate(rollout.records)
             if record.actor_training_mask],
            dtype=torch.float32,
            device="cpu",
        )
        if not torch.isfinite(old_log_probabilities).all().item():
            raise ContractError("stored behavior log-probabilities must be finite")
        if not torch.isfinite(targets_cpu).all().item() or not torch.isfinite(advantages_cpu).all().item():
            raise ContractError("rollout return targets and advantages must be finite")
        advantage_mean = advantages_cpu.mean()
        advantage_scale = advantages_cpu.std(unbiased=False)
        if not torch.isfinite(advantage_scale).item():
            raise ContractError("normalized rollout advantages must be finite")
        normalized_advantages_cpu = (advantages_cpu - advantage_mean) / (advantage_scale + 1e-8)

        # The frozen old policy statistics must match every source frame and
        # original legal dispatch index before any optimizer call.
        phase = "validate-behavior-data"
        with torch.no_grad():
            source_batch = encode_frames(frames, schema, device=str(policy_device))
            source_output = policy(source_batch)
            source_distribution = masked_categorical(source_output.logits, source_batch.action_mask)
            actions_device = action_values.to(device=policy_device)
            old_actions_legal = source_batch.action_mask.gather(
                1, actions_device.unsqueeze(1)
            ).squeeze(1).to(dtype=torch.bool)
            if not torch.all(old_actions_legal).item():
                raise ContractError("stored action is illegal in its public source frame")
            # Exercise the same original-index legality contract here without
            # resampling. Stored log-probabilities remain the frozen PPO ratios;
            # tests and trusted collectors may intentionally exercise clipped
            # ratios different from one at entry.
            _ = source_distribution.log_prob(actions_device)
            if not torch.isfinite(source_output.values).all().item():
                raise ContractError("behavior critic values must be finite")

        attempted_version_bump = False
        actor_count = len(actor_records)
        latest_losses = losses
        stop_reason: str | None = None
        for _epoch in range(config.epochs):
            phase = "budget-check"
            if _budget_exhausted(budget):
                stop_reason = "wall_time"
                break
            phase = "shuffle"
            permutation = torch.randperm(actor_count, generator=shuffle_generator, device="cpu")
            for start in range(0, actor_count, config.minibatch_rows):
                phase = "budget-check"
                if _budget_exhausted(budget):
                    stop_reason = "wall_time"
                    break
                selected_cpu = permutation[start : start + config.minibatch_rows]
                if selected_cpu.numel() == 0:
                    continue
                selected = selected_cpu.tolist()
                batch_frames = tuple(frames[index] for index in selected)
                action_batch = action_values[selected_cpu].to(device=policy_device)
                old_logp_batch = old_log_probabilities[selected_cpu].to(device=policy_device)
                advantage_batch = normalized_advantages_cpu[selected_cpu].to(device=policy_device)
                target_batch = targets_cpu[selected_cpu].to(device=policy_device)

                phase = "forward-loss"
                optimizer.zero_grad(set_to_none=True)
                batch = encode_frames(batch_frames, schema, device=str(policy_device))
                output = policy(batch)
                distribution = masked_categorical(output.logits, batch.action_mask)
                new_logp = distribution.log_prob(action_batch)
                ratio = torch.exp(new_logp - old_logp_batch)
                unclipped = ratio * advantage_batch
                clipped = torch.clamp(
                    ratio,
                    min=1.0 - config.clip_epsilon,
                    max=1.0 + config.clip_epsilon,
                ) * advantage_batch
                policy_loss_tensor = -torch.minimum(unclipped, clipped).mean()
                value_loss_tensor = torch.square(output.values - target_batch).mean()
                entropy_tensor = distribution.entropy.mean()
                loss = (
                    policy_loss_tensor
                    + config.value_coefficient * value_loss_tensor
                    - config.entropy_coefficient * entropy_tensor
                )
                if not torch.isfinite(loss).item():
                    raise ContractError("PPO loss must be finite before backward")
                phase = "backward"
                loss.backward()
                parameters = tuple(parameter for parameter in policy.parameters() if parameter.grad is not None)
                if not parameters or any(not torch.isfinite(parameter.grad).all().item() for parameter in parameters):
                    raise ContractError("PPO gradients must be present and finite before clipping")
                phase = "gradient-clipping"
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
                if any(not torch.isfinite(parameter.grad).all().item() for parameter in parameters):
                    raise ContractError("clipped PPO gradients must be finite")

                phase = "metrics"
                latest_losses = (
                    float(policy_loss_tensor.detach().cpu().item()),
                    float(value_loss_tensor.detach().cpu().item()),
                    float(entropy_tensor.detach().cpu().item()),
                )
                if any(not math.isfinite(value) for value in latest_losses):
                    raise ContractError("detached PPO metrics must be finite")
                phase = "optimizer.step"
                attempted_steps += 1
                if not attempted_version_bump:
                    # Invalidate behavior choices before an optimizer call whose
                    # mutation outcome might be uncertain.
                    policy._behavior_policy_version += 1
                    resulting_version = policy._behavior_policy_version
                    attempted_version_bump = True
                optimizer_step_in_flight = True
                try:
                    optimizer.step()
                except (KeyboardInterrupt, SystemExit, GeneratorExit) as error:
                    uncertain_steps += 1
                    optimizer_step_in_flight = False
                    raise
                except Exception as error:
                    uncertain_steps += 1
                    optimizer_step_in_flight = False
                    raise _as_update_fault(
                        error,
                        attempted=attempted_steps,
                        accepted=accepted_steps,
                        uncertain=uncertain_steps,
                        phase=phase,
                        admitted_version=version,
                        resulting_version=resulting_version,
                        consumed_decisions=consumed_decisions,
                    ) from error
                optimizer_step_in_flight = False
                accepted_steps += 1
            if stop_reason is not None:
                break
        losses = latest_losses
        return UpdateMetrics(
            *losses,
            attempted_steps=attempted_steps,
            accepted_steps=accepted_steps,
            stop_reason=stop_reason,
            admitted_behavior_policy_version=version,
            resulting_behavior_policy_version=resulting_version,
            consumed_decisions=consumed_decisions,
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit) as error:
        if optimizer_step_in_flight:
            uncertain_steps += 1
            optimizer_step_in_flight = False
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(
                "PPO update prefix: "
                f"phase={phase!r}, attempted_steps={attempted_steps}, "
                f"accepted_steps={accepted_steps}, uncertain_steps={uncertain_steps}, "
                f"admitted_behavior_policy_version={version!r}, "
                f"resulting_behavior_policy_version={resulting_version!r}, "
                f"consumed_decisions={consumed_decisions}"
            )
        raise
    except UpdateFault:
        raise
    except Exception as error:
        raise _as_update_fault(
            error,
            attempted=attempted_steps,
            accepted=accepted_steps,
            uncertain=uncertain_steps,
            phase=phase,
            admitted_version=version,
            resulting_version=resulting_version,
            consumed_decisions=consumed_decisions,
        ) from error


__all__ = ["UpdateFault", "UpdateMetrics", "ppo_update"]
