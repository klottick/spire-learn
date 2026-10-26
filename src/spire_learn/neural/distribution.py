"""Strict masked categorical distributions over original action rows."""

from __future__ import annotations

from typing import Any

from spire_learn.contracts import ContractError


class MaskedCategorical:
    """Categorical query object whose sampling method draws one action per row."""

    def __init__(self, logits: Any, action_mask: Any, torch: Any):
        self._torch = torch
        self._logits = logits
        self._mask = action_mask.to(dtype=torch.bool)
        masked_logits = logits.masked_fill(~self._mask, -torch.inf)
        self._log_probs = torch.log_softmax(masked_logits, dim=-1)
        self._probabilities = self._log_probs.exp()
        safe_log_probs = self._log_probs.masked_fill(~self._mask, 0.0)
        self._entropy = -(self._probabilities * safe_log_probs).sum(dim=-1)

    @property
    def probabilities(self) -> Any:
        return self._probabilities

    @property
    def log_probs(self) -> Any:
        return self._log_probs

    @property
    def entropy(self) -> Any:
        return self._entropy

    def log_prob(self, actions: Any) -> Any:
        torch = self._torch
        if not torch.is_tensor(actions):
            raise ContractError("actions must be an int64 tensor")
        if actions.dtype != torch.int64 or tuple(actions.shape) != (self._logits.shape[0],):
            raise ContractError("actions must be an int64 vector with one index per row")
        if actions.device != self._logits.device:
            raise ContractError("actions and logits must be on the same device")
        action_count = self._logits.shape[1]
        if not torch.all((actions >= 0) & (actions < action_count)).item():
            raise ContractError("action index is outside the original dispatch space")
        rows = torch.arange(self._logits.shape[0], device=self._logits.device)
        legal = self._mask[rows, actions]
        if not torch.all(legal).item():
            raise ContractError("action index is illegal under the supplied action mask")
        return self._log_probs[rows, actions]

    def sample(self, *, generator: Any) -> Any:
        """Draw exactly one original dispatch index for every batch row."""
        sampled = self._torch.multinomial(
            self._probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        )
        return sampled.squeeze(-1).to(dtype=self._torch.int64)

    def mode(self) -> Any:
        """Return the best legal original index; argmax keeps the smallest tie."""
        return self._logits.masked_fill(~self._mask, -self._torch.inf).argmax(dim=-1).to(
            dtype=self._torch.int64
        )


def masked_categorical(logits: Any, action_mask: Any) -> MaskedCategorical:
    """Validate a strict action mask and build stable legal-only statistics."""
    import torch

    if not torch.is_tensor(logits):
        raise ContractError("logits must be a tensor")
    if logits.dtype not in {torch.float32, torch.float64}:
        raise ContractError("logits must use float32 or float64")
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ContractError("logits must have non-empty [batch, action] shape")
    if not torch.is_tensor(action_mask):
        raise ContractError("action_mask must be a boolean or int8 tensor")
    if action_mask.dtype not in {torch.bool, torch.int8}:
        raise ContractError("action_mask dtype must be boolean or int8")
    if tuple(action_mask.shape) != tuple(logits.shape):
        raise ContractError("action_mask shape must exactly match logits")
    if action_mask.device != logits.device:
        raise ContractError("action_mask and logits must be on the same device")
    if action_mask.dtype == torch.int8 and not torch.all(
        (action_mask == 0) | (action_mask == 1)
    ).item():
        raise ContractError("int8 action_mask values must be binary")
    mask = action_mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=-1)).item():
        raise ContractError("every ongoing row must have at least one legal action")
    if not torch.isfinite(logits.masked_select(mask)).all().item():
        raise ContractError("legal action logits must be finite")
    return MaskedCategorical(logits, mask, torch)


__all__ = ["MaskedCategorical", "masked_categorical"]
