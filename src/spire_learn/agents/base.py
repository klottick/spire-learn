"""Public-input policy interface and decision boundary validation."""

from typing import Protocol

from ..contracts import Choice, ContractError, DecisionRequest, DecisionResult, PublicDecision


class Policy(Protocol):
    def init_state(self) -> object | None: ...

    def act(self, public: PublicDecision, state: object | None) -> Choice: ...


def _legal_indices(public: PublicDecision) -> tuple[int, ...]:
    if len(public.mask) == 0:
        raise ContractError("decision mask cannot be empty")
    if any(type(legal) is not bool for legal in public.mask):
        raise ContractError("decision mask entries must be booleans")
    indices = tuple(index for index, legal in enumerate(public.mask) if legal)
    for candidate in public.candidates:
        if type(candidate.index) is not int or not 0 <= candidate.index < len(public.mask):
            raise ContractError("candidate index must be an in-range integer")
        if candidate.kind == "padding":
            raise ContractError("padding cannot be a decision candidate")
    candidate_indices = tuple(candidate.index for candidate in public.candidates)
    if (len(set(candidate_indices)) != len(candidate_indices)
            or set(candidate_indices) != set(indices)):
        raise ContractError("candidates must exactly describe current legal indices")
    return indices


def validate_result(request: DecisionRequest, result: DecisionResult) -> None:
    if not isinstance(result, DecisionResult) or not isinstance(result.choice, Choice):
        raise ContractError("policy result must contain a Choice")
    if result.decision_id != request.route.decision_id:
        raise ContractError("result decision ID does not match request")
    action = result.choice.action
    if type(action) is not int:
        raise ContractError("action must be an integer index")
    if action not in _legal_indices(request.public):
        raise ContractError("action is not a current legal candidate")


def resolve_decision(request: DecisionRequest, policy: Policy, state: object | None) -> DecisionResult:
    choice = policy.act(request.public, state)
    result = DecisionResult(request.route.decision_id, choice)
    validate_result(request, result)
    return result
