import random
import unittest
from dataclasses import replace

from spire_learn.agents.base import resolve_decision, validate_result
from spire_learn.agents.baselines import PlayFirstPolicy, RandomPolicy
from spire_learn.contracts import Choice, ContractError, DecisionRequest, DecisionResult, Route
from tests.support import public_decision


def request(public=None):
    return DecisionRequest(public or public_decision(mask=(1, 0, 1), kinds=(1, 0, 2)),
                           Route("exec", 0, "episode", "decision", True))


class BaselineTests(unittest.TestCase):
    def test_random_policy_matches_independent_seeded_legal_sequence(self):
        policy, expected = RandomPolicy(19), random.Random(19)
        view = request().public
        self.assertIsNone(policy.init_state())
        self.assertEqual([policy.act(view, None).action for _ in range(8)],
                         [expected.choice((0, 2)) for _ in range(8)])

    def test_play_first_uses_lowest_legal_kind(self):
        policy = PlayFirstPolicy()
        self.assertEqual(policy.act(request(public_decision(mask=(1, 1, 1, 1), kinds=(1, 2, 1, 2))).public, None).action, 1)
        self.assertEqual(policy.act(request(public_decision(mask=(1, 1, 0), kinds=(1, 1, 0))).public, None).action, 0)

    def test_empty_nonterminal_mask_errors(self):
        for policy in (RandomPolicy(2), PlayFirstPolicy()):
            with self.assertRaises(ContractError):
                policy.act(public_decision(mask=(0, 0, 0), kinds=(0, 0, 0)), None)

    def test_result_rejects_bool_illegal_padding_and_stale_id(self):
        req = request()
        for choice, decision_id in ((Choice(True), "decision"), (Choice(1.0), "decision"),
                                    (Choice(-1), "decision"), (Choice(99), "decision"),
                                    (Choice(0), "stale")):
            with self.subTest(choice=choice, decision_id=decision_id), self.assertRaises(ContractError):
                validate_result(req, DecisionResult(decision_id, choice))
        with self.assertRaises(ContractError):
            validate_result(req, DecisionResult("decision", None))
        # Index 0 remains legal in this decision; a result from an older turn
        # must still fail based on decision identity, not merely action legality.
        with self.assertRaises(ContractError):
            validate_result(req, DecisionResult("older-turn", Choice(0)))

    def test_true_mask_padding_candidate_is_rejected_at_every_boundary(self):
        req = request()
        padding = replace(req.public.candidates[0], kind="padding")
        invalid_public = replace(req.public, candidates=(padding, req.public.candidates[1]))
        invalid_request = replace(req, public=invalid_public)

        for policy in (RandomPolicy(19), PlayFirstPolicy()):
            with self.subTest(policy=type(policy).__name__), self.assertRaises(ContractError):
                policy.act(invalid_public, None)
        with self.assertRaises(ContractError):
            validate_result(invalid_request, DecisionResult("decision", Choice(0)))

    def test_candidate_indices_must_be_exact_integers(self):
        public = public_decision(mask=(1, 1, 0), kinds=(1, 2, 0))
        for malformed_index in (True, 1.0):
            candidate = replace(public.candidates[1], index=malformed_index)
            malformed = replace(public, candidates=(public.candidates[0], candidate))
            with self.subTest(index=malformed_index), self.assertRaises(ContractError):
                RandomPolicy(4).act(malformed, None)

    def test_rejected_random_decision_does_not_advance_policy_stream(self):
        invalid = request()
        padding = replace(invalid.public.candidates[0], kind="padding")
        invalid_public = replace(invalid.public, candidates=(padding, invalid.public.candidates[1]))
        policy, fresh = RandomPolicy(71), RandomPolicy(71)
        with self.assertRaises(ContractError):
            policy.act(invalid_public, None)
        valid = request().public
        self.assertEqual(policy.act(valid, None), fresh.act(valid, None))

    def test_policy_receives_no_route_or_seed(self):
        class Spy:
            def init_state(self): return None
            def act(self, public, state):
                self.arguments = (public, state)
                return Choice(0)
        policy, req = Spy(), request()
        result = resolve_decision(req, policy, "opaque")
        self.assertEqual(policy.arguments, (req.public, "opaque"))
        self.assertEqual(result.decision_id, "decision")


if __name__ == "__main__":
    unittest.main()
