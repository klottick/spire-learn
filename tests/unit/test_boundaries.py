import unittest

from spire_learn.contracts import ContractError
from spire_learn.collection.boundaries import route_single_step, route_vector_step
from tests.support import observation, vector_step_fixture


def hp(observation_value):
    return observation_value["player"].values[0]


class BoundaryRoutingTests(unittest.TestCase):
    def test_single_step_preserves_flags(self):
        route, source, successor = vector_step_fixture()["routes"][0], observation(hp=12), observation(hp=7, mask=(0, 0, 0))
        result = route_single_step(
            route=route, source=source, action=0, reward=-1.0, terminated=True,
            truncated=True, successor=successor, state_after_source=7,
        )
        self.assertEqual(result.transition.successor, successor)
        self.assertTrue(result.transition.terminated)
        self.assertTrue(result.transition.truncated)
        self.assertIsNone(result.transition.replacement)
        self.assertEqual(result.active_observation, successor)
        self.assertFalse(result.episode_started)
        self.assertIsNone(result.active_state)

    def test_single_step_rejects_non_boolean_flags(self):
        args = vector_step_fixture()
        with self.assertRaises(ContractError):
            route_single_step(
                route=args["routes"][0], source=args["sources"][0], action=0,
                reward=0.0, terminated=1, truncated=False,
                successor=args["main_observations"][0], state_after_source=7,
            )

    def test_single_step_rejects_invalid_action_values(self):
        args = vector_step_fixture()
        for action in (True, -1):
            with self.subTest(action=action), self.assertRaises(ContractError):
                route_single_step(
                    route=args["routes"][0], source=args["sources"][0], action=action,
                    reward=0.0, terminated=False, truncated=False,
                    successor=args["main_observations"][0], state_after_source=7,
                )

    def test_both_helpers_reject_invalid_rewards(self):
        args = vector_step_fixture()
        for reward in (True, float("nan"), float("inf"), -float("inf"), 10**400):
            label = reward if type(reward) is bool else str(reward) if type(reward) is float else "oversized-int"
            with self.subTest(reward=label):
                with self.assertRaises(ContractError):
                    route_single_step(
                        route=args["routes"][0], source=args["sources"][0], action=0,
                        reward=reward, terminated=False, truncated=False,
                        successor=args["main_observations"][0], state_after_source=7,
                    )
                vector_args = vector_step_fixture()
                vector_args["rewards"] = (reward, 0.0)
                with self.assertRaises(ContractError):
                    route_vector_step(**vector_args)

    def test_routes_require_valid_slots_and_vector_slots_match_rows(self):
        args = vector_step_fixture()
        for slot in (True, -1):
            bad_route = args["routes"][0].__class__("fixture", slot, "e", "d", True)
            with self.subTest(slot=slot), self.assertRaises(ContractError):
                route_single_step(
                    route=bad_route, source=args["sources"][0], action=0,
                    reward=0.0, terminated=False, truncated=False,
                    successor=args["main_observations"][0], state_after_source=7,
                )
        valid_nonzero_route = args["routes"][0].__class__("fixture", 9, "e", "d", True)
        result = route_single_step(
            route=valid_nonzero_route, source=args["sources"][0], action=0,
            reward=0.0, terminated=False, truncated=False,
            successor=args["main_observations"][0], state_after_source=7,
        )
        self.assertEqual(result.transition.route.slot, 9)
        with self.assertRaises(ContractError):
            route_single_step(
                route=object(), source=args["sources"][0], action=0,
                reward=0.0, terminated=False, truncated=False,
                successor=args["main_observations"][0], state_after_source=7,
            )
        args["routes"] = (args["routes"][0], args["routes"][0])
        with self.assertRaises(ContractError):
            route_vector_step(**args)

    def test_single_step_wraps_malformed_observation_mapping_as_contract_error(self):
        args = vector_step_fixture()
        with self.assertRaises(ContractError):
            route_single_step(
                route=args["routes"][0], source=object(), action=0,
                reward=0.0, terminated=False, truncated=False,
                successor=args["main_observations"][0], state_after_source=7,
            )

    def test_vector_wraps_malformed_observation_before_reset_callbacks(self):
        args = vector_step_fixture()
        args.update(
            terminated=(True, True), final_observations=(observation(hp=10), observation(hp=8)),
            final_presence=(True, True), final_info_presence=(True, True), sources=(observation(hp=12), object()),
        )
        called = []
        args["reset_state"] = lambda slot: called.append(slot)
        with self.assertRaises(ContractError):
            route_vector_step(**args)
        self.assertEqual(called, [])

    def test_empty_vector_batch_is_rejected(self):
        args = vector_step_fixture()
        for key in (
            "routes", "sources", "actions", "rewards", "terminated", "truncated",
            "main_observations", "final_observations", "final_presence",
            "final_info_presence", "states_after_source",
        ):
            args[key] = ()
        with self.assertRaises(ContractError):
            route_vector_step(**args)

    def test_invalid_later_row_is_rejected_before_any_reset_callback(self):
        args = vector_step_fixture()
        args.update(
            terminated=(True, True), final_observations=(observation(hp=10), observation(hp=8)),
            final_presence=(True, True), final_info_presence=(True, True), actions=(0, True),
        )
        called = []
        args["reset_state"] = lambda slot: called.append(slot)
        with self.assertRaises(ContractError):
            route_vector_step(**args)
        self.assertEqual(called, [])

    def test_later_reset_callback_failure_propagates_without_retry(self):
        args = vector_step_fixture()
        args.update(
            terminated=(True, True), final_observations=(observation(hp=10), observation(hp=8)),
            final_presence=(True, True), final_info_presence=(True, True),
        )
        failure = RuntimeError("opaque state reset failed")
        called = []

        def reset_state(slot):
            called.append(slot)
            if slot == 1:
                raise failure
            return 0

        args["reset_state"] = reset_state
        with self.assertRaisesRegex(RuntimeError, "opaque state reset failed") as raised:
            route_vector_step(**args)
        self.assertIs(raised.exception, failure)
        self.assertEqual(called, [0, 1])

    def test_distinct_old_final_and_reset_observation_values_remain_separate(self):
        args = vector_step_fixture()
        old_final = observation(hp=10)
        reset_observation = observation(hp=80)
        args["final_observations"] = (old_final, None)
        args["main_observations"] = (reset_observation, args["main_observations"][1])
        row = route_vector_step(**args)[0]
        old_value = row.transition.successor["player"].values[0] + row.transition.state_after_source
        reset_value = row.active_observation["player"].values[0] + row.active_state
        self.assertEqual(row.transition.successor, old_final)
        self.assertEqual(row.transition.replacement, reset_observation)
        self.assertEqual(row.active_observation, reset_observation)
        self.assertEqual((old_value, reset_value), (17, 80))

    def test_vector_routes_final_and_replacement_with_old_state(self):
        reset_slots = []

        def reset_state(slot):
            reset_slots.append(slot)
            return slot

        args = vector_step_fixture()
        args["reset_state"] = reset_state
        rows = route_vector_step(**args)
        ended = rows[0]
        self.assertEqual(ended.transition.successor, args["final_observations"][0])
        self.assertEqual(hp(ended.transition.successor), 10)
        self.assertEqual(ended.transition.state_after_source, 7)
        self.assertEqual(ended.transition.replacement, args["main_observations"][0])
        self.assertEqual(ended.active_observation, args["main_observations"][0])
        self.assertNotEqual(ended.transition.successor, ended.active_observation)
        self.assertEqual(ended.active_state, 0)
        self.assertTrue(ended.episode_started)
        self.assertEqual(reset_slots, [0])

    def test_vector_live_row_preserves_state(self):
        args = vector_step_fixture()
        rows = route_vector_step(**args)
        self.assertEqual(rows[1].transition.successor, args["main_observations"][1])
        self.assertIsNone(rows[1].transition.replacement)
        self.assertIs(rows[1].active_state, args["states_after_source"][1])
        self.assertFalse(rows[1].episode_started)

    def test_missing_final_keys_allowed_when_no_row_ends(self):
        args = vector_step_fixture()
        args.update(terminated=(False, False), truncated=(False, False), final_observations=(None, None), final_presence=(False, False), final_info_presence=(False, False))
        rows = route_vector_step(**args)
        self.assertEqual(len(rows), 2)

    def test_presence_masks_must_match(self):
        args = vector_step_fixture()
        args["final_info_presence"] = (False, False)
        with self.assertRaises(ContractError):
            route_vector_step(**args)

    def test_terminal_empty_mask_is_not_acted_on(self):
        args = vector_step_fixture()
        result = route_single_step(
            route=args["routes"][0], source=args["sources"][0], action=0,
            reward=0.0, terminated=True, truncated=False,
            successor=observation(hp=10, mask=(0, 0, 0)), state_after_source=7,
        )
        self.assertEqual(result.active_observation["action_mask"].values, (0, 0, 0))
        self.assertFalse(result.episode_started)
        self.assertIsNone(result.active_state)

    def test_cut_does_not_reset_memory(self):
        args = vector_step_fixture()
        state = args["states_after_source"][1]
        result = route_single_step(
            route=args["routes"][1], source=args["sources"][1], action=1,
            reward=0.0, terminated=False, truncated=False,
            successor=args["main_observations"][1], state_after_source=state,
        )
        self.assertIs(result.active_state, state)
        self.assertIs(result.transition.state_after_source, state)
        self.assertFalse(result.episode_started)

    def test_vector_widths_and_presence_types_are_validated_before_callbacks(self):
        for key, value in (("actions", (0,)), ("final_presence", (True,)), ("terminated", (1, False))):
            args = vector_step_fixture()
            args[key] = value
            called = []
            args["reset_state"] = lambda slot: called.append(slot)
            with self.subTest(key=key), self.assertRaises(ContractError):
                route_vector_step(**args)
            self.assertEqual(called, [])

    def test_final_data_without_matching_presence_is_rejected(self):
        args = vector_step_fixture()
        args["final_observations"] = (None, None)
        with self.assertRaises(ContractError):
            route_vector_step(**args)

    def test_final_data_for_live_row_is_rejected(self):
        args = vector_step_fixture()
        args["final_observations"] = (args["final_observations"][0], observation(hp=4))
        with self.assertRaises(ContractError):
            route_vector_step(**args)


if __name__ == "__main__":
    unittest.main()
