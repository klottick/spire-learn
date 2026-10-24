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
