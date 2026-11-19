"""Potential-based HP-shaped objective: pure arithmetic, config, collector and identity."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import random
import struct
import tempfile
from types import SimpleNamespace
import unittest

from spire_learn.contracts import ContractError
from spire_learn.experiments.budget import Budget
from spire_learn.neural.checkpoint import CheckpointCompatibilityError, validate_checkpoint_metadata
from spire_learn.neural.config import (
    HP_SHAPED_OBJECTIVE_ID,
    WINLOSS_OBJECTIVE_ID,
    HpShapedObjective,
    load_neural_config,
)
from spire_learn.neural.contracts import BufferArray, CombatFrame, Lifecycle, PolicyChoice
from spire_learn.neural.envs import DriverRow, DriverTransition
from spire_learn.neural.returns import hp_potential, shaped_reward, task_signal
from spire_learn.neural.schema import CombatCandidate, parse_combat_descriptor
from spire_learn.neural.gym_contracts import CURRENT_CONTRACT


ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = parse_combat_descriptor(
    json.loads((ROOT / "tests/fixtures/neural/common-afb51f5.json").read_text()),
    revision=CURRENT_CONTRACT.source_revision,
)
_ENEMIES = _SCHEMA.tensors["enemies"]
_COLUMNS = tuple(column.field for column in _ENEMIES.columns)
_ROWS = _ENEMIES.shape[0]


def _int64(values, shape):
    return BufferArray("int64", shape, struct.pack(f"={len(values)}q", *values))


def _int8(values, shape):
    return BufferArray("int8", shape, bytes(values))


def enemy_frame(enemies, *, player=0, schema=_SCHEMA):
    """Frame with one enemies row per (hp, max_hp[, hp_known, max_known]) item.

    ``None`` is an unoccupied padded row. Column positions come from the schema.
    """
    spec = schema.tensors["enemies"]
    rows, columns = spec.shape
    fields = tuple(column.field for column in spec.columns)
    hp_index, max_index = fields.index("hp"), fields.index("max_hp")
    values = [0] * (rows * columns)
    known = [0] * (rows * columns)
    occupied = [0] * rows
    for row_index, item in enumerate(enemies):
        if item is None:
            continue
        hp, maximum, *flags = item
        hp_known, max_known = flags if flags else (True, True)
        occupied[row_index] = 1
        for column in range(columns):
            known[row_index * columns + column] = 1
        values[row_index * columns + hp_index] = hp if hp_known else 0
        values[row_index * columns + max_index] = maximum if max_known else 0
        known[row_index * columns + hp_index] = int(hp_known)
        known[row_index * columns + max_index] = int(max_known)
    tensors = {
        "player": _int8((player,), (1,)),
        "enemies": _int64(values, (rows, columns)),
        spec.known_key: _int8(known, (rows, columns)),
        spec.occupied_key: _int8(occupied, (rows,)),
    }
    return CombatFrame(tensors, schema.schema_fingerprint)


class HpPotentialTests(unittest.TestCase):
    def test_potential_ignores_unoccupied_and_zero_max_rows(self):
        frame = enemy_frame(((100, 400), (0, 0), None))
        self.assertEqual(hp_potential(frame, _SCHEMA), -0.25)

    def test_potential_sums_qualifying_rows(self):
        frame = enemy_frame(((100, 400), None, (300, 400)))
        self.assertEqual(hp_potential(frame, _SCHEMA), -0.5)

    def test_potential_excludes_rows_with_unknown_hp_or_max_hp(self):
        frame = enemy_frame(((100, 400), (5, 10, False, True), (5, 10, True, False)))
        self.assertEqual(hp_potential(frame, _SCHEMA), -0.25)

    def test_potential_is_zero_when_no_row_qualifies(self):
        self.assertEqual(hp_potential(enemy_frame(()), _SCHEMA), 0.0)
        self.assertEqual(hp_potential(enemy_frame(((0, 0), None)), _SCHEMA), 0.0)
        self.assertEqual(hp_potential(enemy_frame(((4, 10, False, False),)), _SCHEMA), 0.0)

    def test_potential_reads_columns_by_descriptor_name(self):
        self.assertNotEqual(_COLUMNS.index("hp"), 0)
        frame = enemy_frame(((100, 400),))
        values = struct.unpack(f"={_ROWS * len(_COLUMNS)}q", frame.tensors["enemies"].data)
        self.assertEqual(values[_COLUMNS.index("hp")], 100)
        self.assertEqual(values[_COLUMNS.index("max_hp")], 400)
        self.assertEqual(hp_potential(frame, _SCHEMA), -0.25)

    def test_potential_requires_the_public_enemy_tensors(self):
        with self.assertRaises(ContractError):
            hp_potential(CombatFrame({"player": _int8((0,), (1,))},
                                     _SCHEMA.schema_fingerprint), _SCHEMA)
        with self.assertRaises(ContractError):
            hp_potential(enemy_frame(((1, 2),)), SimpleNamespace(tensors={}))


class ShapedRewardTests(unittest.TestCase):
    def test_shaped_reward_terminal_potential_is_zero(self):
        source = enemy_frame(((100, 400),))
        successor = enemy_frame(((50, 400),))
        shaped = shaped_reward(1.0, source, successor, terminal=True, gamma=0.99,
                               beta=0.5, schema=_SCHEMA)
        self.assertEqual(shaped, 1.0 + 0.5 * (0.0 - (-0.25)))

    def test_shaped_reward_damage_step(self):
        source = enemy_frame(((400, 400),))
        successor = enemy_frame(((300, 400),))
        shaped = shaped_reward(0.0, source, successor, terminal=False, gamma=0.99,
                               beta=0.5, schema=_SCHEMA)
        self.assertEqual(shaped, 0.0 + 0.5 * (0.99 * (-0.75) - (-1.0)))

    def test_shaped_reward_rejects_invalid_parameters(self):
        frame = enemy_frame(((1, 2),))
        for gamma, beta in ((1.5, 0.5), (math.nan, 0.5), (0.99, 0.0), (0.99, 1.5),
                            (0.99, math.inf), (True, 0.5)):
            with self.subTest(gamma=gamma, beta=beta):
                with self.assertRaises(ValueError):
                    shaped_reward(0.0, frame, frame, terminal=False, gamma=gamma,
                                  beta=beta, schema=_SCHEMA)
        with self.assertRaises(ValueError):
            shaped_reward(0.0, frame, frame, terminal="no", gamma=0.99, beta=0.5,
                          schema=_SCHEMA)


# ---------------------------------------------------------------------------
# Config, collector and identity.

_STANDALONE = ROOT / "configs/neural/standalone.toml"
_HP_TABLE = '\n[objective]\nkind = "hp_shaped"\ncoefficient = {coefficient}\n'


def _write_config(directory: Path, *, objective_id=WINLOSS_OBJECTIVE_ID, table=None,
                  source: Path = _STANDALONE) -> Path:
    text = source.read_text().replace(
        'manifest_path = "../cases/neural/train.json"',
        f'manifest_path = "{(ROOT / "configs/cases/neural/train.json").resolve()}"',
    ).replace(
        'output_root = "../../artifacts/development/neural-combat/runs/standalone"',
        f'output_root = "{(ROOT / "artifacts/development/neural-combat/runs/shaped-test").resolve()}"',
    ).replace(
        f'objective_id = "{WINLOSS_OBJECTIVE_ID}"', f'objective_id = "{objective_id}"',
    )
    if table is not None:
        text += table
    path = directory / "config.toml"
    path.write_text(text)
    return path


class ObjectiveConfigTests(unittest.TestCase):
    def test_configs_without_objective_table_stay_winloss(self):
        config = load_neural_config(_STANDALONE)
        self.assertIsNone(config.objective)
        self.assertEqual(config.accepted_identities[0].objective_id, WINLOSS_OBJECTIVE_ID)

    def test_hp_shaped_table_requires_the_hp_shaped_identity_and_vice_versa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = load_neural_config(_write_config(
                root, objective_id=HP_SHAPED_OBJECTIVE_ID,
                table=_HP_TABLE.format(coefficient="0.5")))
            self.assertEqual(accepted.objective, HpShapedObjective(coefficient=0.5))
            self.assertEqual(accepted.accepted_identities[0].objective_id, HP_SHAPED_OBJECTIVE_ID)
            for objective_id, table in (
                (WINLOSS_OBJECTIVE_ID, _HP_TABLE.format(coefficient="0.5")),
                (HP_SHAPED_OBJECTIVE_ID, None),
            ):
                with self.subTest(objective_id=objective_id, table=table):
                    with self.assertRaises(ValueError):
                        load_neural_config(_write_config(root, objective_id=objective_id, table=table))

    def test_objective_table_is_strict(self):
        invalid = (
            _HP_TABLE.format(coefficient="0.0"),
            _HP_TABLE.format(coefficient="-0.5"),
            _HP_TABLE.format(coefficient="1.5"),
            _HP_TABLE.format(coefficient="nan"),
            _HP_TABLE.format(coefficient="inf"),
            _HP_TABLE.format(coefficient='"0.5"'),
            _HP_TABLE.format(coefficient="true"),
            '\n[objective]\nkind = "winloss"\ncoefficient = 0.5\n',
            '\n[objective]\nkind = "hp_shaped"\n',
            '\n[objective]\nkind = "hp_shaped"\ncoefficient = 0.5\nextra = 1\n',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for table in invalid:
                with self.subTest(table=table):
                    with self.assertRaises(ValueError):
                        load_neural_config(_write_config(
                            root, objective_id=HP_SHAPED_OBJECTIVE_ID, table=table))
            config = load_neural_config(_write_config(
                root, objective_id=HP_SHAPED_OBJECTIVE_ID, table=_HP_TABLE.format(coefficient="1")))
            self.assertEqual(config.objective.coefficient, 1.0)
            self.assertIs(type(config.objective.coefficient), float)

    def test_cv2_hp_templates_load_and_roundtrip_through_stream_preparation(self):
        import tomllib
        from spire_learn.neural.stream_runner import _toml

        for suffix in ("", "-r2"):
            template = ROOT / f"configs/neural/baseline-gym-1.1.1-cv2-hp{suffix}/template.toml"
            with self.subTest(replicate=suffix), tempfile.TemporaryDirectory() as directory:
                config = load_neural_config(_write_config(Path(directory), source=template))
                self.assertEqual(config.objective, HpShapedObjective(coefficient=0.5))
                self.assertEqual({identity.objective_id for identity in config.accepted_identities},
                                 {HP_SHAPED_OBJECTIVE_ID})
                document = tomllib.loads(template.read_text())
                self.assertEqual(tomllib.loads(_toml(document)), document)
        with tempfile.TemporaryDirectory() as directory:
            cv2 = load_neural_config(_write_config(
                Path(directory), source=ROOT / "configs/neural/baseline-gym-1.1.1-cv2/template.toml"))
        self.assertIsNone(cv2.objective)

    def test_unknown_objective_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_neural_config(_write_config(Path(directory), objective_id="other-objective"))


def _lifecycle(*, active=True, settled=0, exit_data=None):
    return Lifecycle(active=active, routing="combat" if active else "noncombat",
                     selection_owner=None, active_visit=0 if active else None,
                     combats_entered=1, combats_settled=settled, last_exit=exit_data)


def _row(enemies, *, player, active=True, settled=0, outcome=None):
    exit_data = None if outcome is None else {
        "visit": 0, "encounter": "cultist", "outcome": outcome,
        "hp_before_healing": 51, "hp_after_cleanup": 57,
    }
    candidates = (CombatCandidate(0, "end_turn", None, None, None, False, False,
                                  None, False, False, None, False, False),) if active else ()
    return DriverRow(enemy_frame(enemies, player=player),
                     _lifecycle(active=active, settled=settled, exit_data=exit_data),
                     candidates, ())


class _Driver:
    mode = "standalone"
    batch_size = 2

    def __init__(self, initial, resets, steps, *, schema=None):
        self._rows = tuple(initial)
        self._resets = list(resets)
        self._steps = list(steps)
        if schema is not None:
            self.schema = schema

    @property
    def rows(self):
        return self._rows

    def reset(self, *, seeds, entry_ids, reset_mask=None):
        mask = tuple(reset_mask) if reset_mask is not None else (True,) * self.batch_size
        replacement = self._resets.pop(0)
        self._rows = tuple(replacement[slot] if selected else current
                           for slot, (selected, current) in enumerate(zip(mask, self._rows)))
        return self._rows

    def step(self, actions):
        result = self._steps.pop(0)(self._rows)
        self._rows = tuple(item.successor for item in result)
        return result

    def close(self):
        pass


class _Policy:
    def __init__(self, identity):
        self.model_id = identity.model_id
        self.features_id = identity.features_id
        self.action_id = identity.action_id
        self.objective_id = identity.objective_id

    def act(self, frames, *, generator, deterministic=False):
        return tuple(PolicyChoice(0, -0.25, 0.5, 0) for _ in frames)

    def values(self, frames):
        return tuple(frame.tensors["player"].data[0] * 0.1 for frame in frames)


def _hp_config(config, coefficient=0.5):
    identities = tuple(replace(identity, objective_id=HP_SHAPED_OBJECTIVE_ID)
                       for identity in config.accepted_identities)
    return replace(config, accepted_identities=identities,
                   objective=HpShapedObjective(coefficient=coefficient))


class ShapedCollectorTests(unittest.TestCase):
    def setUp(self):
        self.winloss = replace(load_neural_config(_STANDALONE), rollout_rows=2, rollout_steps=1)
        self.hp = _hp_config(self.winloss)

    def _script(self):
        a0 = _row(((400, 400),), player=1)
        b0 = _row(((300, 400),), player=2)
        a1 = _row(((300, 400),), player=3)
        b1 = _row(((0, 400),), player=4, active=False, settled=1, outcome="victory")
        b_reset = _row(((200, 200), (50, 100)), player=5)
        a2 = _row(((300, 400),), player=6)
        b2 = _row(((150, 200), (50, 100)), player=7)
        steps = (
            lambda rows: (DriverTransition(rows[0], a1, 0.0, False, False),
                          DriverTransition(rows[1], b1, 0.0, True, False)),
            lambda rows: (DriverTransition(rows[0], a2, 0.0, False, False),
                          DriverTransition(rows[1], b2, 0.0, False, False)),
        )
        return (a0, b0), [(b_reset, b_reset)], steps

    @staticmethod
    def _driver(initial, resets, steps, *, schema=None):
        # Fresh slots are reset first, as in production collection.
        return _Driver((None, None), [initial, *resets], steps, schema=schema)

    def _collect(self, config, *, schema):
        from spire_learn.neural.collector import Collector

        initial, resets, steps = self._script()
        driver = self._driver(initial, resets, steps, schema=schema)
        collector = Collector(
            driver, _Policy(config.accepted_identities[0]), config,
            controller_rng=random.Random(901), curriculum_rng=random.Random(1201),
            budget=Budget(60.0, lambda: 0.0), policy_generator=object(),
        )
        records = []
        for _ in range(2):
            rollout = collector.collect(max_learned_decisions=8, max_env_actions=4,
                                        policy_version=0)
            records.extend(rollout.records)
        collector.close()
        return records

    def test_winloss_configs_are_unchanged_without_objective(self):
        # No schema on the driver: the win/loss path never reads frames for reward.
        records = self._collect(self.winloss, schema=None)
        (a0, b0), [(b_reset, _)], steps = self._script()
        first = steps[0]((a0, b0))
        second = steps[1]((first[0].successor, b_reset))
        expected = tuple(
            task_signal(item.source.lifecycle, item.successor.lifecycle,
                        terminated=item.terminated, truncated=item.truncated)
            for item in (*first, *second)
        )
        self.assertEqual(tuple(record.task_signal for record in records), expected)
        self.assertEqual(tuple(record.task_signal.reward for record in records),
                         (0.0, 1.0, 0.0, 0.0))

    def test_hp_shaped_changes_only_the_reward(self):
        winloss = self._collect(self.winloss, schema=None)
        shaped = self._collect(self.hp, schema=_SCHEMA)
        self.assertEqual(len(winloss), len(shaped))
        for plain, record in zip(winloss, shaped, strict=True):
            self.assertEqual(
                (record.task_signal.task_terminal, record.task_signal.bootstrap,
                 record.task_signal.continuation, record.terminated, record.truncated,
                 record.next_value, record.source, record.successor, record.task_id),
                (plain.task_signal.task_terminal, plain.task_signal.bootstrap,
                 plain.task_signal.continuation, plain.terminated, plain.truncated,
                 plain.next_value, plain.source, plain.successor, plain.task_id))
        gamma, beta = self.hp.ppo.gamma, 0.5
        expected = (
            0.0 + beta * (gamma * -0.75 - -1.0),
            # Terminal successor: potential zero, natural victory reward kept.
            1.0 + beta * (0.0 - -0.75),
            0.0 + beta * (gamma * -0.75 - -0.75),
            # First transition after the reset shapes from the reset row itself.
            0.0 + beta * (gamma * -(200 / 300) - -(250 / 300)),
        )
        self.assertEqual(tuple(record.task_signal.reward for record in shaped), expected)

    def test_hp_shaped_collector_requires_a_driver_schema(self):
        from spire_learn.neural.collector import Collector

        initial, resets, steps = self._script()
        with self.assertRaises(ContractError):
            Collector(self._driver(initial, resets, steps), _Policy(self.hp.accepted_identities[0]),
                      self.hp, controller_rng=random.Random(1), curriculum_rng=random.Random(2),
                      budget=Budget(60.0, lambda: 0.0), policy_generator=object())

    def test_policy_identity_must_match_the_configured_objective(self):
        from spire_learn.neural.collector import Collector

        for config, policy_config in ((self.hp, self.winloss), (self.winloss, self.hp)):
            initial, resets, steps = self._script()
            with self.subTest(objective=config.accepted_identities[0].objective_id):
                with self.assertRaises(ContractError):
                    Collector(self._driver(initial, resets, steps, schema=_SCHEMA),
                              _Policy(policy_config.accepted_identities[0]), config,
                              controller_rng=random.Random(1), curriculum_rng=random.Random(2),
                              budget=Budget(60.0, lambda: 0.0), policy_generator=object())


class ObjectiveIdentityTests(unittest.TestCase):
    def _metadata(self, identity):
        from tests.unit.test_neural_checkpoint import _metadata

        return _metadata(identity=identity)

    def test_objective_ids_do_not_cross(self):
        from tests.unit.test_neural_checkpoint import _identity

        winloss = _identity()
        hp = replace(winloss, objective_id=HP_SHAPED_OBJECTIVE_ID)
        for saved, expected in ((hp, winloss), (winloss, hp)):
            for purpose in ("evaluate", "warm_start", "resume"):
                with self.subTest(saved=saved.objective_id, purpose=purpose):
                    with self.assertRaisesRegex(CheckpointCompatibilityError, "objective_id"):
                        validate_checkpoint_metadata(
                            self._metadata(saved), expected, purpose=purpose)
        validate_checkpoint_metadata(self._metadata(hp), hp, purpose="evaluate")

    def test_config_identities_carry_their_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            hp = load_neural_config(_write_config(
                Path(directory), objective_id=HP_SHAPED_OBJECTIVE_ID,
                table=_HP_TABLE.format(coefficient="0.5")))
        winloss = load_neural_config(_STANDALONE)
        from tests.unit.test_neural_checkpoint import _metadata

        with self.assertRaisesRegex(CheckpointCompatibilityError, "objective_id"):
            validate_checkpoint_metadata(_metadata(identity=hp.accepted_identities[0]),
                                         winloss.accepted_identities[0], purpose="evaluate")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "objective_id"):
            validate_checkpoint_metadata(_metadata(identity=winloss.accepted_identities[0]),
                                         hp.accepted_identities[0], purpose="evaluate")


if __name__ == "__main__":
    unittest.main()
