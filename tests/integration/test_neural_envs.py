"""Opt-in public Gym conformance; never runs unless explicitly gated."""

from __future__ import annotations

import os
from importlib.metadata import version
import hashlib
from functools import wraps
import json
from pathlib import Path
import random
import struct
import time
import unittest
from dataclasses import replace

from spire_learn.neural.cases import load_neural_cases
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.envs import (
    CombatDriver,
    DriverFault,
    DriverRow,
    DriverStateError,
    make_driver,
)
from spire_learn.neural.schema import CombatCandidate


_NATIVE_ENABLED = os.environ.get("SPIRE_LEARN_RUN_NEURAL_NATIVE") == "1"
_SOFT_SECONDS = 120.0
_HARD_SECONDS = 300.0
_TOTAL_SLOT_ACTIONS = 4096
_PER_FIGHT_SLOT_ACTIONS = 1024
_RUNTIME_IDENTITY_CHECKED = False


def _require_runtime_identity() -> None:
    global _RUNTIME_IDENTITY_CHECKED
    if _RUNTIME_IDENTITY_CHECKED:
        return
    from spire_lab import _native

    expected_versions = {
        "spire-lab": "0.1.0",
        "gymnasium": "1.4.0",
        "numpy": "2.5.4",
        "torch": "2.14.1+cpu",
    }
    for distribution, expected in expected_versions.items():
        actual = version(distribution)
        if actual != expected:
            raise AssertionError(f"runtime package {distribution}={actual}, expected {expected}")
    identity = _native.build_info()
    if identity.get("revision") != "533dad1c8aec86df624b3742b6b3ad1dbf1d0033":
        raise AssertionError("loaded native build revision is not the accepted Gym pin")
    if identity.get("extension") != "spire-python":
        raise AssertionError("loaded native extension identity is not spire-python")
    native_path = Path(_native.__file__)
    digest = hashlib.sha256(native_path.read_bytes()).hexdigest()
    expected_digest = "071069cea66fbf0bc7277002c6b8a21f138194cf692326595109a6741a5178e9"
    if digest != expected_digest:
        raise AssertionError("loaded native binary hash differs from the reviewed CPU runtime")
    _RUNTIME_IDENTITY_CHECKED = True


class _InvocationBudget:
    """Shared operation, action and lifecycle counters for the three-test run."""

    def __init__(self):
        started = os.environ.get("SPIRE_LEARN_TASK3_STARTED_MONOTONIC")
        self.started = float(started) if started is not None else time.monotonic()
        self.slot_actions = 0
        self.delivered_slot_actions = 0
        self.uncertain_slot_actions = 0
        self.step_requests = 0
        self.delivered_step_calls = 0
        self.rejected_pre_dispatch_calls = 0
        self.reset_calls = 0
        self.reset_selected_slots = 0
        self.delivered_reset_calls = 0
        self.driver_constructions = 0
        self.close_calls = 0
        self.close_failures = 0
        self.automatic_fault_close_attempts = 0
        self.cohort_fault_latched = False
        self.cohort_fault_type: str | None = None
        self.test_names: list[str] = []
        self.catalog_splits: list[str] = []
        self.lifecycle_rows = {"active": 0, "inactive": 0}
        self.natural_settlements = 0
        self.exit_outcomes = {"victory": 0, "death": 0}
        self.exit_outcomes_by_mode = {
            "standalone": {"victory": 0, "death": 0},
            "run": {"victory": 0, "death": 0},
        }
        self.max_fight_slot_actions = 0
        self.fight_actions: dict[tuple[int, int, int, int], int] = {}
        self.generations: dict[tuple[int, int], int] = {}
        self.soft_limit_elapsed: float | None = None
        self.max_elapsed_at_operation_check = 0.0

    def _elapsed(self) -> float:
        elapsed = time.monotonic() - self.started
        self.max_elapsed_at_operation_check = max(self.max_elapsed_at_operation_check, elapsed)
        return elapsed

    def _ensure_cohort_healthy(self) -> None:
        if self.cohort_fault_latched:
            raise AssertionError(
                "Task3 cohort fault latch is set; no more Gym constructors, resets or actions are allowed"
            )

    def before_constructor(self) -> None:
        self._ensure_cohort_healthy()

    def before_reset(self) -> None:
        self._ensure_cohort_healthy()
        elapsed = self._elapsed()
        if elapsed >= _SOFT_SECONDS:
            self.soft_limit_elapsed = elapsed
            raise AssertionError(
                f"INCONCLUSIVE: soft cap stopped before reset at elapsed={elapsed:.3f}s; no rerun"
            )
        if elapsed >= _HARD_SECONDS:
            raise AssertionError(
                f"INCONCLUSIVE: whole-run cap stopped before reset at elapsed={elapsed:.3f}s; no rerun"
            )

    def note_reset(self, driver: CombatDriver, mask: tuple[bool, ...]) -> None:
        self.delivered_reset_calls += 1
        for slot, selected in enumerate(mask):
            if selected:
                key = (id(driver), slot)
                self.generations[key] = self.generations.get(key, 0) + 1
        self.note_rows(driver.rows)

    def begin_reset(self, mask: tuple[bool, ...]) -> None:
        self.reset_calls += 1
        self.reset_selected_slots += sum(mask)

    def note_rows(self, rows) -> None:
        for row in rows:
            if row is not None:
                self.lifecycle_rows["active" if row.lifecycle.active else "inactive"] += 1

    def before_step(self, driver: CombatDriver, actions: tuple[int, ...]):
        self._ensure_cohort_healthy()
        elapsed = self._elapsed()
        if elapsed >= _SOFT_SECONDS:
            self.soft_limit_elapsed = elapsed
            raise AssertionError(
                f"INCONCLUSIVE: soft cap stopped before action at elapsed={elapsed:.3f}s; no rerun"
            )
        if elapsed >= _HARD_SECONDS:
            raise AssertionError(
                f"INCONCLUSIVE: whole-run cap stopped before action at elapsed={elapsed:.3f}s; no rerun"
            )
        if self.slot_actions + len(actions) > _TOTAL_SLOT_ACTIONS:
            raise AssertionError("INCONCLUSIVE: 4096 total native slot-action cap reached")
        current = driver.rows
        next_counts: list[tuple[tuple[int, int, int, int], int]] = []
        for slot, _action in enumerate(actions):
            row = current[slot]
            if row is None or not row.lifecycle.active:
                continue
            generation = self.generations.get((id(driver), slot), 0)
            visit = row.lifecycle.active_visit
            if visit is None:
                continue
            key = (id(driver), slot, generation, visit)
            count = self.fight_actions.get(key, 0) + 1
            if count > _PER_FIGHT_SLOT_ACTIONS:
                raise AssertionError("INCONCLUSIVE: 1024 action cap reached for one natural fight")
            next_counts.append((key, count))
        self.step_requests += 1
        return tuple(next_counts)

    def note_step_delivered(self, driver: CombatDriver, actions: tuple[int, ...], transitions, pending) -> None:
        self.delivered_step_calls += 1
        self.slot_actions += len(actions)
        self.delivered_slot_actions += len(actions)
        self.fight_actions.update(pending)
        if pending:
            self.max_fight_slot_actions = max(
                self.max_fight_slot_actions,
                max(count for _key, count in pending),
            )
        for transition in transitions:
            before = transition.source.lifecycle.combats_settled
            after = transition.successor.lifecycle.combats_settled
            if after > before:
                self.natural_settlements += after - before
                exit_info = transition.lifecycle.last_exit
                if exit_info is not None:
                    outcome = exit_info.get("outcome")
                    if outcome in self.exit_outcomes:
                        self.exit_outcomes[outcome] += 1
                        self.exit_outcomes_by_mode[driver.mode][outcome] += 1
        self.note_rows(tuple(item.successor for item in transitions))

    def note_pre_dispatch_rejection(self) -> None:
        self.rejected_pre_dispatch_calls += 1

    def note_fault(self, error: BaseException) -> None:
        self.cohort_fault_latched = True
        self.cohort_fault_type = type(error).__name__
        if isinstance(error, DriverFault):
            evidence = error.evidence
            if evidence.phase != "close":
                self.automatic_fault_close_attempts += 1
                if evidence.close_error is not None:
                    self.close_failures += 1
            if evidence.phase.startswith("step"):
                known = evidence.known_current_delivery_slots
                uncertain = evidence.uncertain_current_slots
                self.slot_actions += known + uncertain
                self.delivered_slot_actions += known
                self.uncertain_slot_actions += uncertain

    def note_close(self, driver: CombatDriver) -> None:
        self.close_calls += 1
        try:
            driver.close()
        except BaseException as error:
            self.close_failures += 1
            self.note_fault(error)
            raise

    def report(self, *, tests_run: int, tests_skipped: int) -> dict[str, object]:
        elapsed = time.monotonic() - self.started
        return {
            "native_enabled": _NATIVE_ENABLED,
            "tests_expected": 3,
            "tests_run": tests_run,
            "tests_skipped": tests_skipped,
            "test_names": list(self.test_names),
            "elapsed_seconds": round(elapsed, 6),
            "soft_limit_seconds": _SOFT_SECONDS,
            "soft_limit_stopped_before_operation_at_seconds": self.soft_limit_elapsed,
            "soft_limit_overshoot_seconds": round(max(0.0, elapsed - _SOFT_SECONDS), 6),
            "max_elapsed_at_operation_check_seconds": round(self.max_elapsed_at_operation_check, 6),
            "hard_limit_seconds": _HARD_SECONDS,
            "action_budget": {
                "slot_actions_cap": _TOTAL_SLOT_ACTIONS,
                "per_fight_slot_actions_cap": _PER_FIGHT_SLOT_ACTIONS,
                "max_fight_slot_actions": self.max_fight_slot_actions,
                "slot_actions_upper_bound": self.slot_actions,
                "delivered_slot_actions": self.delivered_slot_actions,
                "uncertain_slot_actions": self.uncertain_slot_actions,
                "step_requests": self.step_requests,
                "delivered_step_calls": self.delivered_step_calls,
                "pre_dispatch_rejections": self.rejected_pre_dispatch_calls,
            },
            "reset_counters": {
                "calls": self.reset_calls,
                "selected_slots": self.reset_selected_slots,
                "delivered_calls": self.delivered_reset_calls,
            },
            "lifecycle_counters": {
                "rows_observed": dict(self.lifecycle_rows),
                "natural_settlements": self.natural_settlements,
                "exit_outcomes": dict(self.exit_outcomes),
                "exit_outcomes_by_mode": self.exit_outcomes_by_mode,
            },
            "close_counters": {
                "driver_close_api_calls": self.close_calls,
                "driver_close_api_failures": self.close_failures,
                "automatic_fault_close_attempts": self.automatic_fault_close_attempts,
            },
            "driver_constructions": self.driver_constructions,
            "public_catalog_splits_validated": list(self.catalog_splits),
            "cohort_fault_latched": self.cohort_fault_latched,
            "cohort_fault_type": self.cohort_fault_type,
        }


_BUDGET = _InvocationBudget()


def _config(name: str):
    return load_neural_config(Path("configs/neural") / name)


def _case_by_entry(config, entry_id: str):
    for case in config.cases.for_mode("standalone"):
        if case.entry_id == entry_id:
            return case
    raise AssertionError(f"conformance manifest has no entry {entry_id}")


def _reset(driver: CombatDriver, seeds, entry_ids, mask=None):
    selected = (True,) * driver.batch_size if mask is None else tuple(mask)
    _BUDGET.before_reset()
    _BUDGET.begin_reset(selected)
    try:
        rows = driver.reset(seeds=seeds, entry_ids=entry_ids, reset_mask=mask)
    except BaseException as error:
        _BUDGET.note_fault(error)
        raise
    _BUDGET.note_reset(driver, selected)
    return rows


def _make_driver(config, cases) -> CombatDriver:
    _BUDGET.before_constructor()
    try:
        driver = make_driver(config, cases)
    except BaseException as error:
        _BUDGET.note_fault(error)
        raise
    _BUDGET.driver_constructions += 1
    return driver


def _close_driver(driver: CombatDriver) -> None:
    _BUDGET.note_close(driver)


def _card_for_candidate(row: DriverRow, candidate: CombatCandidate, driver: CombatDriver) -> str | None:
    source = candidate.source
    if source is None or source.group != "combat_cards" or source.row is None:
        return None
    spec = driver.schema.tensors.get("combat_cards")
    if spec is None:
        raise AssertionError("accepted common descriptor has no combat_cards tensor")
    column_index = next(
        (index for index, column in enumerate(spec.columns) if column.vocabulary == "card"),
        None,
    )
    if column_index is None:
        raise AssertionError("accepted combat_cards descriptor has no public card vocabulary")
    buffer = row.frame.tensors["combat_cards"]
    width = len(spec.columns)
    flat_index = source.row * width + column_index
    value = struct.unpack_from("=q", buffer.data, flat_index * 8)[0]
    return driver.schema.vocabularies["card"][value]


def _end_turn_or_first(row: DriverRow) -> int:
    for candidate in row.candidates:
        if candidate.verb == "end_turn":
            return candidate.index
    return row.candidates[0].index


def _combat_action(row: DriverRow, driver: CombatDriver, *, prefer_attack: bool) -> int:
    if not row.lifecycle.active:
        return row.noncombat_indices[0]
    if row.lifecycle.routing == "combat_selection":
        for verb in ("confirm_selection", "combat_select_card"):
            candidate = min(
                (item for item in row.candidates if item.verb == verb),
                key=lambda item: item.index,
                default=None,
            )
            if candidate is not None:
                return candidate.index
        raise AssertionError(
            "public combat selector has no legal confirm_selection or combat_select_card candidate"
        )
    if prefer_attack:
        for candidate in row.candidates:
            if candidate.verb == "play_card":
                return candidate.index
    return _end_turn_or_first(row)


def _dispatch(driver: CombatDriver, actions: tuple[int, ...]):
    pending = _BUDGET.before_step(driver, actions)
    try:
        transitions = driver.step(actions)
    except BaseException as error:
        _BUDGET.note_fault(error)
        raise
    _BUDGET.note_step_delivered(driver, actions, transitions, pending)
    return transitions


def _validate_authored_catalog_splits() -> None:
    """Ask Gym to validate every authored standalone catalog without resetting it."""
    train_config = _config("standalone.toml")
    for split in ("train", "selection", "final"):
        cases = load_neural_cases(Path("configs/cases/neural") / f"{split}.json")
        config = replace(
            train_config,
            split=split,
            partition=split,
            operation=("train" if split == "train" else "evaluate"),
            manifest_path=cases.path,
            cases=cases,
            manifest_sha256=cases.sha256,
            final_requires_opt_in=(split == "final"),
        )
        driver = _make_driver(config, cases)
        try:
            expected_rows = (None,) * config.rollout_rows
            if (
                driver.mode != "standalone"
                or driver.batch_size != config.rollout_rows
                or driver.rows != expected_rows
            ):
                raise AssertionError(
                    f"{split} catalog factory reset, changed mode, or returned the wrong row width"
                )
        finally:
            _close_driver(driver)
        _BUDGET.catalog_splits.append(split)


def _write_machine_evidence() -> None:
    destination = os.environ.get("SPIRE_LEARN_TASK3_EVIDENCE_PATH")
    if not destination:
        return
    tests_run = len(_BUDGET.test_names)
    tests_skipped = 0 if _NATIVE_ENABLED else 3
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_BUDGET.report(tests_run=tests_run, tests_skipped=tests_skipped),
                   sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _latch_test_failure(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException as error:
            _BUDGET.note_fault(error)
            raise
    return guarded


class NeuralEnvNativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        if _NATIVE_ENABLED:
            _BUDGET.test_names.append(self._testMethodName)

    @classmethod
    def tearDownClass(cls):
        _write_machine_evidence()

    @unittest.skipUnless(_NATIVE_ENABLED, "set SPIRE_LEARN_RUN_NEURAL_NATIVE=1 in the protected runbook")
    @_latch_test_failure
    def test_standalone_public_profile(self):
        _BUDGET.before_constructor()
        _require_runtime_identity()
        _validate_authored_catalog_splits()
        config = _config("task3-conformance-standalone.toml")
        driver = _make_driver(config, config.cases)
        try:
            self.assertEqual(driver.mode, "standalone")
            self.assertEqual(driver.schema.schema_fingerprint, config.common_view_fingerprint)
            probe = _case_by_entry(config, "task3-warcry-elixir")
            ongoing = _case_by_entry(config, "task3-mixed-ongoing")
            rows = _reset(
                driver,
                [probe.environment_seed, ongoing.environment_seed],
                [probe.entry_id, ongoing.entry_id],
            )
            self.assertTrue(all(row.lifecycle.active for row in rows))
            selectors = {"warcry": False, "elixir": False}
            elixir_used = False
            for _ in range(256):
                actions: list[int] = []
                for slot, row in enumerate(rows):
                    if slot == 1:
                        actions.append(_combat_action(row, driver, prefer_attack=False))
                        continue
                    if row.lifecycle.routing == "combat_selection":
                        if row.lifecycle.selection_owner == "card":
                            selectors["warcry"] = True
                        if row.lifecycle.selection_owner == "potion":
                            selectors["elixir"] = True
                        actions.append(_combat_action(row, driver, prefer_attack=False))
                        continue
                    potion = next((
                        candidate for candidate in row.candidates
                        if candidate.verb == "use_potion"
                        and candidate.source is not None
                        and candidate.source.group == "potions"
                        and candidate.source.row == 0
                    ), None)
                    if not elixir_used and potion is not None:
                        elixir_used = True
                        actions.append(potion.index)
                        continue
                    warcry = next((
                        candidate for candidate in row.candidates
                        if candidate.verb == "play_card"
                        and _card_for_candidate(row, candidate, driver) == "warcry"
                    ), None)
                    if not selectors["warcry"] and warcry is not None:
                        actions.append(warcry.index)
                        continue
                    actions.append(_combat_action(row, driver, prefer_attack=True))
                transitions = _dispatch(driver, tuple(actions))
                rows = tuple(item.successor for item in transitions)
                for slot, transition in enumerate(transitions):
                    if slot != 0:
                        continue
                    if transition.lifecycle.selection_owner == "card":
                        selectors["warcry"] = True
                    elif transition.lifecycle.selection_owner == "potion":
                        selectors["elixir"] = True
                if all(selectors.values()):
                    break
                if any(item.terminated or item.truncated for item in transitions):
                    break
            self.assertTrue(selectors["warcry"], "INCONCLUSIVE: Warcry selector not reached within the fixed probe cap")
            self.assertTrue(selectors["elixir"], "INCONCLUSIVE: Elixir selector not reached within the fixed probe cap")
            self.assertLessEqual(_BUDGET.slot_actions, _TOTAL_SLOT_ACTIONS)
        finally:
            _close_driver(driver)

    @unittest.skipUnless(_NATIVE_ENABLED, "set SPIRE_LEARN_RUN_NEURAL_NATIVE=1 in the protected runbook")
    @_latch_test_failure
    def test_manual_vector_mixed_reset(self):
        _BUDGET.before_constructor()
        _require_runtime_identity()
        config = _config("task3-conformance-standalone.toml")
        driver = _make_driver(config, config.cases)
        try:
            winner = _case_by_entry(config, "task3-mixed-winner")
            survivor = _case_by_entry(config, "task3-mixed-ongoing")
            rows = _reset(
                driver,
                [winner.environment_seed, survivor.environment_seed],
                [winner.entry_id, survivor.entry_id],
            )
            transitions = None
            for _ in range(_PER_FIGHT_SLOT_ACTIONS):
                actions = tuple(
                    _combat_action(row, driver, prefer_attack=(slot == 0))
                    for slot, row in enumerate(rows)
                )
                transitions = _dispatch(driver, actions)
                rows = tuple(item.successor for item in transitions)
                terminal_slots = [
                    slot for slot, item in enumerate(transitions)
                    if item.terminated or item.truncated
                ]
                if terminal_slots:
                    break
            else:
                self.fail("INCONCLUSIVE: no natural terminal row within 1024 actions; do not rerun automatically")
            self.assertIsNotNone(transitions)
            self.assertEqual(len(terminal_slots), 1, "strong supported entry must finish before the ongoing row")
            finished = terminal_slots[0]
            ongoing_slot = 1 - finished
            self.assertTrue(transitions[finished].terminated)
            self.assertFalse(transitions[ongoing_slot].terminated)
            accepted_before_rejection = driver.accepted_steps
            actions_before_rejection = _BUDGET.slot_actions
            rejected_actions = tuple(
                row.candidates[0].index if row.candidates else
                (row.noncombat_indices[0] if row.noncombat_indices else 0)
                for row in rows
            )
            _BUDGET.before_step(driver, rejected_actions)
            with self.assertRaisesRegex(DriverStateError, "terminal row"):
                driver.step(rejected_actions)
            _BUDGET.note_pre_dispatch_rejection()
            self.assertEqual(driver.accepted_steps, accepted_before_rejection)
            self.assertEqual(_BUDGET.slot_actions, actions_before_rejection)
            ongoing_snapshot = transitions[ongoing_slot].successor
            before_frame = ongoing_snapshot.frame
            before_lifecycle = ongoing_snapshot.lifecycle
            mask = tuple(slot == finished for slot in range(driver.batch_size))
            entries = [winner.entry_id, survivor.entry_id]
            seeds = [winner.environment_seed + 20, survivor.environment_seed + 20]
            rows = _reset(driver, seeds, entries, mask)
            self.assertIs(rows[ongoing_slot], ongoing_snapshot)
            self.assertEqual(rows[ongoing_slot].frame, before_frame)
            self.assertEqual(rows[ongoing_slot].lifecycle, before_lifecycle)
            self.assertTrue(rows[ongoing_slot].lifecycle.active)
            self.assertIs(driver.rows[ongoing_slot], ongoing_snapshot)
            next_actions = tuple(
                _combat_action(row, driver, prefer_attack=(slot == finished))
                for slot, row in enumerate(rows)
            )
            next_transitions = _dispatch(driver, next_actions)
            self.assertIs(next_transitions[ongoing_slot].source, ongoing_snapshot)
            self.assertEqual(next_transitions[ongoing_slot].source.frame, before_frame)
            self.assertLessEqual(_BUDGET.slot_actions, _TOTAL_SLOT_ACTIONS)
        finally:
            _close_driver(driver)

    @unittest.skipUnless(_NATIVE_ENABLED, "set SPIRE_LEARN_RUN_NEURAL_NATIVE=1 in the protected runbook")
    @_latch_test_failure
    def test_normal_run_common_view(self):
        _BUDGET.before_constructor()
        _require_runtime_identity()
        config = _config("task3-conformance-run.toml")
        driver = _make_driver(config, config.cases)
        try:
            run_case = config.cases.for_mode("run")[0]
            rows = _reset(driver, [run_case.environment_seed], [None])
            rng = random.Random(config.seed_streams.noncombat)
            saw_combat = False
            settlement = None
            for _ in range(2048):
                row = rows[0]
                legal = (
                    tuple(candidate.index for candidate in row.candidates)
                    if row.lifecycle.active else row.noncombat_indices
                )
                if not legal:
                    self.fail("INCONCLUSIVE: Run ended before a natural combat exit; do not rerun automatically")
                action = rng.choice(legal)
                transition = _dispatch(driver, (action,))[0]
                rows = (transition.successor,)
                current = transition.lifecycle
                saw_combat = saw_combat or current.active or current.combats_entered > 0
                if current.combats_settled > row.lifecycle.combats_settled:
                    settlement = transition
                    break
                if transition.terminated or transition.truncated:
                    break
            if settlement is None:
                self.fail("INCONCLUSIVE: no natural combat exit within the fixed Run cap; do not rerun automatically")
            self.assertTrue(saw_combat)
            checkpoint = settlement.successor
            self.assertIsNotNone(checkpoint.lifecycle.last_exit)
            self.assertEqual(
                checkpoint.lifecycle.last_exit["outcome"],
                "victory",
                "INCONCLUSIVE: fixed legal-random Run died before victory; no seed search or reinterpretation",
            )
            for key in ("player", "permanent_cards", "relics", "potions"):
                self.assertIn(key, checkpoint.frame.tensors)
            self.assertFalse(settlement.terminated, "Run must continue after the first combat boundary")
            self.assertTrue(checkpoint.noncombat_indices, "settlement must expose the next public Run action mask")
            following = rng.choice(checkpoint.noncombat_indices)
            after = _dispatch(driver, (following,))[0]
            self.assertIs(after.source, checkpoint)
            self.assertEqual(after.source.frame, checkpoint.frame)
            self.assertEqual(after.lifecycle.combats_settled, checkpoint.lifecycle.combats_settled)
            self.assertEqual(after.lifecycle.last_exit, checkpoint.lifecycle.last_exit)
            self.assertLessEqual(_BUDGET.slot_actions, _TOTAL_SLOT_ACTIONS)
        finally:
            _close_driver(driver)


if __name__ == "__main__":
    unittest.main()
