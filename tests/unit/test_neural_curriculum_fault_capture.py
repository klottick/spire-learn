"""Pure unit coverage for bounded curriculum driver fault capture."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class FakeDriver:
    def __init__(self, *, reset_error=None, step_error=None, diagnostic=None):
        self.mode = "standalone"
        self.batch_size = 2
        self.reset_calls = []
        self.step_calls = []
        self.reset_error = reset_error
        self.step_error = step_error
        self.reset_result = object()
        self.step_result = object()
        self._provider = SimpleNamespace(_env=SimpleNamespace(diagnostic=diagnostic))

    def reset(self, *, seeds, entry_ids, reset_mask):
        self.reset_calls.append(((), {"seeds": seeds, "entry_ids": entry_ids,
                                      "reset_mask": reset_mask}))
        if self.reset_error is not None:
            raise self.reset_error
        return self.reset_result

    def step(self, actions):
        self.step_calls.append(((actions,), {}))
        if self.step_error is not None:
            raise self.step_error
        return self.step_result


def make_runtime(driver=None, *, batch_size=2):
    driver = driver or FakeDriver()
    cases = [SimpleNamespace(case_id="old-case", entry_id="old-entry"), None][:batch_size]
    seeds = [4101, None][:batch_size]
    collector = SimpleNamespace(driver=driver, _batch_size=batch_size,
                                _slot_cases=cases, _slot_seeds=seeds)
    config = SimpleNamespace(config_sha256="a" * 64, manifest_sha256="b" * 64)
    return SimpleNamespace(collector=collector, driver=driver, config=config)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CurriculumFaultCaptureTests(unittest.TestCase):
    def api(self):
        import importlib
        return importlib.import_module("scripts.neural.curriculum_fault_capture")

    def test_success_preserves_calls_returns_and_reset_proposals(self):
        api = self.api()
        driver = FakeDriver()
        runtime = make_runtime(driver)
        with tempfile.TemporaryDirectory() as temporary:
            directory = api.install_fault_capture(runtime, Path(temporary))
            reset_result = driver.reset(seeds=(501, 502), entry_ids=("next-a", "next-b"),
                                        reset_mask=(True, True))
            step_result = driver.step((17, 23))
            records = read_jsonl(directory / "operations.jsonl")

        self.assertIs(reset_result, driver.reset_result)
        self.assertIs(step_result, driver.step_result)
        self.assertEqual(driver.reset_calls, [(
            (), {"seeds": (501, 502), "entry_ids": ("next-a", "next-b"),
                 "reset_mask": (True, True)}
        )])
        self.assertEqual(driver.step_calls, [(((17, 23),), {})])
        intents = [item for item in records if item["record"] == "intent"]
        self.assertEqual([item["operation"] for item in intents], ["reset", "step"])
        self.assertEqual(intents[0]["reset"], {
            "seeds": [501, 502], "entry_ids": ["next-a", "next-b"],
            "reset_mask": [True, True],
        })
        self.assertEqual(intents[0]["slot_state"], [
            {"slot": 0, "case_id": "old-case", "entry_id": "old-entry", "actual_seed": 4101},
            {"slot": 1, "case_id": None, "entry_id": None, "actual_seed": None},
        ])
        self.assertEqual(intents[1]["actions"], [17, 23])
        self.assertEqual([item["record"] for item in records], [
            "intent", "delivered", "intent", "delivered",
        ])
        self.assertEqual([item["ordinal"] for item in records], [1, 1, 2, 2])

    def test_first_fault_retains_exact_actions_and_current_case_seed_without_retry(self):
        api = self.api()
        failure = RuntimeError("native batch rejected after dispatch")
        public = {"phase": "step", "accepted_prefix": 4, "fault_slots": [1]}
        driver = FakeDriver(step_error=failure, diagnostic=lambda: public)
        runtime = make_runtime(driver)
        runtime.collector._slot_cases[1] = SimpleNamespace(case_id="case-b", entry_id="entry-b")
        runtime.collector._slot_seeds[1] = 4202
        with tempfile.TemporaryDirectory() as temporary:
            directory = api.install_fault_capture(runtime, Path(temporary))
            with self.assertRaises(RuntimeError) as raised:
                driver.step((19, 27))
            fault = json.loads((directory / "first-fault.json").read_text(encoding="utf-8"))
            records = read_jsonl(directory / "operations.jsonl")

        self.assertIs(raised.exception, failure)
        self.assertEqual(len(driver.step_calls), 1)
        self.assertEqual([item["record"] for item in records], ["intent"])
        pending = fault["pending_intent"]
        self.assertEqual(pending["actions"], [19, 27])
        self.assertEqual(pending["slot_state"][0], {
            "slot": 0, "case_id": "old-case", "entry_id": "old-entry", "actual_seed": 4101,
        })
        self.assertEqual(pending["slot_state"][1], {
            "slot": 1, "case_id": "case-b", "entry_id": "entry-b", "actual_seed": 4202,
        })
        self.assertEqual(fault["public_diagnostic"]["data"]["fault_slots"], [1])
        self.assertEqual(fault["primary_error"], {
            "exception_type": "RuntimeError", "message": "native batch rejected after dispatch",
        })
        self.assertEqual(fault["public_diagnostic"], {"status": "available", "data": public})

    def test_diagnostic_failure_is_recorded_without_replacing_operation_error(self):
        api = self.api()
        failure = ValueError("primary driver failure")
        def broken_diagnostic():
            raise OSError("diagnostic offline")
        driver = FakeDriver(step_error=failure, diagnostic=broken_diagnostic)
        with tempfile.TemporaryDirectory() as temporary:
            directory = api.install_fault_capture(make_runtime(driver), Path(temporary))
            with self.assertRaises(ValueError) as raised:
                driver.step((3, 8))
            fault = json.loads((directory / "first-fault.json").read_text(encoding="utf-8"))

        self.assertIs(raised.exception, failure)
        self.assertEqual(fault["primary_error"]["exception_type"], "ValueError")
        self.assertEqual(fault["public_diagnostic"], {
            "status": "unavailable", "reason": "OSError: diagnostic offline",
        })

    def test_diagnostic_process_control_promotes_with_original_error_context(self):
        api = self.api()
        failure = RuntimeError("primary driver failure")
        control = KeyboardInterrupt("diagnostic control")
        def controlled_diagnostic():
            raise control
        driver = FakeDriver(step_error=failure, diagnostic=controlled_diagnostic)
        with tempfile.TemporaryDirectory() as temporary:
            directory = api.install_fault_capture(make_runtime(driver), Path(temporary))
            with self.assertRaises(KeyboardInterrupt) as raised:
                driver.step((3, 8))
            fault = json.loads((directory / "first-fault.json").read_text(encoding="utf-8"))

        self.assertIs(raised.exception, control)
        self.assertTrue(any("primary driver failure" in note
                            for note in getattr(control, "__notes__", ())))
        self.assertEqual(fault["primary_error"]["exception_type"], "RuntimeError")
        self.assertEqual(fault["finalization_control"]["exception_type"], "KeyboardInterrupt")

    def test_first_process_control_is_retained_when_later_finalization_control_raises(self):
        api = self.api()
        first = KeyboardInterrupt("operation interrupted")
        later = SystemExit("capture finalization interrupted")
        diagnostic_calls = []
        def forbidden_diagnostic():
            diagnostic_calls.append(True)
            return {"should_not": "be called"}
        driver = FakeDriver(step_error=first, diagnostic=forbidden_diagnostic)
        with tempfile.TemporaryDirectory() as temporary:
            directory = api.install_fault_capture(make_runtime(driver), Path(temporary))
            with patch.object(api, "_write_first_fault", side_effect=later):
                with self.assertRaises(KeyboardInterrupt) as raised:
                    driver.step((1, 2))

        self.assertIs(raised.exception, first)
        self.assertTrue(any("SystemExit" in note for note in getattr(first, "__notes__", ())))
        self.assertEqual(len(driver.step_calls), 1)
        self.assertEqual(diagnostic_calls, [])
        self.assertFalse((directory / "first-fault.json").exists())

    def test_fault_record_write_failure_adds_a_note_to_the_original_error(self):
        api = self.api()
        failure = RuntimeError("native operation failed")
        driver = FakeDriver(step_error=failure)
        with tempfile.TemporaryDirectory() as temporary:
            directory = api.install_fault_capture(make_runtime(driver), Path(temporary))
            original_open = Path.open
            def failed_fault_open(path, *args, **kwargs):
                if path.name == "first-fault.json":
                    raise OSError("fault volume unavailable")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", new=failed_fault_open):
                with self.assertRaises(RuntimeError) as raised:
                    driver.step((6, 7))
            self.assertIs(raised.exception, failure)
            self.assertFalse((directory / "first-fault.json").exists())

        self.assertTrue(any("fault volume unavailable" in note
                            for note in getattr(failure, "__notes__", ())))
        self.assertEqual(len(driver.step_calls), 1)

    def test_persistence_failure_refuses_before_native_effect(self):
        api = self.api()
        driver = FakeDriver()
        with tempfile.TemporaryDirectory() as temporary:
            api.install_fault_capture(make_runtime(driver), Path(temporary))
            original_open = Path.open
            def failed_open(path, *args, **kwargs):
                if path.name == "operations.jsonl":
                    raise OSError("disk full")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", new=failed_open):
                with self.assertRaisesRegex(OSError, "disk full"):
                    driver.step((4, 5))
                with self.assertRaises(api.FaultCaptureError):
                    driver.step((4, 5))
        self.assertEqual(driver.step_calls, [])

    def test_batch_and_record_bounds_refuse_before_native_effect(self):
        api = self.api()
        too_wide = FakeDriver()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                api.install_fault_capture(make_runtime(too_wide, batch_size=257), Path(temporary))
        self.assertEqual(too_wide.step_calls, [])

        oversized_case = SimpleNamespace(case_id="x" * (64 * 1024), entry_id="entry")
        driver = FakeDriver()
        runtime = make_runtime(driver)
        runtime.collector._slot_cases[0] = oversized_case
        with tempfile.TemporaryDirectory() as temporary:
            api.install_fault_capture(runtime, Path(temporary))
            with self.assertRaises(ValueError):
                driver.step((1, 2))
        self.assertEqual(driver.step_calls, [])

        self.assertEqual(api.MAX_TOTAL_BYTES, 8 * 1024 * 1024)
        driver = FakeDriver()
        with tempfile.TemporaryDirectory() as temporary:
            api.install_fault_capture(make_runtime(driver), Path(temporary))
            with patch.object(api, "MAX_TOTAL_BYTES", api.MAX_FAULT_BYTES):
                with self.assertRaises(api.FaultCaptureError):
                    driver.step((1, 2))
        self.assertEqual(driver.step_calls, [])

    def test_each_runtime_gets_an_exclusive_directory_under_shared_root(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = api.install_fault_capture(make_runtime(), root)
            second = api.install_fault_capture(make_runtime(), root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_import_has_no_optional_runtime_imports(self):
        code = """
import builtins, importlib.abc, sys
blocked = ('torch', 'numpy', 'gymnasium', 'spire_lab', 'torchrl', 'tensordict')
attempts = []
def forbidden(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            attempts.append(fullname)
            raise AssertionError('optional import attempted: ' + fullname)
guard = Guard()
sys.meta_path.insert(0, guard)
original = builtins.__import__
def checked(name, *args, **kwargs):
    if forbidden(name):
        attempts.append(name)
        raise AssertionError('optional import attempted: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import tempfile
from types import SimpleNamespace
import scripts.neural.curriculum_fault_capture as capture
class Driver:
    batch_size = 1
    def reset(self, **kwargs): return ()
    def step(self, actions): return ()
driver = Driver()
collector = SimpleNamespace(driver=driver, _batch_size=1, _slot_cases=[None], _slot_seeds=[None])
runtime = SimpleNamespace(collector=collector, driver=driver,
    config=SimpleNamespace(config_sha256='a'*64, manifest_sha256='b'*64))
with tempfile.TemporaryDirectory() as directory:
    capture.install_fault_capture(runtime, directory)
assert attempts == [], attempts
"""
        import os
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[2]
        env = {**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + str(root)}
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
