"""Worker faults cross the process pool as picklable WorkerFault values (stdlib only)."""

from __future__ import annotations

from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts.neural import distill_label_entry as label
from scripts.neural import full_run_entry as full
from spire_learn.neural.worker_faults import WorkerFault


def _unpicklable_error():
    class RunFaultError(Exception):
        def __reduce__(self):
            raise pickle.PicklingError("import of module '_native' failed")

    return RunFaultError("illegal action 7 accepted twice")


def _raiser(error):
    def job(item):
        raise error
    return job


class _BadStr(Exception):
    def __str__(self):
        raise RuntimeError("no str")


def _pool_job(item):
    """Runs in the spawned child (defined in this test module, so no numpy/Gym is needed)."""
    label._label_job = _raiser(_unpicklable_error())
    return label._guarded_label_job(item)


class WorkerFaultTests(unittest.TestCase):
    def test_precondition_original_error_does_not_pickle(self):
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(_unpicklable_error())

    def check(self, module, job_name, guarded, job, fields, expected):
        with patch.object(module, job_name, _raiser(_unpicklable_error())):
            with self.assertRaises(WorkerFault) as caught:
                guarded((job, object()))
        fault = caught.exception
        text = str(fault)
        for part in (*expected, "RunFaultError", "illegal action 7 accepted twice", "Traceback"):
            self.assertIn(part, text)
        self.assertTrue(fault.error_type.endswith("RunFaultError"))
        self.assertEqual(fault.error_message, "illegal action 7 accepted twice")
        self.assertIn("raise error", fault.traceback_text)
        for value in fault.case.values():
            self.assertIsInstance(value, (str, int, type(None)))
        clone = pickle.loads(pickle.dumps(fault))
        self.assertIs(type(clone), WorkerFault)
        self.assertEqual(str(clone), text)
        self.assertEqual(clone.__dict__, fault.__dict__)

    def test_label_worker_fault(self):
        job = SimpleNamespace(shard=3, local=5, run_id="run-abc", case={})
        self.check(label, "_label_job", label._guarded_label_job, job, (),
                   ("shard 3", "local 5", "run-abc"))

    def test_full_run_worker_fault(self):
        job = SimpleNamespace(index=9, run_id="run-xyz")
        self.check(full, "_play_job", full._guarded_play_job, job, (),
                   ("index 9", "run-xyz"))

    def test_results_serial_path_surfaces_fault(self):
        job = SimpleNamespace(shard=1, local=2, run_id="r")
        with patch.object(label, "_label_job", _raiser(_unpicklable_error())):
            with self.assertRaises(WorkerFault):
                list(label._results([(job, None)], 1))

    def test_broken_str_still_yields_fault(self):
        for module, job_name, guarded, job in (
                (label, "_label_job", label._guarded_label_job, SimpleNamespace(shard=1, local=2, run_id="r")),
                (full, "_play_job", full._guarded_play_job, SimpleNamespace(index=1, run_id="r"))):
            with patch.object(module, job_name, _raiser(_BadStr())):
                with self.assertRaises(WorkerFault) as caught:
                    guarded((job, None))
            self.assertIn("_BadStr", caught.exception.error_message)

    def test_real_spawn_pool_surfaces_fault(self):
        job = SimpleNamespace(shard=3, local=5, run_id="run-abc")
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        try:
            with self.assertRaises(WorkerFault) as caught:
                list(pool.map(_pool_job, [(job, None)]))
        finally:
            pool.shutdown(cancel_futures=True)
        fault = caught.exception
        self.assertEqual((fault.shard, fault.local, fault.run_id), (3, 5, "run-abc"))
        self.assertTrue(fault.error_type.endswith("RunFaultError"))
        self.assertEqual(fault.error_message, "illegal action 7 accepted twice")

    def run_as_script(self, entry, kind):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(root / "tests/fixtures/worker_fault_main.py"), str(root / entry), kind],
            capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": str(root / "src")})
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_script_as_main_label_fault_survives_spawn(self):
        out = self.run_as_script("scripts/neural/distill_label_entry.py", "label")
        self.assertIn("RAISED spire_learn.neural.worker_faults.WorkerFault", out)
        for part in ("shard 3", "local 5", "run-abc", "UnpicklableFault", "illegal action 7 accepted twice"):
            self.assertIn(part, out)

    def test_script_as_main_full_run_fault_survives_spawn(self):
        out = self.run_as_script("scripts/neural/full_run_entry.py", "run")
        self.assertIn("RAISED spire_learn.neural.worker_faults.WorkerFault", out)
        for part in ("index 9", "run-xyz", "UnpicklableFault", "illegal action 7 accepted twice"):
            self.assertIn(part, out)

    def test_success_is_unchanged(self):
        job = SimpleNamespace(shard=1, local=2, run_id="r")
        with patch.object(label, "_label_job", lambda item: "ok"):
            self.assertEqual(label._guarded_label_job((job, None)), "ok")


if __name__ == "__main__":
    unittest.main()
