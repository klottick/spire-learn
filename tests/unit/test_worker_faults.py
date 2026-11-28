"""Worker faults cross the process pool as picklable WorkerFault values (stdlib only)."""

from __future__ import annotations

from types import SimpleNamespace
import pickle
import unittest
from unittest.mock import patch

from scripts.neural import distill_label_entry as label
from scripts.neural import full_run_entry as full


def _unpicklable_error():
    class RunFaultError(Exception):
        def __reduce__(self):
            raise pickle.PicklingError("import of module '_native' failed")

    return RunFaultError("illegal action 7 accepted twice")


def _raiser(error):
    def job(item):
        raise error
    return job


class WorkerFaultTests(unittest.TestCase):
    def test_precondition_original_error_does_not_pickle(self):
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(_unpicklable_error())

    def check(self, module, job_name, guarded, job, fields, expected):
        with patch.object(module, job_name, _raiser(_unpicklable_error())):
            with self.assertRaises(module.WorkerFault) as caught:
                guarded((job, object()))
        fault = caught.exception
        text = str(fault)
        for part in (*expected, "RunFaultError", "illegal action 7 accepted twice", "Traceback"):
            self.assertIn(part, text)
        self.assertTrue(fault.error_type.endswith("RunFaultError"))
        self.assertEqual(fault.error_message, "illegal action 7 accepted twice")
        self.assertIn("raise error", fault.traceback_text)
        for name in fields:
            self.assertIsInstance(getattr(fault, name), (str, int, type(None)))
        clone = pickle.loads(pickle.dumps(fault))
        self.assertIs(type(clone), module.WorkerFault)
        self.assertEqual(str(clone), text)
        self.assertEqual(clone.traceback_text, fault.traceback_text)

    def test_label_worker_fault(self):
        job = SimpleNamespace(shard=3, local=5, run_id="run-abc", case={})
        self.check(label, "_label_job", label._guarded_label_job, job, label.WorkerFault.FIELDS,
                   ("shard 3", "local 5", "run-abc"))

    def test_full_run_worker_fault(self):
        job = SimpleNamespace(index=9, run_id="run-xyz")
        self.check(full, "_play_job", full._guarded_play_job, job, full.WorkerFault.FIELDS,
                   ("index 9", "run-xyz"))

    def test_results_serial_path_surfaces_fault(self):
        job = SimpleNamespace(shard=1, local=2, run_id="r")
        with patch.object(label, "_label_job", _raiser(_unpicklable_error())):
            with self.assertRaises(label.WorkerFault):
                list(label._results([(job, None)], 1))

    def test_success_is_unchanged(self):
        job = SimpleNamespace(shard=1, local=2, run_id="r")
        with patch.object(label, "_label_job", lambda item: "ok"):
            self.assertEqual(label._guarded_label_job((job, None)), "ok")


if __name__ == "__main__":
    unittest.main()
