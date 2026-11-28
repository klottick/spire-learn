"""distill_train_entry: the GPU gate, argument parsing, output checks and summary assembly (stdlib fakes)."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.neural import distill_train_entry as entry

ROOT = Path(__file__).resolve().parents[2]


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def gate(settings, free, *, device="cuda:0", journal=None):
    clock = FakeClock()
    readings = iter(free)
    released = []
    kwargs = {} if journal is None else {"journal": journal}
    instance = entry.GpuGate(settings, device=device, free_bytes=lambda: next(readings),
                             release=lambda: released.append(True), sleep=clock.sleep, clock=clock, **kwargs)
    return instance, clock, released


class GpuGateTests(unittest.TestCase):
    def test_waits_until_free_and_records_events(self):
        events = []
        instance, clock, released = gate((100, 5, 60), [10, 20, 150], journal=events.append)
        instance.wait("eval-2")
        self.assertEqual(instance.waited_seconds, 10.0)
        self.assertEqual([event["event"] for event in events], ["gpu_wait", "gpu_resumed"])
        self.assertEqual(events[0], {"event": "gpu_wait", "stage": "eval-2", "free_bytes": 10, "min_free_bytes": 100})
        self.assertEqual(events[1]["waited_seconds"], 10.0)
        self.assertEqual(instance.events, events)
        instance.release()
        self.assertEqual(released, [True])

    def test_no_wait_when_free(self):
        instance, clock, _ = gate((100, 5, 60), [500])
        instance.wait("train-start")
        self.assertEqual((instance.waited_seconds, instance.events, clock.now), (0.0, [], 100.0))

    def test_exhaustion_and_cumulative_cap(self):
        # The first wait uses 20 s of the 30 s cap; the second runs out after 10 more.
        instance, clock, _ = gate((100, 10, 30), [0, 0, 200, 0, 0, 0, 0])
        instance.wait("qualify")
        self.assertEqual(instance.waited_seconds, 20.0)
        with self.assertRaises(entry.GpuWaitExhausted):
            instance.wait("eval-4")
        self.assertEqual(instance.waited_seconds, 30.0)
        self.assertEqual(instance.events[-1]["event"], "gpu_wait_exhausted")
        self.assertEqual(instance.events[-1]["stage"], "eval-4")

    def test_inactive_on_cpu_or_without_settings(self):
        for settings, device in (((100, 5, 60), "cpu"), (None, "cuda:0")):
            with self.subTest(device=device):
                instance, _, released = gate(settings, [], device=device)
                instance.wait("qualify")  # would raise StopIteration if it polled
                instance.release()
                self.assertEqual((instance.waited_seconds, instance.events, released), (0.0, [], []))

    def test_journal_attached_later_replays_earlier_events(self):
        instance, _, _ = gate((100, 5, 60), [0, 150, 0, 150])
        instance.wait("qualify")
        log = []
        instance.set_journal(log.append)
        self.assertEqual([event["event"] for event in log], ["gpu_wait", "gpu_resumed"])
        instance.wait("eval-2")
        self.assertEqual(len(log), 4)
        instance.set_journal(None)
        self.assertEqual(len(instance.events), 4)


class ArgumentTests(unittest.TestCase):
    def test_parse_gpu_wait(self):
        self.assertEqual(entry.parse_gpu_wait("1024:5:600"), (1024, 5, 600))
        for bad in ("1024:5", "1024:5:600:1", "a:5:6", "0:5:6", "1:-5:6", ""):
            with self.subTest(text=bad), self.assertRaises(argparse.ArgumentTypeError):
                entry.parse_gpu_wait(bad)

    def test_cuda_config_requires_gpu_wait(self):
        config = SimpleNamespace(device="cuda:0", output_root=ROOT / "artifacts")
        argv = ["--config", "c.toml", "--dataset", "d.json", "--output", str(ROOT / "artifacts/x")]
        with patch("spire_learn.neural.config.load_distill_config", return_value=config), \
             contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as raised:
            entry.main(argv)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--gpu-wait", stderr.getvalue())

    def test_check_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = SimpleNamespace(output_root=root / "runs")
            fresh = root / "runs" / "s1"
            with patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("SPIRE_LEARN_RUNNER_OUTPUT", None)
                entry._check_output(fresh, config)  # absent is fine
                fresh.mkdir(parents=True)
                entry._check_output(fresh, config)  # empty is fine
                (fresh / "x").write_text("")
                with self.assertRaisesRegex(ValueError, "fresh or empty"):
                    entry._check_output(fresh, config)
                with self.assertRaisesRegex(ValueError, "absolute"):
                    entry._check_output(Path("relative/s1"), config)
                with self.assertRaisesRegex(ValueError, "output_root"):
                    entry._check_output(root / "elsewhere", config)
            with patch.dict("os.environ", {"SPIRE_LEARN_RUNNER_OUTPUT": str(root / "runs" / "other")}):
                with self.assertRaisesRegex(ValueError, "SPIRE_LEARN_RUNNER_OUTPUT"):
                    entry._check_output(root / "runs" / "s2", config)
                entry._check_output(root / "runs" / "other", config)
            runner = root / "runs" / "run-1"
            for name in ("manifest.json", "stdout.txt", "stderr.txt"):  # what the runner writes first
                runner.mkdir(parents=True, exist_ok=True)
                (runner / name).write_text("")
            (runner / ".runtime").mkdir()
            with patch.dict("os.environ", {"SPIRE_LEARN_RUNNER_OUTPUT": str(runner)}):
                entry._check_output(runner / "train", config)  # a fresh directory inside the runner's
                with self.assertRaisesRegex(ValueError, "fresh or empty"):
                    entry._check_output(runner, config)  # the runner's own directory is never empty
                with self.assertRaisesRegex(ValueError, "SPIRE_LEARN_RUNNER_OUTPUT"):
                    entry._check_output(root / "runs" / "run-1-sibling", config)


class SummaryTests(unittest.TestCase):
    def test_finalize_summary_records_waits_and_process_time(self):
        instance, _, _ = gate((100, 5, 60), [0, 150])
        instance.wait("qualify")
        summary = {"status": "complete", "timing": {"train_eval_wall_seconds": 12.0, "gpu_wait_seconds": 0.0}}
        result = entry.finalize_summary(summary, instance, cpu_seconds=30.5, entry_wall_seconds=40.0)
        self.assertIs(result, summary)
        self.assertEqual([event["event"] for event in summary["gpu_waits"]], ["gpu_wait", "gpu_resumed"])
        self.assertEqual(summary["timing"], {"train_eval_wall_seconds": 12.0, "gpu_wait_seconds": 5.0,
                                             "cpu_seconds": 30.5, "entry_wall_seconds": 40.0})


if __name__ == "__main__":
    unittest.main()
