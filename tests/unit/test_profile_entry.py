"""Throughput profile orchestration with fake segment, runtime and train dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.neural import profile_entry


ROOT = Path(__file__).resolve().parents[2]
PLAN = Path("configs/neural/baseline-8726d467/plan.json")
V8726 = "8726d467ad8579730ddf4fa236b24aeb2206b71a"


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.0
        return self.now


class _Driver:
    def reset(self, **kwargs):
        return "reset"

    def step(self, actions):
        return "stepped"


class _Policy:
    def act(self, frames, **kwargs):
        return "acted"

    def values(self, frames):
        return "values"


class _Collector:
    def __init__(self):
        self.driver, self.policy = _Driver(), _Policy()

    def collect(self, **kwargs):
        self.driver.step(())
        self.policy.act(())
        return "rollout"


class _Runtime:
    def __init__(self):
        self.collector = _Collector()
        self.config = SimpleNamespace(device="cpu")
        self.torch = None
        self.optimizer = None

    def update(self, rollout):
        return "updated"


class _Fakes:
    def __init__(self, *, missing_checkpoint_at=None, fault_at=None):
        self.calls, self.decisions, self.runtimes = [], 0, []
        self.missing_checkpoint_at, self.fault_at = missing_checkpoint_at, fault_at

    def prepare_stream(self, recipe, destination, *, arm, split, shard_index, template):
        Path(destination).mkdir()
        (Path(destination) / "config.toml").write_text("")

    def load_config(self, path):
        return SimpleNamespace(environment_revision=V8726, path=str(path))

    def create_runtime(self, config, **kwargs):
        runtime = _Runtime()
        self.runtimes.append(runtime)
        return runtime

    def train(self, config, *, runtime, clock):
        runtime.collector.collect()
        runtime.update("rollout")
        return "report"

    def run_stream_segment(self, config, *, purpose, checkpoint, operation_limits, dependencies):
        index = len(self.calls)
        self.calls.append({"purpose": purpose, "parent": checkpoint,
                           "max_env": operation_limits.max_environment_actions})
        if index == self.fault_at:
            raise RuntimeError("native fault")
        runtime = dependencies["create_runtime"](config, purpose=purpose, checkpoint=checkpoint)
        dependencies["train"](config, runtime=runtime, clock=dependencies["clock"])
        if index == self.missing_checkpoint_at:
            return {"report": SimpleNamespace(summary={"status": "incomplete"}), "checkpoint": None,
                    "saved_prefix": None, "next_shard": None}
        self.decisions += 4096
        prefix = {"learned_decisions": self.decisions, "attempted_environment_slots": self.decisions,
                  "update_count": 8 * (index + 1)}
        return {"report": "report", "checkpoint": f"ckpt-{index}", "saved_prefix": prefix,
                "next_shard": {"shard_index": index + 1}}

    def deps(self):
        from scripts.neural.baseline_entry import load_plan
        return dict(load_plan=load_plan, prepare_stream=self.prepare_stream, load_config=self.load_config,
                    create_runtime=self.create_runtime, train=self.train,
                    run_stream_segment=self.run_stream_segment)


class ProfileEntryTests(unittest.TestCase):
    def run_profile(self, fakes):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            result = profile_entry.run_profile(ROOT, PLAN, output, dependencies=fakes.deps(), clock=_Clock())
            written = json.loads((output / "profile.json").read_text())
        return result, written

    def test_segments_chain_from_fresh_weights_with_cumulative_limits(self):
        fakes = _Fakes()
        result, written = self.run_profile(fakes)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(written["status"], "complete")
        self.assertEqual([call["purpose"] for call in fakes.calls], ["train", "resume", "resume"])
        self.assertEqual([call["parent"] for call in fakes.calls], [None, "ckpt-0", "ckpt-1"])
        self.assertEqual([call["max_env"] for call in fakes.calls], [4096, 8192, 12288])
        self.assertEqual([row["shard_index"] for row in written["segments"]], [0, 1, 2])
        self.assertEqual([row["synchronized"] for row in written["segments"]], [True, True, False])

    def test_checkpoint_warm_starts_like_the_baseline(self):
        fakes = _Fakes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            profile_entry.run_profile(ROOT, PLAN, output, checkpoint=Path("fresh"),
                                      dependencies=fakes.deps(), clock=_Clock())
        self.assertEqual([call["purpose"] for call in fakes.calls], ["warm_start", "resume", "resume"])
        self.assertEqual(fakes.calls[0]["parent"], Path("fresh"))

    def test_spans_cover_hot_path_and_wrappers_are_restored(self):
        fakes = _Fakes()
        _, written = self.run_profile(fakes)
        spans = written["segments"][0]["spans"]
        for name in ("collection", "driver_step", "policy_act", "ppo_update"):
            self.assertEqual(spans[name]["count"], 1, name)
        self.assertEqual(written["segments"][0]["learned_decisions"], 4096)
        self.assertEqual(written["segments"][0]["ranked"][0]["span"],
                         max(spans, key=lambda name: spans[name]["exclusive_wall_seconds"]))
        runtime = fakes.runtimes[0]
        for obj, name in ((runtime.collector.driver, "step"), (runtime.collector.policy, "act"),
                          (runtime.collector, "collect"), (runtime, "update")):
            self.assertNotIn(name, vars(obj), name)
        from spire_learn.neural import features, model
        self.assertIs(model.encode_frames, features.encode_frames)
        self.assertEqual(features.encode_frames.__module__, "spire_learn.neural.features")

    def test_missing_checkpoint_stops_and_fault_is_recorded(self):
        result, written = self.run_profile(_Fakes(missing_checkpoint_at=1))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(written["stop_reason"], "segment_without_checkpoint")
        self.assertEqual(len(written["segments"]), 2)
        result, written = self.run_profile(_Fakes(fault_at=0))
        self.assertEqual(written["status"], "incomplete")
        self.assertEqual(written["fault"], {"type": "RuntimeError", "message": "native fault"})
        self.assertNotIn("_timing", written["segments"][0])


if __name__ == "__main__":
    unittest.main()
