"""Throughput profile orchestration with fake segment, runtime and train dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
        from spire_learn.neural import model
        model.encode_frames(frames)  # looked up at call time, as the real model does
        return "acted"

    def values(self, frames):
        return "values"


class _SlottedDriver:
    __slots__ = ()

    def reset(self, **kwargs):
        return "reset"

    def step(self, actions):
        return "stepped"


class _Optimizer:
    def step(self):
        return None


class _Collector:
    def __init__(self, driver=None):
        self.driver, self.policy = driver or _Driver(), _Policy()

    def collect(self, **kwargs):
        self.driver.step(())
        self.policy.act(())
        self.policy.values(())
        return "rollout"


class _Runtime:
    def __init__(self, driver=None):
        self.collector = _Collector(driver)
        self.config = SimpleNamespace(device="cpu")
        self.torch = None
        self.optimizer = _Optimizer()

    def update(self, rollout):
        from spire_learn.neural import features
        features.encode_frames(())  # PPO re-encodes through the features module
        self.optimizer.step()
        return "updated"


class _Fakes:
    def __init__(self, *, missing_checkpoint_at=None, fault_at=None, train_fault_at=None, driver=None):
        self.calls, self.decisions, self.runtimes = [], 0, []
        self.missing_checkpoint_at, self.fault_at = missing_checkpoint_at, fault_at
        self.train_fault_at, self.driver = train_fault_at, driver

    def prepare_stream(self, recipe, destination, *, arm, split, shard_index, template):
        Path(destination).mkdir()
        (Path(destination) / "config.toml").write_text("")

    def load_config(self, path):
        return SimpleNamespace(environment_revision=V8726, path=str(path))

    def create_runtime(self, config, **kwargs):
        runtime = _Runtime(self.driver)
        self.runtimes.append(runtime)
        return runtime

    def train(self, config, *, runtime, clock):
        runtime.collector.collect()
        if len(self.calls) - 1 == self.train_fault_at:
            raise RuntimeError("train fault")
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
    def test_v1_1_0_profile_plan_matches_the_baseline_shape_on_the_new_build(self):
        from scripts.neural.baseline_entry import load_plan
        plan = load_plan(ROOT / "configs/neural/profile-gym-1.1.0/plan.json")
        baseline = load_plan(ROOT / PLAN)
        self.assertEqual(plan["gym_revision"], "26615ef470e420114f29a208103260b517ee4906")
        self.assertEqual(plan["template"], "configs/neural/profile-gym-1.1.0/template.toml")
        own = {"baseline_id", "gym_revision", "template", "notes", "qualifications"}
        self.assertEqual({k: v for k, v in plan.items() if k not in own},
                         {k: v for k, v in baseline.items() if k not in own})
        template = (ROOT / plan["template"]).read_text()
        self.assertIn('gym_revision = "26615ef470e420114f29a208103260b517ee4906"', template)
        self.assertNotIn("8726d467", template.split("\n", 1)[1])
        self.assertIn("+gym-1.1.0", template)
        base_lines = (ROOT / baseline["template"]).read_text().splitlines()[1:]
        changed = [new.split("=", 1)[0].strip()
                   for old, new in zip(base_lines, template.splitlines()[1:]) if old != new]
        self.assertEqual(changed, ["gym_revision", "standalone", "run"])

    def test_standard_profile_plans_use_the_standard_identity_and_the_gpu_wait_gate(self):
        from scripts.neural.baseline_entry import load_plan
        baseline = load_plan(ROOT / "configs/neural/baseline-gym-1.1.1-cv2/plan.json")
        for rows in (16, 64):
            plan = load_plan(ROOT / f"configs/neural/profile-standard-1.1.1/plan-{rows}.json")
            self.assertEqual(plan["gpu_wait"], baseline["gpu_wait"])
            self.assertEqual(plan["gym_revision"], "e3c13836bd1458be8a87b93160f5c350f78fcd80")
            self.assertEqual(plan["recipe"], "configs/neural/baalorlord-standard-v1/recipe.json")
            template = (ROOT / plan["template"]).read_text()
            self.assertIn(f"rows = {rows}\n", template)
            self.assertIn('mode = "standard-standalone"', template)
            self.assertIn('device = "cuda:0"', template)
            self.assertIn("spire-learn-combat-ff-standard-v1", template)

    def test_gpu_wait_polls_until_free_and_is_off_without_a_plan_entry_or_cuda(self):
        plan = {"gpu_wait": {"min_free_bytes": 10, "poll_seconds": 5, "max_total_seconds": 100}}
        cuda, cpu = SimpleNamespace(device="cuda:0"), SimpleNamespace(device="cpu")
        free, sleeps = iter([1, 2, 10]), []
        deps = {"gpu_free_bytes": lambda: next(free), "sleep": sleeps.append}
        profile = {}
        profile_entry._gpu_wait(plan, cuda, deps, _Clock(), profile, "segment-0")
        self.assertEqual(sleeps, [5, 5])
        self.assertEqual(profile["gpu_waits"]["events"][0]["stage"], "segment-0")
        self.assertGreater(profile["gpu_waits"]["total_waited_seconds"], 0)
        untouched = {}
        profile_entry._gpu_wait({}, cuda, {}, _Clock(), untouched, "s")
        profile_entry._gpu_wait(plan, cpu, {}, _Clock(), untouched, "s")
        self.assertEqual(untouched, {})
        with self.assertRaisesRegex(RuntimeError, "gpu_wait_exhausted"):
            profile_entry._gpu_wait({"gpu_wait": {"min_free_bytes": 10, "poll_seconds": 1,
                                                  "max_total_seconds": 3}}, cuda,
                                    {"gpu_free_bytes": lambda: 0, "sleep": lambda s: None},
                                    _Clock(), {}, "s")

    def setUp(self):
        from spire_learn.neural import features, model
        self.stub = stub = lambda frames, *args, **kwargs: "batch"
        for module in (features, model):
            patcher = patch.object(module, "encode_frames", stub)
            patcher.start()
            self.addCleanup(patcher.stop)

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
        for name in ("collection", "driver_step", "policy_act", "bootstrap_values", "ppo_update", "adam_step"):
            self.assertEqual(spans[name]["count"], 1, name)
        # One encode in inference, one in the update: each counted once, not double-wrapped.
        self.assertEqual(spans["encode_frames"]["count"], 2)
        ranked = {row["span"]: row for row in written["segments"][0]["ranked"]}
        self.assertIn("unattributed", ranked)
        self.assertAlmostEqual(sum(row["exclusive_wall_seconds"] for row in ranked.values()),
                               written["segments"][0]["wall_seconds"], places=2)
        self.assertEqual(written["segments"][0]["not_instrumented"], [])
        self.assertEqual(written["segments"][0]["learned_decisions"], 4096)
        runtime = fakes.runtimes[0]
        for obj, name in ((runtime.collector.driver, "step"), (runtime.collector.policy, "act"),
                          (runtime.collector, "collect"), (runtime, "update")):
            self.assertNotIn(name, vars(obj), name)
        from spire_learn.neural import features, model
        self.assertIs(model.encode_frames, self.stub)
        self.assertIs(features.encode_frames, self.stub)

    def test_phases_partition_the_segment_and_cprofile_is_opt_in(self):
        fakes = _Fakes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            profile_entry.run_profile(ROOT, PLAN, output, dependencies=fakes.deps(), clock=_Clock(),
                                      cprofile_segment=1)
            written = json.loads((output / "profile.json").read_text())
        for row in written["segments"]:
            phases = row["phases"]
            self.assertEqual(set(phases), {"create_runtime_seconds", "train_seconds", "after_train_seconds"})
            self.assertAlmostEqual(sum(phases.values()), row["wall_seconds"], places=2)
            self.assertGreaterEqual(phases["after_train_seconds"], 0)
        self.assertNotIn("cprofile_top_cumulative", written["segments"][0])
        self.assertTrue(written["segments"][1]["cprofile_top_cumulative"])

    def test_unpatchable_native_driver_is_reported_not_instrumented(self):
        _, written = self.run_profile(_Fakes(driver=_SlottedDriver()))
        self.assertEqual(written["status"], "complete")
        self.assertEqual(written["segments"][0]["not_instrumented"], ["driver_reset", "driver_step"])
        self.assertNotIn("driver_step", written["segments"][0]["spans"])

    def test_faulting_segment_still_reports_its_spans_and_wall(self):
        _, written = self.run_profile(_Fakes(train_fault_at=0))
        self.assertEqual(written["fault"], {"type": "RuntimeError", "message": "train fault"})
        row = written["segments"][0]
        self.assertGreater(row["wall_seconds"], 0)
        self.assertEqual(row["spans"]["collection"]["count"], 1)

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
