"""Baseline orchestration with fake segment, runtime and evaluation dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.neural import baseline_entry


ROOT = Path(__file__).resolve().parents[2]
PLAN = Path("configs/neural/baseline-8726d467/plan.json")
V8726 = "8726d467ad8579730ddf4fa236b24aeb2206b71a"


class _Clock:
    def __init__(self, step=1.0):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


class _Runtime:
    def __init__(self, log):
        self.log, self.closed, self.operation_limits = log, False, None

    def save(self, path, purpose):
        Path(path).mkdir()
        self.log.append(("save", str(path), purpose))

    def close(self):
        self.closed = True


class _Fakes:
    """Scripted dependencies; ``decisions_per_segment`` drives learned progress."""

    def __init__(self, *, decisions_per_segment=4096, missing_checkpoint_at=None, fault_at=None,
                 qualification_checkpoint=True, disk_free=10**12, tree_bytes=0, panel_cases=3):
        self.log, self.segments, self.decisions = [], [], 0
        self.decisions_per_segment = decisions_per_segment
        self.missing_checkpoint_at, self.fault_at = missing_checkpoint_at, fault_at
        self.qualification_checkpoint = qualification_checkpoint
        self.free, self.bytes, self.panel_cases = disk_free, tree_bytes, panel_cases
        self.evaluated = []

    def prepare_stream(self, recipe, destination, *, arm, split, shard_index, template):
        Path(destination).mkdir()
        (Path(destination) / "config.toml").write_text("")
        self.log.append(("prepare", Path(destination).name, split, shard_index, Path(template).name))

    def load_config(self, path):
        return SimpleNamespace(environment_revision=V8726, config_sha256="c" * 64, path=str(path),
                               accepted_identities=("identity",))

    def create_runtime(self, config, **kwargs):
        self.log.append(("create_runtime", kwargs))
        return _Runtime(self.log)

    def run_stream_segment(self, config, *, purpose, checkpoint, operation_limits):
        call = {"purpose": purpose, "parent": None if checkpoint is None else str(checkpoint),
                "max_env": operation_limits.max_environment_actions,
                "max_updates": operation_limits.max_updates}
        self.segments.append(call)
        index = len(self.segments) - 1
        if index == 0:  # qualification
            path = Path(config.path).parent / "qual-checkpoint"
            prefix = {"learned_decisions": 64}
            return {"checkpoint": path if self.qualification_checkpoint else None,
                    "saved_prefix": prefix if self.qualification_checkpoint else None,
                    "next_shard": {"shard_index": 1}, "report": SimpleNamespace(summary={"status": "x"})}
        training_index = index  # 1-based among training segments
        if self.fault_at == training_index:
            raise RuntimeError("segment fault")
        if self.missing_checkpoint_at == training_index:
            return {"checkpoint": None, "saved_prefix": None, "next_shard": None,
                    "report": SimpleNamespace(summary={"status": "incomplete"})}
        self.decisions += self.decisions_per_segment
        path = Path(config.path).parent / "checkpoint-final"
        return {"checkpoint": path, "saved_prefix": {"learned_decisions": self.decisions},
                "next_shard": {"shard_index": training_index}, "report": SimpleNamespace(summary={})}

    def evaluate(self, config, *, runtime, clock, allow_final):
        self.evaluated.append(config.path)
        return SimpleNamespace(summary={"status": "complete"})

    def preflight(self, checkpoint, identity, *, purpose):
        assert purpose == "evaluate"

    def load_recipe(self, path):
        return "recipe"

    def build_panel(self, recipe):
        cases = [{"case_id": f"dev-{i}", "environment_seed": i} for i in range(self.panel_cases)]
        return {"manifest": {"schema": "spire-learn-neural-cases-v1", "split": "selection",
                             "partition": "selection", "cases": cases}}

    def summarize(self, panel, reports):
        return {"status": "complete", "cases": sorted(reports)}

    def deps(self):
        return dict(prepare_stream=self.prepare_stream, load_config=self.load_config,
                    create_runtime=self.create_runtime, run_stream_segment=self.run_stream_segment,
                    evaluate=self.evaluate, preflight=self.preflight, load_recipe=self.load_recipe,
                    build_panel=self.build_panel, summarize=self.summarize,
                    disk_free=lambda path: self.free, tree_bytes=lambda path: self.bytes)


def _run(fakes, *, clock=None):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "out"
        output.mkdir()
        summary = baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=fakes.deps(),
                                              clock=clock or _Clock())
        events = [json.loads(line)["event"] for line in (output / "progress.jsonl").read_text().splitlines()]
        written = json.loads((output / "summary.json").read_text())
        return summary, events, written


class BaselineEntryTests(unittest.TestCase):
    def test_plan_binds_the_8726d467_template(self):
        plan = baseline_entry.load_plan(ROOT / PLAN)
        self.assertEqual(plan["gym_revision"], V8726)
        self.assertEqual(plan["target_learned_decisions"], 405504)
        template = (ROOT / plan["template"]).read_text()
        self.assertIn(V8726, template)
        self.assertIn('device = "cuda:0"', template)

    def test_fresh_qualification_then_warm_start_and_resume_until_target(self):
        fakes = _Fakes(decisions_per_segment=135168)
        summary, events, written = _run(fakes)
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(summary["learned_decisions"], 405504)
        purposes = [call["purpose"] for call in fakes.segments]
        self.assertEqual(purposes, ["warm_start", "warm_start", "resume", "resume"])
        fresh = summary["checkpoints"][0]["path"]
        self.assertEqual(fakes.segments[0]["parent"], fresh)  # qualification from fresh weights
        self.assertEqual(fakes.segments[1]["parent"], fresh)  # training starts from fresh weights
        self.assertNotEqual(fakes.segments[2]["parent"], fakes.segments[0]["parent"])
        self.assertEqual(fakes.segments[0]["max_env"], 512)
        self.assertEqual(fakes.segments[1]["max_env"], 4096)
        self.assertEqual(fakes.segments[1]["max_updates"], 8)
        self.assertIn("qualification_passed", events)
        self.assertEqual(written["stop_reason"], "target_reached")
        templates = {entry[4] for entry in fakes.log if entry[0] == "prepare"}
        self.assertEqual(templates, {"template.toml"})

    def test_decisions_counted_from_saved_prefix_not_segment_count(self):
        fakes = _Fakes(decisions_per_segment=100000)
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["segments"], 5)  # 500,000 >= 405,504 after the fifth segment
        self.assertEqual(summary["learned_decisions"], 500000)

    def test_missing_checkpoint_stops_without_restart(self):
        fakes = _Fakes(decisions_per_segment=4096, missing_checkpoint_at=3)
        summary, events, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "segment_without_checkpoint")
        self.assertEqual(len(fakes.segments), 4)  # qualification + 3 training attempts; no restart
        self.assertEqual(summary["learned_decisions"], 8192)
        self.assertIn("segment_without_checkpoint", events)
        self.assertEqual([c["label"] for c in summary["checkpoints"]][:2], ["fresh-initial", "final"])

    def test_qualification_failure_stops_before_training(self):
        fakes = _Fakes(qualification_checkpoint=False)
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "qualification_without_checkpoint")
        self.assertEqual(len(fakes.segments), 1)
        self.assertEqual(summary["learned_decisions"], 0)
        self.assertIn("fresh-initial", summary["evaluations"])

    def test_fault_is_recorded_not_retried(self):
        fakes = _Fakes(fault_at=2)
        summary, events, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "fault")
        self.assertEqual(summary["fault"]["type"], "RuntimeError")
        self.assertEqual(len(fakes.segments), 3)
        self.assertIn("fault", events)

    def test_wall_and_storage_caps(self):
        summary, _, _ = _run(_Fakes(), clock=_Clock(step=5000.0))
        self.assertEqual(summary["stop_reason"], "training_wall")
        summary, _, _ = _run(_Fakes(disk_free=1))
        self.assertEqual(summary["stop_reason"], "disk_free_below_minimum")
        summary, _, _ = _run(_Fakes(tree_bytes=2**40))
        self.assertEqual(summary["stop_reason"], "output_byte_cap")

    def test_retention_and_panel_evaluation(self):
        fakes = _Fakes(decisions_per_segment=45056)
        summary, _, _ = _run(fakes)
        labels = {row["label"]: row["learned_decisions"] for row in summary["checkpoints"]}
        self.assertEqual(labels["fresh-initial"], 0)
        self.assertGreaterEqual(labels["final"], 405504)
        self.assertEqual(labels["near-135168"], 135168)
        self.assertEqual(labels["near-270336"], 270336)
        self.assertEqual(set(summary["evaluations"]), {"fresh-initial", "final", "near-135168", "near-270336"})
        self.assertEqual(summary["evaluations"]["final"]["cases_evaluated"], 3)
        self.assertEqual(len(fakes.evaluated), 12)
        evaluate_calls = [entry[1] for entry in fakes.log
                          if entry[0] == "create_runtime" and entry[1].get("purpose") == "evaluate"]
        self.assertTrue(all(call["action_selection"] == "deterministic" for call in evaluate_calls))

    def test_evaluation_fault_still_writes_summary(self):
        fakes = _Fakes(decisions_per_segment=405504)

        def broken(config, **kwargs):
            raise ValueError("evaluation broke")

        fakes.evaluate = broken
        summary, events, written = _run(fakes)
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(written["evaluation_fault"]["type"], "ValueError")
        self.assertIn("evaluation_fault", events)


class TemplatePreparationTests(unittest.TestCase):
    def test_prepare_stream_template_is_optional_and_binds_8726d467(self):
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.stream_runner import prepare_stream
        base = ROOT / "artifacts/development/neural-combat"
        base.mkdir(parents=True, exist_ok=True)
        recipe = ROOT / "configs/neural/curriculum-v1/recipe.json"
        with tempfile.TemporaryDirectory(dir=base) as directory:
            prepare_stream(recipe, Path(directory) / "baseline", arm="mixed", split="train", shard_index=0,
                           template=ROOT / "configs/neural/baseline-8726d467/template.toml")
            baseline = load_neural_config(Path(directory) / "baseline" / "config.toml")
            prepare_stream(recipe, Path(directory) / "default", arm="mixed", split="train", shard_index=0)
            default = load_neural_config(Path(directory) / "default" / "config.toml")
        self.assertEqual((baseline.environment_revision, baseline.device), (V8726, "cuda:0"))
        self.assertEqual((baseline.rollout_rows, baseline.rollout_steps), (16, 32))
        self.assertEqual(baseline.runtime_gpu_uuid, "GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d")
        self.assertEqual(default.environment_revision, "533dad1c8aec86df624b3742b6b3ad1dbf1d0033")
        self.assertEqual((default.device, default.rollout_rows), ("cpu", 4))


if __name__ == "__main__":
    unittest.main()
