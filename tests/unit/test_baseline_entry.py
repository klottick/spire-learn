"""Baseline orchestration with fake segment, runtime and evaluation dependencies."""

from __future__ import annotations

import hashlib
import json
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

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
        self.env_attempts = self.updates = 0
        self.outcome = "win"
        self.corrupt_fresh = False
        self.failed_segment_at = None
        self.fail_evaluation_at = None
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
            # A wall-limited segment: healthy exit, incomplete status, nothing published.
            return {"checkpoint": None, "saved_prefix": None, "next_shard": None,
                    "report": SimpleNamespace(exit_code=0, summary={"status": "incomplete"})}
        if self.failed_segment_at == training_index:
            return {"checkpoint": None, "saved_prefix": None, "next_shard": None,
                    "report": SimpleNamespace(exit_code=1, summary={"status": "operational_failure"})}
        # Like runner._collection_budget_stop: limits are cumulative, including the restored prefix.
        restored_env = 0 if purpose == "warm_start" else self.env_attempts
        restored_updates = 0 if purpose == "warm_start" else self.updates
        if purpose == "warm_start":
            self.decisions = self.env_attempts = self.updates = 0
        env_room = max(0, operation_limits.max_environment_actions - restored_env)
        update_room = max(0, operation_limits.max_updates - restored_updates)
        work = min(self.decisions_per_segment, env_room) if update_room else 0
        self.decisions += work
        self.env_attempts += work
        self.updates += min(update_room, 8) if work else 0
        path = Path(config.path).parent / "checkpoint-final"
        prefix = {"learned_decisions": self.decisions, "attempted_environment_slots": self.env_attempts,
                  "update_count": self.updates}
        return {"checkpoint": path, "saved_prefix": prefix,
                "next_shard": {"shard_index": training_index}, "report": SimpleNamespace(summary={})}

    def evaluate(self, config, *, runtime, clock, allow_final):
        self.evaluated.append(config.path)
        if self.fail_evaluation_at is not None and len(self.evaluated) == self.fail_evaluation_at:
            return SimpleNamespace(exit_code=1, run_dir=None, summary={"status": "operational_failure"})
        outcome = self.outcome(len(self.evaluated)) if callable(self.outcome) else self.outcome
        fight = {"victories": int(outcome == "win"), "deaths": int(outcome == "loss"), "escaped": 0}
        return SimpleNamespace(exit_code=0, run_dir=None,
                               summary={"status": "complete", "fights": {"standalone_fights": fight}})

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
                    disk_free=lambda path: self.free, tree_bytes=lambda path: self.bytes,
                    load_checkpoint=self.load_checkpoint, weights_snapshot=lambda runtime: {"w": 1},
                    weights_equal=lambda before, after: before == after)

    def load_checkpoint(self, path, identity, *, purpose, device):
        self.log.append(("load_checkpoint", purpose, device))
        weights = {"w": 2} if self.corrupt_fresh else {"w": 1}
        return SimpleNamespace(model_tensors=weights,
                               metadata={"training_state": {"optimizer_step": 0, "behavior_policy_version": 0}})


def _run(fakes, *, clock=None, plan_path=PLAN, dependencies=None):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "out"
        output.mkdir()
        summary = baseline_entry.run_baseline(ROOT, plan_path, output,
                                              dependencies=fakes.deps() if dependencies is None else dependencies,
                                              clock=clock or _Clock())
        events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
        written = json.loads((output / "summary.json").read_text())
        return summary, events, written


GIB = 2**30


def _budget_exhausted(*, turn=False):
    """A clean fixed-case budget exhaustion, shaped like the gym-1.1.1 cv2 fresh-initial report."""
    return SimpleNamespace(exit_code=1, run_dir=None, summary={
        "status": "incomplete", "fault": None,
        "stop_reason": "turn_action_budget" if turn else "fixed_case_budget_incomplete",
        "exit_code": 1, "mode": "standalone",
        "case_accounting": {"cases": [{"case_id": "dev-x", "status": "incomplete"}],
                            "complete": 0, "incomplete": 1},
        "collection": {"fault": None, "faults": 0, "uncertain_environment_slots": 0,
                       "stop_reason": "turn_action_budget" if turn else "environment_action_budget"},
        "fights": {"standalone_fights": {"reached_combat_denominator": 1, "completed": 0,
                                         "victories": 0, "deaths": 0, "escaped": 0, "fights": []}},
        "lifecycle": {"schema_id": "spire-learn-neural-lifecycle-v1", "status": "incomplete",
                      "exit_code": 1, "close_errors": []},
    })


def _gpu_run(fakes, free, *, max_wait=14400, clock=None, additional_panels=None):
    """Run with a CUDA config and a gpu_wait plan; ``free(call)`` scripts free bytes."""
    clock = clock or _Clock()
    plan = json.loads((ROOT / PLAN).read_text())
    if additional_panels is not None:
        plan["additional_panels"] = additional_panels
    plan["gpu_wait"] = {"min_free_bytes": 13 * GIB, "poll_seconds": 30, "max_total_seconds": max_wait}
    plan["runner_timeout_seconds"] = plan["wall_seconds"]["total"] + max_wait + 30 + 1200
    load_config = fakes.load_config
    fakes.load_config = lambda path: SimpleNamespace(**vars(load_config(path)), device="cuda:0")
    calls = {"free": 0, "release": 0, "sleep": 0}

    def gpu_free():
        calls["free"] += 1
        return free(calls["free"])

    def release():
        calls["release"] += 1

    def sleep(seconds):
        calls["sleep"] += 1
        clock.now += seconds

    deps = {**fakes.deps(), "gpu_free_bytes": gpu_free, "release_gpu_cache": release, "sleep": sleep}
    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "plan.json"
        plan_path.write_text(json.dumps(plan))
        output = Path(directory) / "out"
        output.mkdir()
        summary = baseline_entry.run_baseline(ROOT, plan_path, output, dependencies=deps, clock=clock)
        events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    return summary, events, calls


class GpuGateTests(unittest.TestCase):
    def test_waits_instead_of_faulting_and_waiting_does_not_consume_the_wall(self):
        # The GPU is busy for longer than the whole training wall, then frees up.
        busy_polls = 500  # 500 x 30 s = 15,000 s > 12,600 s training wall
        summary, events, calls = _gpu_run(_Fakes(decisions_per_segment=4096),
                                          lambda n: 1 * GIB if n <= busy_polls else 20 * GIB,
                                          max_wait=20000)
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(summary["result"], "complete")
        self.assertEqual(calls["sleep"], busy_polls)
        names = [event["event"] for event in events]
        self.assertEqual(names.count("gpu_wait"), 1)
        self.assertEqual(names.count("gpu_resumed"), 1)
        resumed = next(event for event in events if event["event"] == "gpu_resumed")
        self.assertGreaterEqual(resumed["waited_seconds"], busy_polls * 30)
        self.assertGreater(calls["release"], 99)  # after fresh-initial, qualification, segments, cases

    def test_wait_exhaustion_in_training_stops_without_evaluation(self):
        summary, events, calls = _gpu_run(_Fakes(decisions_per_segment=4096),
                                          lambda n: 20 * GIB if n <= 3 else 1 * GIB, max_wait=600)
        self.assertEqual(summary["stop_reason"], "gpu_wait_exhausted")
        self.assertEqual(summary["evaluation_skipped"], "gpu_wait_exhausted")
        self.assertEqual(summary["result"], "incomplete")
        self.assertLessEqual(calls["sleep"], 600 // 30 + 1)

    def test_wait_exhaustion_in_evaluation_marks_the_panel_and_keeps_training(self):
        fakes = _Fakes(decisions_per_segment=4096)
        trained = {"done": False}
        original = fakes.evaluate

        def evaluate(*args, **kwargs):
            trained["done"] = True
            return original(*args, **kwargs)

        fakes.evaluate = evaluate
        # Plenty of memory until the first evaluated case, then the GPU stays busy.
        summary, events, calls = _gpu_run(
            fakes, lambda n: 20 * GIB if not trained["done"] else 1 * GIB, max_wait=300)
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(summary["evaluations"]["final"]["status"], "gpu_wait_exhausted")
        self.assertEqual(summary["result"], "incomplete")

    def test_busy_from_the_start_gates_the_fresh_initial_save(self):
        fakes = _Fakes(decisions_per_segment=4096)
        summary, events, calls = _gpu_run(fakes, lambda n: 1 * GIB, max_wait=120)
        self.assertEqual(summary["stop_reason"], "gpu_wait_exhausted")
        self.assertFalse(any(entry[0] == "create_runtime" for entry in fakes.log))
        self.assertEqual(next(e for e in events if e["event"] == "gpu_wait")["stage"], "fresh_initial")

    def test_cache_is_released_when_a_segment_raises(self):
        fakes = _Fakes(decisions_per_segment=4096, fault_at=2)
        summary, events, calls = _gpu_run(fakes, lambda n: 20 * GIB)
        self.assertEqual(summary["stop_reason"], "fault")
        # fresh-initial, qualification, segment 1 and the faulting segment each release once.
        self.assertEqual(calls["release"], 4)

    def test_cpu_configs_and_plans_without_gpu_wait_never_poll(self):
        fakes = _Fakes(decisions_per_segment=4096)
        deps = {**fakes.deps(), "gpu_free_bytes": lambda: self.fail("polled")}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            summary = baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=deps, clock=_Clock())
        self.assertEqual(summary["result"], "complete")

    def test_gpu_wait_plan_validation(self):
        plan = json.loads((ROOT / PLAN).read_text())
        cases = {
            "malformed": {"min_free_bytes": 1, "poll_seconds": 0, "max_total_seconds": 1},
            "unknown key": {"min_free_bytes": 1, "poll_seconds": 1, "max_total_seconds": 1, "x": 1},
            "timeout too short": {"min_free_bytes": 1, "poll_seconds": 1, "max_total_seconds": 10**6},
        }
        for label, wait in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps({**plan, "gpu_wait": wait}))
                with self.assertRaises(ValueError):
                    baseline_entry.load_plan(path)


class BaselineEntryTests(unittest.TestCase):
    def test_plan_binds_the_8726d467_template(self):
        plan = baseline_entry.load_plan(ROOT / PLAN)
        self.assertEqual(plan["gym_revision"], V8726)
        self.assertEqual(plan["target_learned_decisions"], 405504)
        template = (ROOT / plan["template"]).read_text()
        self.assertIn(V8726, template)
        self.assertIn('device = "cuda:0"', template)

    def test_v1_1_1_plan_repeats_the_8726d467_baseline_on_the_new_build(self):
        plan = baseline_entry.load_plan(ROOT / "configs/neural/baseline-gym-1.1.1/plan.json")
        previous = baseline_entry.load_plan(ROOT / PLAN)
        own = {"baseline_id", "gym_revision", "template", "qualifications"}
        self.assertEqual({k: v for k, v in plan.items() if k not in own},
                         {k: v for k, v in previous.items() if k not in own})
        self.assertEqual(plan["gym_revision"], "e3c13836bd1458be8a87b93160f5c350f78fcd80")
        template = (ROOT / plan["template"]).read_text()
        base_lines = (ROOT / previous["template"]).read_text().splitlines()[1:]
        self.assertEqual(len(base_lines), len(template.splitlines()[1:]))
        changed = [new.split("=", 1)[0].strip()
                   for old, new in zip(base_lines, template.splitlines()[1:]) if old != new]
        self.assertEqual(changed, ["gym_revision", "standalone", "run"])
        self.assertIn("+gym-1.1.1", template)

    def test_v1_1_1_replicate_changes_only_the_seeds(self):
        plan = baseline_entry.load_plan(ROOT / "configs/neural/baseline-gym-1.1.1-r2/plan.json")
        first = baseline_entry.load_plan(ROOT / "configs/neural/baseline-gym-1.1.1/plan.json")
        own = {"baseline_id", "template", "notes", "gpu_wait", "runner_timeout_seconds"}
        self.assertEqual({k: v for k, v in plan.items() if k not in own},
                         {k: v for k, v in first.items() if k not in own})
        self.assertEqual(plan["gpu_wait"], {"min_free_bytes": 13 * 2**30, "poll_seconds": 30,
                                            "max_total_seconds": 21600})
        self.assertEqual(plan["runner_timeout_seconds"], 37800)
        lines = (ROOT / plan["template"]).read_text().splitlines()[1:]
        first_lines = (ROOT / first["template"]).read_text().splitlines()[1:]
        self.assertEqual(len(lines), len(first_lines))
        changed = {new.split("=", 1)[0].strip() for old, new in zip(first_lines, lines) if old != new}
        self.assertEqual(changed, {"policy", "optimizer", "curriculum", "noncombat"})

    def test_cv2_plans_differ_from_r1_r2_only_in_recipe_and_panels(self):
        for suffix in ("", "-r2"):
            with self.subTest(replicate=suffix):
                existing = baseline_entry.load_plan(ROOT / f"configs/neural/baseline-gym-1.1.1{suffix}/plan.json")
                cv2 = baseline_entry.load_plan(ROOT / f"configs/neural/baseline-gym-1.1.1-cv2{suffix}/plan.json")
                ignored = {"baseline_id", "template", "notes", "recipe", "additional_panels",
                           "gpu_wait", "runner_timeout_seconds"}
                self.assertEqual({k: v for k, v in cv2.items() if k not in ignored},
                                 {k: v for k, v in existing.items() if k not in ignored})
                self.assertEqual(cv2["recipe"], "configs/neural/curriculum-v2/recipe.json")
                self.assertEqual(cv2["additional_panels"], ["configs/neural/curriculum-v1/recipe.json"])
                existing_template = (ROOT / existing["template"]).read_text()
                cv2_template = (ROOT / cv2["template"]).read_text()
                self.assertEqual(cv2_template, existing_template)

    def test_cv2_hp_plans_differ_from_cv2_only_in_objective_and_baseline_id(self):
        for suffix in ("", "-r2"):
            with self.subTest(replicate=suffix):
                cv2 = baseline_entry.load_plan(ROOT / f"configs/neural/baseline-gym-1.1.1-cv2{suffix}/plan.json")
                hp = baseline_entry.load_plan(ROOT / f"configs/neural/baseline-gym-1.1.1-cv2-hp{suffix}/plan.json")
                own = {"baseline_id", "template"}
                self.assertEqual({k: v for k, v in hp.items() if k not in own},
                                 {k: v for k, v in cv2.items() if k not in own})
                self.assertEqual(hp["baseline_id"], f"baseline-gym-1.1.1-cv2-hp{suffix}-20261119")
                self.assertEqual(hp["template"],
                                 f"configs/neural/baseline-gym-1.1.1-cv2-hp{suffix}/template.toml")
                cv2_template = (ROOT / cv2["template"]).read_text()
                hp_template = (ROOT / hp["template"]).read_text()
                winloss = 'objective_id = "spire-learn-combat-winloss-v1"'
                self.assertEqual(cv2_template.count(winloss), 1)
                self.assertEqual(
                    hp_template,
                    cv2_template.replace(
                        winloss, 'objective_id = "spire-learn-combat-winloss-hpshaped-v1"')
                    + '\n[objective]\nkind = "hp_shaped"\ncoefficient = 0.5\n')


    def test_v1_1_1_envs64_arm_changes_only_the_rollout_width(self):
        plan = baseline_entry.load_plan(ROOT / "configs/neural/baseline-gym-1.1.1-envs64/plan.json")
        gated = baseline_entry.load_plan(ROOT / "configs/neural/baseline-gym-1.1.1-r2/plan.json")
        own = {"baseline_id", "template", "notes"}
        self.assertEqual({k: v for k, v in plan.items() if k not in own},
                         {k: v for k, v in gated.items() if k not in own})
        first = (ROOT / "configs/neural/baseline-gym-1.1.1/template.toml").read_text().splitlines()[1:]
        lines = (ROOT / plan["template"]).read_text().splitlines()[1:]
        self.assertEqual(len(lines), len(first))
        self.assertEqual([(old, new) for old, new in zip(first, lines) if old != new],
                         [("rows = 16", "rows = 64")])
        # Whole 64 x 32 rollouts fill each 4,096-action segment.
        self.assertEqual(plan["segment"]["max_environment_actions"] % (64 * 32), 0)

    def test_fresh_qualification_then_warm_start_and_resume_until_target(self):
        fakes = _Fakes(decisions_per_segment=4096)
        summary, events, written = _run(fakes)
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(summary["learned_decisions"], 405504)
        purposes = [call["purpose"] for call in fakes.segments]
        self.assertEqual(purposes, ["warm_start", "warm_start"] + ["resume"] * 98)
        self.assertEqual(summary["result"], "complete")
        fresh = summary["checkpoints"][0]["path"]
        self.assertEqual(fakes.segments[0]["parent"], fresh)  # qualification from fresh weights
        self.assertEqual(fakes.segments[1]["parent"], fresh)  # training starts from fresh weights
        self.assertNotEqual(fakes.segments[2]["parent"], fakes.segments[0]["parent"])
        self.assertEqual(fakes.segments[0]["max_env"], 512)
        self.assertEqual(fakes.segments[1]["max_env"], 4096)
        self.assertEqual(fakes.segments[1]["max_updates"], 8)
        # Resumed limits are the parent's cumulative counters plus one segment's allowance.
        self.assertEqual(fakes.segments[2]["max_env"], 8192)
        self.assertEqual(fakes.segments[2]["max_updates"], 16)
        self.assertEqual(fakes.segments[-1]["max_env"], 405504)
        self.assertIn("qualification_passed", [event["event"] for event in events])
        self.assertEqual(written["stop_reason"], "target_reached")
        templates = {entry[4] for entry in fakes.log if entry[0] == "prepare"}
        self.assertEqual(templates, {"template.toml"})

    def test_decisions_counted_from_saved_prefix_not_segment_count(self):
        fakes = _Fakes(decisions_per_segment=3000)  # fewer decisions than the 4,096 allowance
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["segments"], 136)  # 136 * 3000 = 408,000 >= 405,504
        self.assertEqual(summary["learned_decisions"], 408000)

    def test_fixed_limits_would_stall_and_are_stopped(self):
        fakes = _Fakes(decisions_per_segment=4096)
        original = baseline_entry._limits
        baseline_entry._limits = lambda prefix, allowance, deadline: original(None, allowance, deadline)
        try:
            summary, events, _ = _run(fakes)
        finally:
            baseline_entry._limits = original
        self.assertEqual(summary["stop_reason"], "segment_without_progress")
        self.assertEqual(summary["segments"], 2)
        self.assertEqual(summary["learned_decisions"], 4096)
        self.assertEqual(summary["evaluations"]["final"]["status"], "complete")  # stalled run is still measured
        self.assertEqual(summary["result"], "incomplete")

    def test_no_progress_after_training_deadline_is_training_wall(self):
        fakes = _Fakes(decisions_per_segment=0)
        clock = _Clock()
        run_segment = fakes.run_stream_segment
        def expire_after_training_segment(config, **kwargs):
            result = run_segment(config, **kwargs)
            if "train-0-" in str(config.path):
                clock.now = 12602
            return result
        fakes.run_stream_segment = expire_after_training_segment
        summary, _, _ = _run(fakes, clock=clock)
        self.assertEqual(summary["stop_reason"], "training_wall")

    def test_missing_checkpoint_stops_without_restart(self):
        fakes = _Fakes(decisions_per_segment=4096, missing_checkpoint_at=3)
        summary, events, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "segment_without_checkpoint")
        self.assertEqual(len(fakes.segments), 4)  # qualification + 3 training attempts; no restart
        self.assertEqual(summary["learned_decisions"], 8192)
        self.assertIn("segment_without_checkpoint", [event["event"] for event in events])
        self.assertEqual([c["label"] for c in summary["checkpoints"]][:2], ["fresh-initial", "final"])

    def test_qualification_failure_stops_before_training(self):
        fakes = _Fakes(qualification_checkpoint=False)
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "qualification_without_checkpoint")
        self.assertEqual(len(fakes.segments), 1)
        self.assertEqual(summary["learned_decisions"], 0)
        self.assertEqual(summary["evaluations"], {})
        self.assertEqual(summary["evaluation_skipped"], "qualification_without_checkpoint")
        self.assertEqual(summary["result"], "incomplete")

    def test_fault_is_recorded_not_retried(self):
        fakes = _Fakes(fault_at=2)
        summary, events, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "fault")
        self.assertEqual(summary["fault"]["type"], "RuntimeError")
        self.assertEqual(len(fakes.segments), 3)
        self.assertIn("fault", [event["event"] for event in events])

    def test_wall_and_storage_caps(self):
        summary, _, _ = _run(_Fakes(), clock=_Clock(step=5000.0))
        self.assertEqual(summary["stop_reason"], "training_wall")
        summary, _, _ = _run(_Fakes(disk_free=1))
        self.assertEqual(summary["stop_reason"], "disk_free_below_minimum")
        summary, _, _ = _run(_Fakes(tree_bytes=2**40))
        self.assertEqual(summary["stop_reason"], "output_byte_cap")

    def test_retention_and_panel_evaluation(self):
        fakes = _Fakes(decisions_per_segment=4096)
        summary, _, _ = _run(fakes)
        labels = {row["label"]: row["learned_decisions"] for row in summary["checkpoints"]}
        self.assertEqual(labels["fresh-initial"], 0)
        self.assertGreaterEqual(labels["final"], 405504)
        self.assertEqual(labels["near-135168"], 135168)  # 33 segments * 4096
        self.assertEqual(labels["near-270336"], 270336)  # 66 segments * 4096
        self.assertEqual(set(summary["evaluations"]), {"fresh-initial", "final", "near-135168", "near-270336"})
        self.assertEqual(summary["evaluations"]["final"]["cases_evaluated"], 3)
        self.assertEqual(len(fakes.evaluated), 12)
        self.assertEqual(fakes.evaluated[0].split("/")[-1], "config-0.toml")
        self.assertEqual(summary["evaluations"]["final"]["cases"]["dev-0"]["outcome"], "win")
        evaluate_calls = [entry[1] for entry in fakes.log
                          if entry[0] == "create_runtime" and entry[1].get("purpose") == "evaluate"]
        self.assertTrue(all(call["action_selection"] == "deterministic" for call in evaluate_calls))

    def test_evaluation_order_is_derived_from_the_plan(self):
        cv2 = baseline_entry.load_plan(ROOT / "configs/neural/baseline-gym-1.1.1-cv2/plan.json")
        self.assertEqual(baseline_entry._evaluation_order(cv2),
                         ("final", "fresh-initial", "near-135168", "near-270336"))
        for name in ("baseline-8726d467", "baseline-gym-1.1.1-cv2-r2", "baseline-gym-1.1.1-cv2-hp"):
            plan = baseline_entry.load_plan(ROOT / "configs/neural" / name / "plan.json")
            self.assertEqual(baseline_entry._evaluation_order(plan),
                             ("final", "fresh-initial", "near-135168", "near-270336"), name)
        standard = baseline_entry.load_plan(ROOT / "configs/neural/baseline-standard-1.1.1/plan.json")
        self.assertEqual(baseline_entry._evaluation_order(standard),
                         ("final", "fresh-initial", "near-405504", "near-811008", "near-1216512"))

    def test_plan_declared_near_labels_are_evaluated_and_can_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan = json.loads((ROOT / PLAN).read_text())
            plan["retain_near_decisions"] = [245760, 81920, 163840]  # unsorted on purpose
            plan_path.write_text(json.dumps(plan))
            fakes = _Fakes(decisions_per_segment=4096, panel_cases=1)
            summary, _, written = _run(fakes, plan_path=plan_path)
        self.assertEqual([row["label"] for row in summary["checkpoints"]],
                         ["fresh-initial", "final", "near-245760", "near-81920", "near-163840"])
        self.assertEqual(list(summary["evaluations"]),
                         ["final", "fresh-initial", "near-81920", "near-163840", "near-245760"])
        self.assertTrue(all(row["status"] == "complete" for row in summary["evaluations"].values()))
        self.assertEqual(summary["result"], "complete")
        self.assertEqual(written["result"], "complete")

    def test_evaluation_fault_still_writes_summary(self):
        fakes = _Fakes(decisions_per_segment=405504)

        def broken(config, **kwargs):
            raise ValueError("evaluation broke")

        fakes.evaluate = broken
        summary, events, written = _run(fakes)
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(written["evaluation_fault"]["type"], "ValueError")
        self.assertIn("evaluation_fault", [event["event"] for event in events])

    def test_final_is_evaluated_first_and_paired_with_fresh(self):
        fakes = _Fakes(decisions_per_segment=4096, panel_cases=3)
        # Final evaluates cases 1-3 first, fresh-initial cases 4-6: final wins all, fresh wins only the first.
        fakes.outcome = lambda n: "win" if n <= 3 or n == 4 else "loss"
        summary, _, _ = _run(fakes)
        paired = summary["paired_fresh_vs_final"]
        self.assertEqual((paired["both_win"], paired["final_only_win"], paired["fresh_only_win"]), (1, 2, 0))
        self.assertEqual(paired["paired_cases"], 3)
        self.assertAlmostEqual(paired["exact_mcnemar_two_sided_p"], 0.5)

    def test_additional_panel_is_evaluated_for_final_and_fresh(self):
        recipe = "configs/neural/curriculum-v1/recipe.json"
        fakes = _Fakes(decisions_per_segment=405504, panel_cases=1)
        busy_once = {"done": False}

        def free(_call):
            if len(fakes.evaluated) >= 4 and not busy_once["done"]:
                busy_once["done"] = True
                return 1 * GIB
            return 20 * GIB

        summary, events, _ = _gpu_run(fakes, free, additional_panels=[recipe])
        panel = summary["additional_panels"]["configs/neural/curriculum-v1/recipe.json"]
        self.assertEqual(list(panel["evaluations"]), ["final", "fresh-initial"])
        self.assertTrue(all(row["status"] == "complete" for row in panel["evaluations"].values()))
        self.assertIn("paired_fresh_vs_final", panel)
        self.assertEqual(panel["paired_fresh_vs_final"]["paired_cases"], 1)
        self.assertEqual(summary["result"], "complete")
        completed = [event for event in events if event["event"] == "evaluation_complete"]
        self.assertEqual([event["panel"] for event in completed[:4]], ["primary"] * 4)
        self.assertEqual([event["panel"] for event in completed[4:]], [recipe, recipe])
        extra_wait = next(event for event in events if event["event"] == "gpu_wait"
                          and event["stage"].startswith("evaluation final"))
        self.assertIn(recipe, extra_wait["stage"])

    def test_failing_additional_panel_keeps_result_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan = json.loads((ROOT / PLAN).read_text())
            plan["additional_panels"] = ["configs/neural/curriculum-v1/recipe.json"]
            plan_path.write_text(json.dumps(plan))
            fakes = _Fakes(decisions_per_segment=405504, panel_cases=2)
            fakes.fail_evaluation_at = 9  # four two-case primary labels, then extra final's first case
            summary, _, written = _run(fakes, plan_path=plan_path)
        self.assertEqual(summary["evaluations"]["final"]["status"], "complete")
        self.assertEqual(summary["evaluations"]["fresh-initial"]["status"], "complete")
        extra = summary["additional_panels"]["configs/neural/curriculum-v1/recipe.json"]
        self.assertEqual(extra["evaluations"]["final"]["status"], "operational_failure")
        self.assertNotIn("fresh-initial", extra["evaluations"])
        self.assertEqual(summary["result"], "incomplete")
        self.assertEqual(written["evaluations"], summary["evaluations"])

    def test_additional_recipe_load_exception_preserves_primary_evaluations(self):
        missing_recipe = "configs/neural/missing-extra-recipe.json"
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory) / "plan.json"
            plan = json.loads((ROOT / PLAN).read_text())
            plan["additional_panels"] = [missing_recipe]
            plan_path.write_text(json.dumps(plan))

            control, _, _ = _run(_Fakes(decisions_per_segment=405504, panel_cases=1))
            fakes = _Fakes(decisions_per_segment=405504, panel_cases=1)
            dependencies = fakes.deps()
            load_recipe = dependencies["load_recipe"]

            def load_recipe_with_missing_extra(path):
                if Path(path) == ROOT / missing_recipe:
                    raise FileNotFoundError(path)
                return load_recipe(path)

            dependencies["load_recipe"] = load_recipe_with_missing_extra
            summary, events, _ = _run(fakes, plan_path=plan_path, dependencies=dependencies)

        def stable_evaluations(result):
            return {
                label: {
                    "status": evaluation["status"],
                    "cases": {case_id: row["outcome"] for case_id, row in evaluation["cases"].items()},
                    "summary": evaluation["summary"],
                }
                for label, evaluation in result["evaluations"].items()
            }

        self.assertEqual(stable_evaluations(summary), stable_evaluations(control))
        self.assertTrue(all(row["status"] == "complete" for row in summary["evaluations"].values()))
        self.assertEqual(summary["additional_panels"][missing_recipe]["fault"]["type"], "FileNotFoundError")
        self.assertEqual(summary["result"], "incomplete")
        self.assertIn("additional_panel_fault", [event["event"] for event in events])
        primary_completions = [event for event in events
                               if event["event"] == "evaluation_complete" and event.get("panel") == "primary"]
        self.assertEqual(len(primary_completions), 4)

    def test_returned_evaluation_fault_stops_further_evaluation(self):
        fakes = _Fakes(decisions_per_segment=4096)
        fakes.fail_evaluation_at = 2
        summary, events, _ = _run(fakes)
        self.assertEqual(summary["evaluations"]["final"]["status"], "operational_failure")
        self.assertEqual(len(fakes.evaluated), 2)
        self.assertNotIn("fresh-initial", summary["evaluations"])
        self.assertEqual(summary["result"], "incomplete")

    def test_storage_stop_during_evaluation(self):
        fakes = _Fakes(decisions_per_segment=4096)
        calls = {"n": 0}

        def free(path):
            calls["n"] += 1
            # 1 initial + 99 per-segment checks reach 405,504; the first evaluation check is exhausted.
            return 10**12 if calls["n"] <= 100 else 1

        deps = fakes.deps()
        deps["disk_free"] = free
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            summary = baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=deps, clock=_Clock())
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(summary["evaluations"]["final"]["status"], "stopped_disk_free_below_minimum")
        self.assertEqual(fakes.evaluated, [])

    def test_interrupt_still_writes_summary(self):
        fakes = _Fakes()

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        fakes.run_stream_segment = interrupt
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            with self.assertRaises(KeyboardInterrupt):
                baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=fakes.deps(), clock=_Clock())
            written = json.loads((output / "summary.json").read_text())
        self.assertEqual(written["result"], "interrupted")
        self.assertEqual(written["interrupted"], "KeyboardInterrupt")

    def test_prepared_directory_failure_still_writes_summary(self):
        fakes = _Fakes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            (output / "prepared").mkdir()
            with self.assertRaises(FileExistsError):
                baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=fakes.deps(), clock=_Clock())
            summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["result"], "interrupted")
        self.assertEqual(summary["interrupted"], "FileExistsError")

    def test_baseline_requires_absolute_root_and_output_and_runner_output_match(self):
        fakes = _Fakes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "root must be absolute"):
                baseline_entry.run_baseline(Path("relative-root"), PLAN, output,
                                            dependencies=fakes.deps(), clock=_Clock())
            with self.assertRaisesRegex(ValueError, "output must be absolute"):
                baseline_entry.run_baseline(ROOT, PLAN, Path("relative-output"),
                                            dependencies=fakes.deps(), clock=_Clock())
            with patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_OUTPUT": str(output / "other")}):
                with self.assertRaisesRegex(ValueError, "must match SPIRE_LEARN_RUNNER_OUTPUT"):
                    baseline_entry.run_baseline(ROOT, PLAN, output,
                                                dependencies=fakes.deps(), clock=_Clock())

    def test_baseline_requires_runner_timeout_at_least_plan_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            with patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS": "15599"}):
                with self.assertRaisesRegex(ValueError, "at least 15600"):
                    baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=_Fakes().deps(),
                                                clock=_Clock())

    def test_cli_rejects_mismatched_runner_output_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            stderr = io.StringIO()
            with patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_OUTPUT": str(output / "other"),
                                         "SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS": "15600"}), \
                 patch.object(baseline_entry, "run_baseline", return_value={
                     "result": "complete", "stop_reason": "target_reached",
                     "learned_decisions": 1, "segments": 1,
                 }) as run_baseline, \
                 redirect_stderr(stderr):
                status = baseline_entry.main(["--plan", str(PLAN), "--output", str(output),
                                              "--root", str(ROOT)])
            self.assertEqual(status, 2)
            self.assertFalse(run_baseline.called)
            self.assertIn("must match SPIRE_LEARN_RUNNER_OUTPUT", stderr.getvalue())

    def test_cli_rejects_runner_timeout_shorter_than_plan_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            output.mkdir()
            stderr = io.StringIO()
            with patch.dict(os.environ, {"SPIRE_LEARN_RUNNER_OUTPUT": str(output),
                                         "SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS": "15599"}), \
                 patch.object(baseline_entry, "run_baseline", return_value={
                     "result": "complete", "stop_reason": "target_reached",
                     "learned_decisions": 1, "segments": 1,
                 }) as run_baseline, \
                 redirect_stderr(stderr):
                status = baseline_entry.main(["--plan", str(PLAN), "--output", str(output),
                                              "--root", str(ROOT)])
            self.assertEqual(status, 2)
            self.assertFalse(run_baseline.called)
            self.assertIn("at least 15600", stderr.getvalue())

    def test_returned_segment_fault_is_not_evaluated(self):
        fakes = _Fakes(decisions_per_segment=4096)
        fakes.failed_segment_at = 3
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "segment_fault")
        self.assertEqual(summary["evaluations"], {})
        self.assertEqual(fakes.evaluated, [])
        self.assertEqual(summary["result"], "incomplete")

    def test_report_without_fights_is_invalid(self):
        fakes = _Fakes(decisions_per_segment=4096)
        original = fakes.evaluate

        def no_fights(config, **kwargs):
            original(config, **kwargs)
            return SimpleNamespace(exit_code=0, run_dir=None, summary={"status": "complete"})

        fakes.evaluate = no_fights
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["evaluations"]["final"]["status"], "invalid_report")
        self.assertEqual(len(fakes.evaluated), 1)
        self.assertEqual(summary["result"], "incomplete")
        self.assertNotIn("paired_fresh_vs_final", summary)

    def test_summary_error_keeps_result_incomplete(self):
        fakes = _Fakes(decisions_per_segment=4096)

        def broken_summary(panel, reports):
            raise ValueError("healthy development report is missing fight metrics")

        fakes.summarize = broken_summary
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["evaluations"]["final"]["status"], "summary_error")
        self.assertEqual(summary["result"], "incomplete")

    def test_censored_cases_are_excluded_from_pairing(self):
        fakes = _Fakes(decisions_per_segment=4096, panel_cases=3)
        # Final: win, win, censored; fresh: loss, loss, loss.
        fakes.outcome = lambda n: {1: "win", 2: "win", 3: "incomplete"}.get(n, "loss")
        summary, _, _ = _run(fakes)
        paired = summary["paired_fresh_vs_final"]
        self.assertEqual(paired["paired_cases"], 2)
        self.assertEqual(paired["excluded_censored"], 1)
        self.assertEqual(paired["excluded_failed_invalid_or_missing"], 0)
        self.assertEqual(paired["final_only_win"], 2)

    def test_clean_case_budget_exhaustion_is_censored_and_evaluation_continues(self):
        for turn, reason in ((False, "environment_action_budget"), (True, "turn_action_budget")):
            with self.subTest(reason=reason):
                fakes = _Fakes(decisions_per_segment=4096, panel_cases=3)
                original = fakes.evaluate

                def censor_fresh_second_case(config, **kwargs):
                    report = original(config, **kwargs)
                    # Final runs cases 1-3; fresh-initial runs 4-6; its second case exhausts the case budget.
                    return _budget_exhausted(turn=turn) if len(fakes.evaluated) == 5 else report

                fakes.evaluate = censor_fresh_second_case
                summary, events, _ = _run(fakes)
                fresh = summary["evaluations"]["fresh-initial"]
                self.assertEqual(fresh["status"], "complete")
                self.assertEqual(fresh["cases_evaluated"], 3)
                self.assertEqual(fresh["cases"]["dev-1"]["outcome"], "incomplete")
                self.assertEqual(fresh["cases"]["dev-1"]["censor_reason"], reason)
                self.assertEqual(len(fakes.evaluated), 12)  # every label still evaluated every case
                paired = summary["paired_fresh_vs_final"]
                self.assertEqual((paired["paired_cases"], paired["excluded_censored"],
                                  paired["excluded_failed_invalid_or_missing"]), (2, 1, 0))
                self.assertEqual(summary["result"], "complete")

    def test_unclean_incomplete_reports_still_fail(self):
        variants = {
            "fault": lambda s: s.update(fault={"type": "RuntimeError", "message": "boom"}),
            "other stop reason": lambda s: s.update(stop_reason="wall_budget"),
            "missing stop reason": lambda s: s.pop("stop_reason"),
            "close error": lambda s: s["lifecycle"].update(close_errors=[{"type": "OSError"}]),
            "lifecycle operation error": lambda s: s["lifecycle"].update(operation_error={"type": "X"}),
            "lifecycle not incomplete": lambda s: s["lifecycle"].update(status="operational_failure"),
            "lifecycle receipt pending": lambda s: s["lifecycle"].update(status="receipt_pending"),
            "receipt write error": lambda s: s.update(lifecycle_receipt_error={"type": "OSError"}),
            "collection fault": lambda s: s["collection"].update(fault={"type": "X"}, faults=1),
            "uncertain slots": lambda s: s["collection"].update(uncertain_environment_slots=1),
            "status not incomplete": lambda s: s.update(status="operational_failure"),
        }
        for name, mutate in variants.items():
            with self.subTest(name):
                report = _budget_exhausted()
                mutate(report.summary)
                self.assertEqual(baseline_entry._case_outcome(report), "failed")
        self.assertEqual(baseline_entry._case_outcome(_budget_exhausted()), "incomplete")

    def test_faulted_incomplete_case_stops_as_operational_failure(self):
        fakes = _Fakes(decisions_per_segment=4096, panel_cases=3)
        original = fakes.evaluate

        def faulted_second_case(config, **kwargs):
            report = original(config, **kwargs)
            if len(fakes.evaluated) != 2:
                return report
            faulted = _budget_exhausted()
            faulted.summary["fault"] = {"type": "RuntimeError", "message": "simulator fault"}
            return faulted

        fakes.evaluate = faulted_second_case
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["evaluations"]["final"]["status"], "operational_failure")
        self.assertEqual(summary["evaluations"]["final"]["cases"]["dev-1"]["outcome"], "failed")
        self.assertEqual(len(fakes.evaluated), 2)
        self.assertNotIn("fresh-initial", summary["evaluations"])
        self.assertEqual(summary["result"], "incomplete")

    def test_failed_retention_panel_keeps_result_incomplete(self):
        fakes = _Fakes(decisions_per_segment=4096, panel_cases=3)
        fakes.fail_evaluation_at = 7  # first case of the third label, after final and fresh-initial
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["evaluations"]["final"]["status"], "complete")
        self.assertEqual(summary["evaluations"]["fresh-initial"]["status"], "complete")
        self.assertEqual(summary["evaluations"]["near-135168"]["status"], "operational_failure")
        self.assertNotIn("near-270336", summary["evaluations"])
        self.assertEqual(summary["result"], "incomplete")

    def test_exception_mid_panel_preserves_partial_rows(self):
        fakes = _Fakes(decisions_per_segment=4096)
        original = fakes.evaluate

        def explode_second(config, **kwargs):
            if len(fakes.evaluated) == 1:
                fakes.evaluated.append(config.path)
                raise RuntimeError("case exploded")
            return original(config, **kwargs)

        fakes.evaluate = explode_second
        summary, _, written = _run(fakes)
        self.assertEqual(written["evaluation_fault"]["type"], "RuntimeError")
        self.assertEqual(list(written["evaluations"]["final"]["cases"]), ["dev-0"])
        self.assertEqual(written["evaluations"]["final"]["status"], "fault")
        self.assertEqual(written["result"], "incomplete")

    def test_fresh_roundtrip_mismatch_stops_before_training(self):
        fakes = _Fakes()
        fakes.corrupt_fresh = True
        summary, _, _ = _run(fakes)
        self.assertEqual(summary["stop_reason"], "fresh_initial_roundtrip_mismatch")
        self.assertEqual(fakes.segments, [])


def _prior_output(directory, *, plan=None, stop_reason="target_reached", missing=()):
    """A finished prior run: summary.json plus checkpoint directories, made read-only by tests."""
    prior = Path(directory) / "prior"
    prior.mkdir()
    checkpoints = []
    for label, decisions in (("fresh-initial", 0), ("final", 405504),
                             ("near-135168", 135168), ("near-270336", 270336)):
        path = prior / label
        if label not in missing:
            path.mkdir()
        checkpoints.append({"label": label, "path": str(path), "learned_decisions": decisions})
    plan = json.loads((ROOT / PLAN).read_text()) if plan is None else plan
    summary = {"schema": "spire-learn-baseline-summary-v1", "plan": plan, "stop_reason": stop_reason,
               "checkpoints": checkpoints, "segments": 99, "learned_decisions": 405504,
               "config_sha256": "a" * 64, "device": "cuda:0",
               "evaluations": {"fresh-initial": {"status": "operational_failure"}},
               "result": "incomplete"}
    (prior / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return prior, summary


def _tree(path):
    return sorted((str(p.relative_to(path)), p.stat().st_mtime_ns) for p in Path(path).rglob("*"))


class ReevaluateTests(unittest.TestCase):
    def _reevaluate(self, fakes, prior, directory, *, plan_path=PLAN):
        output = Path(directory) / "new"
        output.mkdir()
        summary = baseline_entry.run_baseline(ROOT, plan_path, output, dependencies=fakes.deps(),
                                              clock=_Clock(), reevaluate=prior)
        return summary, output

    def test_reevaluate_writes_only_to_new_output_and_evaluates_prior_checkpoints(self):
        fakes = _Fakes(panel_cases=2)
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory)
            before = _tree(prior)
            summary, output = self._reevaluate(fakes, prior, directory)
            self.assertEqual(_tree(prior), before)
            written = json.loads((output / "summary.json").read_text())
            self.assertTrue((output / "progress.jsonl").is_file())
            self.assertTrue((output / "panel").is_dir())
            events = [json.loads(line)["event"] for line in (output / "progress.jsonl").read_text().splitlines()]
            self.assertIn("reevaluation_started", events)
            self.assertTrue(all(Path(path).is_relative_to(output) for path in fakes.evaluated))
        self.assertEqual(fakes.segments, [])  # nothing trains
        self.assertFalse(any(entry[0] == "save" for entry in fakes.log))
        evaluated = [str(entry[1]["checkpoint"]) for entry in fakes.log
                     if entry[0] == "create_runtime" and entry[1].get("purpose") == "evaluate"]
        expected = [str(prior / label) for label in ("final", "fresh-initial", "near-135168", "near-270336")
                    for _ in range(2)]
        self.assertEqual(evaluated, expected)
        self.assertEqual(set(summary["evaluations"]), {"final", "fresh-initial", "near-135168", "near-270336"})
        self.assertEqual(summary["result"], "complete")
        self.assertEqual(written["result"], "complete")

    def test_reevaluate_copies_the_training_record(self):
        fakes = _Fakes(panel_cases=1)
        with tempfile.TemporaryDirectory() as directory:
            prior, previous = _prior_output(directory, stop_reason="training_wall")
            previous["fault"] = {"type": "X", "message": "recorded"}
            (prior / "summary.json").write_text(json.dumps(previous))
            digest = __import__("hashlib").sha256((prior / "summary.json").read_bytes()).hexdigest()
            summary, _ = self._reevaluate(fakes, prior, directory)
        for key in ("stop_reason", "learned_decisions", "segments", "checkpoints", "config_sha256",
                    "device", "fault"):
            self.assertEqual(summary[key], previous[key], key)
        self.assertEqual(summary["reevaluated_from"], str(prior))
        self.assertEqual(summary["reevaluated_from_summary_sha256"], digest)
        self.assertEqual(summary["evaluations"]["final"]["status"], "complete")
        self.assertEqual(summary["result"], "incomplete")  # training_wall never completes, as in a normal run

    def test_reevaluate_allows_only_adding_cap_to_target_reached_prior(self):
        import tomllib

        fakes = _Fakes(panel_cases=2)
        plan = baseline_entry.load_plan(ROOT / PLAN)
        with tempfile.TemporaryDirectory() as directory:
            prior, previous = _prior_output(directory, plan=plan)
            updated = {**plan, "max_turn_actions": 500}
            new_plan = Path(directory) / "plan.json"
            new_plan.write_text(json.dumps(updated))
            summary, output = self._reevaluate(fakes, prior, directory, plan_path=new_plan)
            self.assertEqual(fakes.segments, [])
            self.assertEqual(summary["result"], "complete")
            self.assertEqual(set(summary["evaluations"]), {row["label"] for row in previous["checkpoints"]})
            for path in (output / "prepared/template.toml", output / "panel/config-0.toml"):
                self.assertEqual(tomllib.loads(path.read_text())["budgets"]["max_turn_actions"], 500)

    def test_reevaluate_rejects_ineligible_turn_cap_migrations(self):
        plan = baseline_entry.load_plan(ROOT / PLAN)
        capped = {**plan, "max_turn_actions": 500}
        with tempfile.TemporaryDirectory() as directory:
            prior, previous = _prior_output(directory, plan=plan)
            output = Path(directory) / "new"
            missing_label = [row for row in previous["checkpoints"] if row["label"] != "final"]
            for name, changes, requested, error in (
                ("training wall", {"stop_reason": "training_wall"}, capped, "different plan"),
                ("training fault", {"fault": {"type": "X"}}, capped, "different plan"),
                ("missing label", {"checkpoints": missing_label}, capped, "every scheduled checkpoint"),
                ("different cap", {"plan": {**plan, "max_turn_actions": 499}}, capped, "different plan"),
                ("removed cap", {"plan": capped}, plan, "different plan"),
                ("other plan change", {}, {**capped, "target_learned_decisions": 123}, "different plan"),
            ):
                with self.subTest(name=name):
                    (prior / "summary.json").write_text(json.dumps({**previous, **changes}))
                    with self.assertRaisesRegex(ValueError, error):
                        baseline_entry.load_prior(prior, requested, output)

    def test_reevaluate_rejects_plan_mismatch(self):
        plan = json.loads((ROOT / PLAN).read_text())
        changed = {**plan, "target_learned_decisions": plan["target_learned_decisions"] + 1}
        fakes = _Fakes(panel_cases=1)
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory, plan=changed)
            output = Path(directory) / "new"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "plan"):
                baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=fakes.deps(),
                                            clock=_Clock(), reevaluate=prior)
            self.assertEqual(list(output.iterdir()), [])
        self.assertEqual(fakes.evaluated, [])

    def test_reevaluate_rejects_unevaluable_stop_and_output_inside_prior(self):
        fakes = _Fakes(panel_cases=1)
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory, stop_reason="segment_fault")
            output = Path(directory) / "new"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "stop"):
                baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=fakes.deps(),
                                            clock=_Clock(), reevaluate=prior)
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory)
            for target in (prior, prior / "nested"):
                target.mkdir(exist_ok=True)
                with self.assertRaisesRegex(ValueError, "prior output"):
                    baseline_entry.run_baseline(ROOT, PLAN, target, dependencies=fakes.deps(),
                                                clock=_Clock(), reevaluate=prior)
            self.assertFalse((prior / "progress.jsonl").exists())
        self.assertEqual(fakes.evaluated, [])

    def test_reevaluate_missing_checkpoint_fails_closed(self):
        fakes = _Fakes(panel_cases=1)
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory, missing=("near-135168",))
            output = Path(directory) / "new"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "near-135168"):
                baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=fakes.deps(),
                                            clock=_Clock(), reevaluate=prior)
            self.assertEqual(list(output.iterdir()), [])
        self.assertEqual(fakes.evaluated, [])

    def test_reevaluate_checkpoint_failing_preflight_fails_closed(self):
        fakes = _Fakes(panel_cases=1)

        def preflight(checkpoint, identity, *, purpose):
            if Path(checkpoint).name == "near-270336":
                raise ValueError("checkpoint identity mismatch")

        fakes.preflight = preflight
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory)
            summary, output = self._reevaluate(fakes, prior, directory)
            written = json.loads((output / "summary.json").read_text())
        self.assertEqual(fakes.evaluated, [])
        self.assertEqual(summary["evaluations"], {})
        self.assertEqual(summary["reevaluation_refused"]["type"], "ValueError")
        self.assertIn("near-270336", summary["reevaluation_refused"]["message"])
        self.assertEqual(summary["result"], "incomplete")
        self.assertEqual(written["result"], "incomplete")

    def test_reevaluate_respects_the_total_wall(self):
        fakes = _Fakes(panel_cases=1)
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory)
            output = Path(directory) / "new"
            output.mkdir()
            summary = baseline_entry.run_baseline(ROOT, PLAN, output, dependencies=fakes.deps(),
                                                  clock=_Clock(step=20000.0), reevaluate=prior)
        self.assertEqual(summary["evaluations"]["final"]["status"], "skipped_wall")
        self.assertEqual(fakes.evaluated, [])
        self.assertEqual(summary["result"], "incomplete")

    def test_cli_passes_reevaluate_and_validates_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            prior, _ = _prior_output(directory)
            output = Path(directory) / "new"
            output.mkdir()
            env = {"SPIRE_LEARN_RUNNER_OUTPUT": str(output), "SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS": "15600"}
            result = {"result": "complete", "stop_reason": "target_reached", "learned_decisions": 1,
                      "segments": 1}
            with patch.dict(os.environ, env), \
                 patch.object(baseline_entry, "run_baseline", return_value=result) as run_baseline:
                status = baseline_entry.main(["--plan", str(PLAN), "--output", str(output),
                                              "--root", str(ROOT), "--reevaluate", str(prior)])
            self.assertEqual(status, 0)
            self.assertEqual(run_baseline.call_args.kwargs["reevaluate"], prior)
            self.assertEqual(run_baseline.call_args.kwargs["panel"], "selection")
            with patch.dict(os.environ, env), \
                 patch.object(baseline_entry, "run_baseline", return_value=result) as run_baseline:
                status = baseline_entry.main(["--plan", str(PLAN), "--output", str(output),
                                              "--root", str(ROOT), "--reevaluate", str(prior),
                                              "--panel", "boss"])
            self.assertEqual(status, 0)
            self.assertEqual(run_baseline.call_args.kwargs["reevaluate"], prior)
            self.assertEqual(run_baseline.call_args.kwargs["panel"], "boss")
            stderr = io.StringIO()
            with patch.dict(os.environ, env), redirect_stderr(stderr), \
                 patch.object(baseline_entry, "run_baseline", return_value=result) as run_baseline:
                status = baseline_entry.main(["--plan", str(PLAN), "--output", str(output),
                                              "--root", str(ROOT), "--reevaluate",
                                              str(Path(directory) / "absent")])
            self.assertEqual(status, 2)
            self.assertFalse(run_baseline.called)


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


class PanelRecipeRevisionTests(unittest.TestCase):
    def test_panel_recipe_must_match_the_plan_revision(self):
        plan = {"gym_revision": "e3c13836bd1458be8a87b93160f5c350f78fcd80", "evaluation": {}}
        deps = {"load_recipe": lambda path: SimpleNamespace(gym_revision="4d198064b3b568b6f0d08d6e3937d236fbc48e9b"),
                "build_panel": lambda recipe: self.fail("the panel must not be built")}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Gym"):
                baseline_entry._evaluate({}, deps, plan, None, "recipe.json", Path(folder), None, None, None,
                                         None, None, labels=())



def _output_digest(output: Path, replacements: dict[str, str], extra: object) -> str:
    """sha256 over every written file plus the scripted call log, with run-specific text normalised."""
    import hashlib

    def normal(text: str) -> str:
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    digest = hashlib.sha256()
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(output)).encode() + b"\0")
        digest.update(normal(path.read_text()).encode() + b"\0")
    digest.update(normal(json.dumps(extra, sort_keys=True, default=str)).encode())
    return digest.hexdigest()


class _ScriptedReports:
    """Evaluation reports by call number: win, loss, a turn-cap censor, escaped, ... (cycling)."""

    def __init__(self, fakes, cycle, *, fail_at=None, raise_at=None):
        self.fakes, self.cycle, self.fail_at, self.raise_at, self.calls = fakes, cycle, fail_at, raise_at, 0

    def __call__(self, config, *, runtime, clock, allow_final):
        self.calls += 1
        if self.raise_at == self.calls:
            raise RuntimeError(f"scripted evaluation fault at call {self.calls}")
        self.fakes.evaluated.append(config.path)
        self.fakes.log.append(("evaluate", config.path, type(runtime).__name__,
                               runtime.operation_limits.max_environment_actions,
                               runtime.operation_limits.monotonic_deadline, allow_final))
        if self.fail_at == self.calls:
            return SimpleNamespace(exit_code=1, run_dir=None, summary={"status": "operational_failure"})
        kind = self.cycle[(self.calls - 1) % len(self.cycle)]
        if kind == "censored":
            return _budget_exhausted(turn=True)
        fight = {"victories": int(kind == "win"), "deaths": int(kind == "loss"), "escaped": int(kind == "escaped")}
        return SimpleNamespace(exit_code=0, run_dir=f"run-{self.calls}",
                               summary={"status": "complete", "fights": {"standalone_fights": fight}})


class BaselineEvalByteIdentityTests(unittest.TestCase):
    """Pins every byte the baseline's panel evaluation writes, recorded before panel_eval was extracted."""

    # Recorded at 65e198d, with baseline_entry._evaluate still inline.
    PINS = {
        "selection": "44fc90297c679f8c87a0681e7caa30c1c2a7e05b6c146ccfe12df645fa474dba",
        "boss": "b02fa1d9d652d8f10e98f33149d126aa62ab59d26467ab04b51fde020d4a2bad",
    }

    def _deps(self, fakes, reports, log):
        clock = _Clock()

        def preflight(checkpoint, identity, *, purpose):
            log.append(("preflight", str(checkpoint), identity, purpose))

        def build_panel(recipe, *args, **kwargs):
            log.append(("build_panel", recipe, args, kwargs))
            cases = [{"case_id": f"case-{i}", "environment_seed": i} for i in range(4)]
            return {"schema": "spire-learn-standard-panel-v2", "panel_sha256": "p" * 64,
                    "manifest_sha256": "m" * 64,
                    "manifest": {"schema": "spire-learn-neural-cases-v1", "split": "selection",
                                 "partition": "selection", "cases": cases}}

        def summarize(panel, reports_by_case):
            return {"status": "complete", "panel_sha256": panel["panel_sha256"], "cases": sorted(reports_by_case),
                    "outcomes": {case: getattr(report, "run_dir", None) for case, report in reports_by_case.items()}}

        free = {"calls": 0, "busy": 3}

        def gpu_free():
            # Busy for three polls before the third evaluated case, so the gate waits mid-panel.
            free["calls"] += 1
            log.append(("gpu_free", free["calls"]))
            if reports.calls == 2 and free["busy"]:
                free["busy"] -= 1
                return 1 * GIB
            return 20 * GIB

        def sleep(seconds):
            log.append(("sleep", seconds))
            clock.now += seconds

        load_config = fakes.load_config
        deps = {**fakes.deps(), "evaluate": reports, "preflight": preflight, "build_panel": build_panel,
                "summarize": summarize, "gpu_free_bytes": gpu_free, "sleep": sleep,
                "release_gpu_cache": lambda: log.append(("release",)),
                "load_config": lambda path: SimpleNamespace(**vars(load_config(path)), device="cuda:0")}
        return deps, clock

    def _plan(self, directory):
        plan = json.loads((ROOT / PLAN).read_text())
        plan["max_turn_actions"] = 500
        plan["additional_panels"] = ["configs/neural/curriculum-v1/recipe.json",
                                     {"recipe": "configs/neural/curriculum-v2/recipe.json", "case_actions": 64,
                                      "case_seconds": 7}]
        plan["gpu_wait"] = {"min_free_bytes": 13 * GIB, "poll_seconds": 30, "max_total_seconds": 14400}
        plan["runner_timeout_seconds"] = plan["wall_seconds"]["total"] + 14400 + 30 + 1200
        path = Path(directory) / "plan.json"
        path.write_text(json.dumps(plan))
        return plan, path

    def test_baseline_eval_outputs_unchanged(self):
        digests = {}
        with tempfile.TemporaryDirectory() as directory:
            plan, plan_path = self._plan(directory)
            fakes = _Fakes(panel_cases=4)
            log = fakes.log
            # 16 primary cases, then 8 per additional panel: call 27 faults the second extra panel.
            reports = _ScriptedReports(fakes, ["win", "loss", "censored", "escaped", "win"], raise_at=27)
            deps, clock = self._deps(fakes, reports, log)
            output = Path(directory) / "selection"
            output.mkdir()
            baseline_entry.run_baseline(ROOT, plan_path, output, dependencies=deps, clock=clock)
            digests["selection"] = _output_digest(output, {directory: "<TMP>"}, log)

        with tempfile.TemporaryDirectory() as directory:
            plan, plan_path = self._plan(directory)
            prior, _ = _prior_output(directory, plan=plan)
            fakes = _Fakes(panel_cases=4)
            log = fakes.log
            reports = _ScriptedReports(fakes, ["loss", "escaped", "win"], fail_at=11)
            deps, clock = self._deps(fakes, reports, log)
            output = Path(directory) / "boss"
            output.mkdir()
            baseline_entry.run_baseline(ROOT, plan_path, output, dependencies=deps, clock=clock,
                                        reevaluate=prior, panel="boss")
            # The prior summary embeds the temp path, so its recorded sha256 varies per run.
            prior_sha256 = hashlib.sha256((prior / "summary.json").read_bytes()).hexdigest()
            digests["boss"] = _output_digest(output, {directory: "<TMP>", prior_sha256: "<PRIOR>"}, log)
        self.assertEqual(digests, self.PINS)



STUDENT_IDENTITY = {"model_id": "spire-learn-combat-ff-standard-v1",
                    "features_id": "spire-learn-combat-features-standard-v1",
                    "action_id": "spire-learn-combat-dispatch-v1", "objective_id": "spire-learn-combat-distill-v1",
                    "config_hashes": {"model": "a" * 64}}
STUDENT_GYM = "731fca0e0341d80e7cf08a8be7a03be1fa32c3f1"


def _student_template(*, objective="spire-learn-combat-distill-v1", device="cuda:0", ppo=False, model_hash="a" * 64,
                      preset="standard-combat-compact-v1") -> str:
    lines = ['schema = "spire-learn-neural-config-v1"', 'mode = "standard-standalone"', 'operation = "train"',
             f'device = "{device}"', "", "[environment]", f'gym_revision = "{STUDENT_GYM}"',
             'standard_view_layout = "compact"', f'standard_view_preset = "{preset}"', "",
             "[model]", "embedding_width = 16", "hidden_width = 64", "max_parameters = 10000000", "value_heads = 3",
             ""]
    if ppo:
        lines += ["[ppo]", "learning_rate = 0.0003", ""]
    lines += ["[[accepted_identities]]", f'model_id = "{STUDENT_IDENTITY["model_id"]}"',
              f'features_id = "{STUDENT_IDENTITY["features_id"]}"', f'action_id = "{STUDENT_IDENTITY["action_id"]}"',
              f'objective_id = "{objective}"', "", "[accepted_identities.config_hashes]", f'model = "{model_hash}"']
    return "\n".join(lines) + "\n"


def _student_plan(directory: Path, **changes) -> Path:
    plan = {"schema": "spire-learn-student-eval-plan-v1", "student_id": "s1-seed1", "gym_revision": STUDENT_GYM,
            "recipe": "configs/neural/curriculum-v1/recipe.json", "template": str(directory / "template.toml"),
            "evaluation": {"action_selection": "deterministic", "case_actions": 1024, "case_seconds": 120},
            "max_turn_actions": 500, "wall_seconds": {"total": 7200},
            "gpu_wait": {"min_free_bytes": 13 * GIB, "poll_seconds": 30, "max_total_seconds": 3600},
            "runner_timeout_seconds": 7200 + 3600 + 30 + 1200, "max_new_output_bytes": 10 * GIB}
    for key, value in changes.items():
        if value is None:
            plan.pop(key)
        else:
            plan[key] = value
    path = directory / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def _student_run(directory: Path, *, status="complete", chosen="checkpoint-best", identity=None) -> Path:
    student = directory / "student"
    (student / "checkpoint-best").mkdir(parents=True)
    (student / "checkpoint-final").mkdir()
    # The recorded paths are from wherever the trainer ran; the entry resolves them in this directory.
    (student / "summary.json").write_text(json.dumps({
        "schema": "spire-learn-distill-train-summary-v1", "status": status,
        "identity": identity or STUDENT_IDENTITY,
        "chosen": {"step": 4000, "tied_steps": [4000], "checkpoint": f"/trained/elsewhere/{chosen}"},
        "final": {"step": 6000, "checkpoint": "/trained/elsewhere/checkpoint-final"}}))
    return student


def _standard_panel_deps(fakes, reports, log, *, cases=4, busy_at=None):
    """Fakes for a Standard panel: real-shaped summaries, a cuda config and a scripted shared GPU."""
    clock = _Clock()
    case_ids = [f"p-{i:02d}" for i in range(cases)]

    def build_panel(recipe, *args, **kwargs):
        log.append(("build_panel", args, kwargs))
        return {"schema": "spire-learn-standard-panel-v2", "panel_sha256": "p" * 64, "manifest_sha256": "m" * 64,
                "manifest": {"schema": "spire-learn-neural-cases-v1", "split": "selection",
                             "partition": "selection",
                             "cases": [{"case_id": cid, "environment_seed": i} for i, cid in enumerate(case_ids)]}}

    def summarize(panel, reports_by_case):
        return {"panel_sha256": panel["panel_sha256"], "panel_manifest_sha256": panel["manifest_sha256"],
                "failures": [], "total": {"operational_failures": 0, "mean_hp_retention": 0.25,
                                          "cases": len(reports_by_case)}}

    def gpu_free():
        log.append(("gpu_free", reports.calls))
        if busy_at is not None and reports.calls == busy_at and not log.count(("busy",)):
            log.append(("busy",))
            return 1 * GIB
        return 20 * GIB

    def sleep(seconds):
        clock.now += seconds

    load_config = fakes.load_config

    def measure_inference(config, checkpoint, gate):
        gate.wait(config, "inference")
        log.append(("measure_inference", str(checkpoint)))
        gate.release(config)
        return {"cpu": {"median_seconds": 0.001, "passes": 1000}, "gpu": {"median_seconds": 0.0005, "passes": 1000}}

    deps = {**fakes.deps(), "evaluate": reports, "build_panel": build_panel, "summarize": summarize,
            "gpu_free_bytes": gpu_free, "sleep": sleep, "release_gpu_cache": lambda: log.append(("release",)),
            "load_config": lambda path: SimpleNamespace(**vars(load_config(path)), device="cuda:0"),
            "measure_inference": measure_inference, "provenance": lambda: {"learner": {"revision": "r"}}}
    return deps, clock, case_ids


class StudentEvalSummaryTests(unittest.TestCase):
    def _run(self, directory, reports_cycle, *, panel="selection", fail_at=None, raise_at=None, busy_at=None,
             raise_error=None, timing_error=None):
        from scripts.neural import student_eval_entry as entry

        directory = Path(directory)
        (directory / "template.toml").write_text(_student_template())
        plan_path = _student_plan(directory)
        student = _student_run(directory)
        fakes = _Fakes()
        reports = _ScriptedReports(fakes, reports_cycle, fail_at=fail_at, raise_at=raise_at)
        if raise_error is not None:
            original = reports.__call__

            def evaluate(config, **kwargs):
                if reports.calls + 1 == raise_error[0]:
                    reports.calls += 1
                    raise raise_error[1]
                return original(config, **kwargs)

            deps, clock, case_ids = _standard_panel_deps(fakes, reports, fakes.log, busy_at=busy_at)
            deps["evaluate"] = evaluate
        else:
            deps, clock, case_ids = _standard_panel_deps(fakes, reports, fakes.log, busy_at=busy_at)
        if timing_error is not None:
            def measure_inference(config, checkpoint, gate):
                raise timing_error
            deps["measure_inference"] = measure_inference
        output = directory / f"out-{panel}"
        summary = entry.run_student_eval(ROOT, plan_path, student, output, panel=panel, dependencies=deps,
                                         clock=clock)
        written = json.loads((output / "summary.json").read_text())
        events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
        return summary, written, events, fakes, case_ids, student, output

    def test_student_summary_format(self):
        from scripts.neural import standard_results, teacher_audit

        with tempfile.TemporaryDirectory() as directory:
            summary, written, events, fakes, case_ids, student, output = self._run(
                directory, ["win", "loss", "censored", "escaped"], busy_at=2)
            self.assertEqual(written["schema"], "spire-learn-student-eval-summary-v1")
            self.assertEqual(written["result"], "complete")
            final = written["evaluations"]["final"]
            # The baselines' per-case evaluation format, under the label the readers use.
            self.assertEqual(set(written["evaluations"]), {"final"})
            self.assertEqual(set(final), {"status", "checkpoint", "cases", "device", "cases_evaluated", "summary"})
            self.assertEqual((final["status"], final["cases_evaluated"], final["device"]), ("complete", 4, "cuda:0"))
            self.assertEqual(final["checkpoint"], str(student / "checkpoint-best"))
            self.assertEqual(final["cases"]["p-02"], {"outcome": "incomplete", "config_sha256": "c" * 64,
                                                      "run_dir": "None", "censor_reason": "turn_action_budget"})
            self.assertEqual(set(final["cases"]["p-00"]), {"outcome", "config_sha256", "run_dir"})
            expected = {"p-00": "win", "p-01": "loss", "p-02": "incomplete", "p-03": "escaped"}
            self.assertEqual(standard_results.load_outcomes(written, "final"), expected)
            panel = {"kind": "selection", "panel_sha256": "p" * 64, "manifest_sha256": "m" * 64,
                     "cases": {cid: {"level": 0 if i < 2 else 4} for i, cid in enumerate(case_ids)}}
            row = teacher_audit.load_ppo(output / "summary.json", "S1-seed1", panel)
            self.assertEqual({key: row[key] for key in ("cases", "wins", "deaths", "escaped", "censored",
                                                        "wins_level0", "mean_hp_kept")},
                             {"cases": 4, "wins": 1, "deaths": 1, "escaped": 1, "censored": 1, "wins_level0": 1,
                              "mean_hp_kept": 0.25})
            # Same case and evaluation keys as a baseline written by the same panel code.
            baseline_reports = _ScriptedReports(_Fakes(), ["win", "loss", "censored", "escaped"])
            baseline_deps, clock, _ = _standard_panel_deps(baseline_reports.fakes, baseline_reports,
                                                           baseline_reports.fakes.log)
            baseline_out = Path(directory) / "baseline"
            baseline_out.mkdir()
            prior, _ = _prior_output(directory)
            baseline = baseline_entry.run_baseline(ROOT, PLAN, baseline_out, dependencies=baseline_deps,
                                                   clock=clock, reevaluate=prior)
            self.assertEqual(set(baseline["evaluations"]["final"]), set(final))
            self.assertEqual({k: set(v) for k, v in baseline["evaluations"]["final"]["cases"].items()},
                             {k: set(v) for k, v in final["cases"].items()})
            panel_written = json.loads((output / "development-panel.json").read_text())
            cases = sorted(p.name for p in (output / "panel").iterdir())
            config = (output / "panel" / "config-0.toml").read_text()
        self.assertEqual(panel_written["panel_sha256"], "p" * 64)
        self.assertEqual(cases, sorted([f"case-{i}.json" for i in range(4)] + [f"config-{i}.toml" for i in range(4)]))
        # FIGHT_CAP and TURN_CAP reach every case config.
        import tomllib
        budgets = tomllib.loads(config)["budgets"]
        self.assertEqual((budgets["max_environment_actions"], budgets["max_turn_actions"]), (1024, 500))
        # Reflex argmax through the checkpoint policy, one runtime per case.
        runtimes = [entry[1] for entry in fakes.log if entry[0] == "create_runtime"]
        self.assertEqual(len(runtimes), 4)
        for kwargs in runtimes:
            self.assertEqual(kwargs, {"purpose": "evaluate", "checkpoint": student / "checkpoint-best",
                                      "policy_kind": "checkpoint", "action_selection": "deterministic"})
        # The GPU gate is checked before every case (and once for inference) and released after each.
        polls = [entry for entry in fakes.log if entry[0] == "gpu_free"]
        self.assertEqual([calls for _, calls in polls], [0, 1, 2, 2, 3, 4])
        self.assertEqual(fakes.log.count(("release",)), 5)
        names = [event["event"] for event in events]
        self.assertEqual(names.count("gpu_wait"), 1)
        self.assertEqual(names.count("gpu_resumed"), 1)
        self.assertEqual(next(e for e in events if e["event"] == "gpu_wait")["stage"], "evaluation final")
        self.assertGreater(written["gpu_wait_seconds"], 0)
        self.assertEqual(written["inference"]["cpu"]["passes"], 1000)
        self.assertEqual(written["student"]["chosen_step"], 4000)
        self.assertEqual(written["student"]["chosen_name"], "checkpoint-best")
        self.assertEqual(set(written["inputs"]), {"plan_sha256", "template_sha256", "recipe_sha256"})
        self.assertEqual(written["provenance"], {"learner": {"revision": "r"}})

    def test_boss_panel_summary(self):
        from scripts.neural import standard_results

        with tempfile.TemporaryDirectory() as directory:
            summary, written, events, fakes, case_ids, student, output = self._run(
                directory, ["loss", "win"], panel="boss")
            self.assertTrue((output / "boss-panel.json").is_file())
        self.assertEqual(written["panel"], "boss")
        self.assertEqual((written["panel_sha256"], written["manifest_sha256"]), ("p" * 64, "m" * 64))
        self.assertEqual(written["boss_case_seconds"], 7200)
        self.assertEqual(standard_results.load_outcomes(written, "final"),
                         {"p-00": "loss", "p-01": "win", "p-02": "loss", "p-03": "win"})
        self.assertIn(("build_panel", ("selection",), {"kind": "boss"}), fakes.log)
        self.assertTrue(all(event.get("panel") == "boss" for event in events))
        self.assertEqual(written["result"], "complete")

    def test_student_fault_is_never_a_loss(self):
        from spire_learn.contracts import ContractError

        with tempfile.TemporaryDirectory() as directory:
            summary, written, events, fakes, *_ = self._run(
                directory, ["win"], raise_error=(3, ContractError("a legal set has no finite logit")))
        final = written["evaluations"]["final"]
        self.assertEqual(final["status"], "fault")
        self.assertEqual(set(final["cases"]), {"p-00", "p-01"})  # the faulted case is not scored
        self.assertEqual(written["evaluation_fault"]["type"], "ContractError")
        self.assertEqual(written["result"], "incomplete")
        self.assertNotIn("inference", written)
        self.assertFalse(any(entry[0] == "measure_inference" for entry in fakes.log))
        # An inference timing fault after a complete panel is recorded; the panel result stays complete.
        with tempfile.TemporaryDirectory() as directory:
            summary, written, *_ = self._run(directory, ["win"], timing_error=RuntimeError("driver refused"))
        self.assertEqual(written["evaluations"]["final"]["status"], "complete")
        self.assertEqual(written["inference_fault"], {"type": "RuntimeError", "message": "driver refused"})
        self.assertNotIn("inference", written)
        self.assertEqual(written["result"], "complete")
        # A failed case report stops the panel as an operational failure.
        with tempfile.TemporaryDirectory() as directory:
            summary, written, *_ = self._run(directory, ["win"], fail_at=2)
        self.assertEqual(written["evaluations"]["final"]["status"], "operational_failure")
        self.assertEqual(written["evaluations"]["final"]["cases"]["p-01"]["outcome"], "failed")
        self.assertEqual(written["result"], "incomplete")

    def test_student_plan_template_and_run_are_checked(self):
        from scripts.neural import student_eval_entry as entry

        plan = {"evaluation": {"action_selection": "seeded-sampling", "case_actions": 1024, "case_seconds": 120}}
        refusals = {
            "sampling": plan,
            "fight cap": {"evaluation": {"action_selection": "deterministic", "case_actions": 256,
                                         "case_seconds": 120}},
            "turn cap": {"max_turn_actions": 64},
            "no turn cap": {"max_turn_actions": None},
            "short timeout": {"runner_timeout_seconds": 7200},
            "unknown key": {"retain_near_decisions": [1]},
        }
        for name, changes in refusals.items():
            with self.subTest(plan=name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    entry.load_plan(_student_plan(Path(directory), **changes))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            loaded = entry.load_plan(_student_plan(directory))
            for name, text in {"PPO objective": _student_template(objective="spire-learn-combat-winloss-v1"),
                               "[ppo]": _student_template(ppo=True),
                               "play preset": _student_template(preset="standard-combat-compact-play-v1")}.items():
                with self.subTest(template=name), self.assertRaises(ValueError):
                    (directory / "t.toml").write_text(text)
                    entry.check_template(directory / "t.toml", loaded)
            (directory / "t.toml").write_text(_student_template())
            with self.assertRaisesRegex(ValueError, "gpu_wait"):
                entry.check_template(directory / "t.toml", {**loaded, "gpu_wait": None})
            identity = entry.check_template(directory / "t.toml", loaded)
            (directory / "t.toml").write_text(_student_template(device="cpu"))
            entry.check_template(directory / "t.toml", {**loaded, "gpu_wait": None})
            for name, kwargs in {"failed run": {"status": "failed"},
                                 "other model": {"identity": {**STUDENT_IDENTITY, "config_hashes": {"model": "b" * 64}}},
                                 "PPO run": {"identity": {**STUDENT_IDENTITY,
                                                          "objective_id": "spire-learn-combat-winloss-v1"}},
                                 "unknown checkpoint": {"chosen": "checkpoint-other"}}.items():
                with self.subTest(student=name), tempfile.TemporaryDirectory() as folder:
                    with self.assertRaises(ValueError):
                        entry.load_student(_student_run(Path(folder), **kwargs), identity)
            with tempfile.TemporaryDirectory() as folder:
                student = _student_run(Path(folder))
                record = entry.load_student(student, identity)
                self.assertEqual(record["checkpoint"], str(student / "checkpoint-best"))
                (student / "checkpoint-best").rmdir()
                with self.assertRaisesRegex(ValueError, "missing"):
                    entry.load_student(student, identity)


    def test_student_case_configs_load_as_distillation_evaluation(self):
        # A v1.2.5 distillation evaluation template, sliced per case by panel_eval, loads as a
        # student config: FIGHT_CAP, TURN_CAP, the distill identity and a registered FF family.
        from scripts.neural import student_eval_entry as entry
        from spire_learn.neural import panel_eval
        from spire_learn.neural.config import load_neural_config
        from spire_learn.neural.model_families import check_family_config
        from tests.unit.test_distill_config import eval_toml

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            template = folder / "template.toml"
            template.write_text(eval_toml(folder))
            plan = entry.load_plan(_student_plan(folder, template=str(template)))
            identity = entry.check_template(template, plan)
            self.assertEqual(identity["objective_id"], "spire-learn-combat-distill-v1")
            manifest = json.loads((folder / "selection.json").read_text())
            rows = panel_eval.panel_configs({"manifest": manifest}, template, folder / "panel", plan)
            # Real outputs live under the artifacts root; this temp folder does not, so only the
            # case config's relative output_root is redirected there (nothing is written to it).
            case_config = Path(rows[0]["config"])
            text = case_config.read_text()
            self.assertIn('output_root = "runs"', text)
            case_config.write_text(text.replace('output_root = "runs"', 'output_root = "{}"'.format(
                ROOT / "artifacts/development/neural-combat/runs/test-student-eval")))
            config = load_neural_config(case_config)
        self.assertEqual((config.operation, config.split, config.rollout_rows, config.rollout_steps),
                         ("evaluate", "selection", 1, 1))
        self.assertEqual((config.budgets.max_environment_actions, config.budgets.max_turn_actions), (1024, 500))
        self.assertIsNone(config.ppo)
        self.assertEqual(config.model.value_heads, 3)
        selected = config.accepted_identities[0]
        self.assertEqual(selected.objective_id, "spire-learn-combat-distill-v1")
        self.assertEqual(check_family_config(selected.model_id, config.model).family_id, "FF")


if __name__ == "__main__":
    unittest.main()
