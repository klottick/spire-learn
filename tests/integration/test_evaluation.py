"""Single bounded opt-in smoke through the fixed-case evaluation runner."""

import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import tempfile
import unittest

_RUN_NATIVE = os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1"


@unittest.skipUnless(
    _RUN_NATIVE,
    "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 after explicit runtime approval",
)
class EvaluationIntegrationTests(unittest.TestCase):
    def test_two_bounded_selection_executions_reconcile_artifacts(self):
        for distribution in ("spire-lab", "gymnasium", "numpy"):
            try:
                version(distribution)
            except PackageNotFoundError as error:
                self.fail(f"native integration dependency is missing: {distribution}: {error}")

        from spire_learn.envs.native import make_single
        from spire_learn.evaluation.runner import evaluate
        from spire_learn.experiments.config import load_config

        output_override = os.environ.get("SPIRE_LEARN_EVALUATION_OUTPUT_ROOT")
        with tempfile.TemporaryDirectory() as temporary:
            owned_root = Path(temporary)
            output_root = (
                Path(output_override).expanduser().resolve()
                if output_override
                else owned_root / "runs"
            )
            cases_path = owned_root / "cases.json"
            config_path = owned_root / "evaluation.toml"
            schema_id = "spire-lab-combat-observation-v1"
            fingerprint = "8fd66d212259d2bdd27139988a772e1a6c223f8d65524f38c74174a0d0843357"
            scenarios = (
                ("starter-cultist-a0-v1", 7, 101),
                ("starter-jaw-worm-a0-v1", 11, 103),
            )
            expected_execution_records = {
                "evaluation-1": {
                    "execution_id": "evaluation-1",
                    "case_id": "starter-cultist-a0-v1",
                    "split": "selection",
                    "partition": "selection",
                    "scenario_id": "starter-cultist-a0-v1",
                    "environment_seed": 7,
                    "policy_seed": 101,
                    "expected_schema_id": schema_id,
                    "expected_fingerprint": fingerprint,
                },
                "evaluation-2": {
                    "execution_id": "evaluation-2",
                    "case_id": "starter-jaw-worm-a0-v1",
                    "split": "selection",
                    "partition": "selection",
                    "scenario_id": "starter-jaw-worm-a0-v1",
                    "environment_seed": 11,
                    "policy_seed": 103,
                    "expected_schema_id": schema_id,
                    "expected_fingerprint": fingerprint,
                },
            }
            cases = [
                {
                    "case_id": scenario_id,
                    "scenario_id": scenario_id,
                    "partition": "selection",
                    "expected_schema_id": schema_id,
                    "expected_fingerprint": fingerprint,
                    "executions": [
                        {
                            "execution_id": f"evaluation-{index}",
                            "environment_seed": environment_seed,
                            "policy_seed": policy_seed,
                        }
                    ],
                }
                for index, (scenario_id, environment_seed, policy_seed) in enumerate(scenarios, 1)
            ]
            cases_path.write_text(
                json.dumps({"schema": "spire-learn-cases-v1", "split": "selection", "cases": cases}),
                encoding="utf-8",
            )
            config_path.write_text(
                "\n".join(
                    (
                        'mode = "evaluate"',
                        'profile = "smoke"',
                        'agent = "random"',
                        'cases = "cases.json"',
                        f"output_root = {json.dumps(str(output_root))}",
                        "per_case_decisions = 128",
                        "max_wall_seconds = 60.0",
                        "traces = false",
                        'expected_partition = "selection"',
                        f'expected_schema_id = "{schema_id}"',
                        f'expected_fingerprint = "{fingerprint}"',
                        "",
                    )
                ),
                encoding="utf-8",
            )
            config = load_config(config_path, expected_mode="evaluate")
            counts = {"factory": 0, "reset": 0, "step": 0, "close": 0}
            per_case_steps = []
            reset_arguments = []

            class CountingEnv:
                def __init__(self, execution, env):
                    self._execution = execution
                    self._env = env
                    self._steps = 0

                def observation_spec(self):
                    return self._env.observation_spec()

                def reset(self, *, seed=None, options=None):
                    counts["reset"] += 1
                    reset_arguments.append((self._execution.execution_id, seed, options))
                    return self._env.reset(seed=seed, options=options)

                def step(self, action):
                    counts["step"] += 1
                    self._steps += 1
                    return self._env.step(action)

                def close(self):
                    counts["close"] += 1
                    per_case_steps.append(self._steps)
                    return self._env.close()

            def counted_factory(execution):
                counts["factory"] += 1
                return CountingEnv(execution, make_single(execution))

            report = evaluate(config, env_factory=counted_factory)

            self.assertEqual(report.exit_code, 0)
            self.assertIsNone(report.fatal_error)
            self.assertEqual(counts["factory"], 2)
            self.assertEqual(counts["reset"], 2)
            self.assertEqual(
                reset_arguments,
                [
                    ("evaluation-1", 7, {"scenario_id": "starter-cultist-a0-v1"}),
                    ("evaluation-2", 11, {"scenario_id": "starter-jaw-worm-a0-v1"}),
                ],
            )
            self.assertEqual(counts["close"], 2)
            self.assertEqual(len(per_case_steps), 2)
            self.assertTrue(all(0 < steps <= 128 for steps in per_case_steps))
            self.assertLessEqual(counts["step"], 256)
            self.assertEqual(report.summary["scheduled"], 2)
            self.assertEqual(report.summary["started"], 2)
            self.assertEqual(report.summary["not_started"], 0)
            self.assertEqual(report.summary["terminal_completion_coverage"], 1.0)
            self.assertIsNotNone(report.run_dir)

            episode_records = [
                json.loads(line)
                for line in (report.run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            case_results = [record["record"] for record in episode_records]
            self.assertEqual(len(case_results), 2)
            actual_execution_records = {
                item["execution"]["execution_id"]: item["execution"]
                for item in case_results
            }
            self.assertEqual(len(actual_execution_records), len(case_results))
            self.assertEqual(actual_execution_records, expected_execution_records)
            self.assertTrue(all(item["status"] == "terminal" for item in case_results))
            self.assertTrue(all(item["outcome"] in {"victory", "death"} for item in case_results))
            self.assertTrue(all(item["primary_failure"] is None for item in case_results))
            self.assertTrue(all(item["secondary_errors"] == [] for item in case_results))
            self.assertEqual(counts["step"], sum(item["steps"] for item in case_results))

            config_artifact = json.loads(
                (report.run_dir / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config_artifact["schema_id"], "spire-learn-config-v1")
            resolved_config = config_artifact["config"]
            self.assertEqual(
                {
                    key: resolved_config[key]
                    for key in (
                        "mode",
                        "profile",
                        "agent",
                        "cases",
                        "output_root",
                        "per_case_decisions",
                        "max_wall_seconds",
                        "traces",
                        "expected_partition",
                        "expected_schema_id",
                        "expected_fingerprint",
                        "allow_final",
                    )
                },
                {
                    "mode": "evaluate",
                    "profile": "smoke",
                    "agent": "random",
                    "cases": str(cases_path.resolve()),
                    "output_root": str(output_root.resolve()),
                    "per_case_decisions": 128,
                    "max_wall_seconds": 60.0,
                    "traces": False,
                    "expected_partition": "selection",
                    "expected_schema_id": schema_id,
                    "expected_fingerprint": fingerprint,
                    "allow_final": False,
                },
            )
            manifest_artifact = json.loads(
                (report.run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_artifact["schema_id"], "spire-learn-cases-v1")
            resolved_manifest = manifest_artifact["manifest"]
            self.assertEqual(resolved_manifest["path"], str(cases_path.resolve()))
            self.assertEqual(resolved_manifest["split"], "selection")
            manifest_execution_records = {
                item["execution_id"]: item for item in resolved_manifest["executions"]
            }
            self.assertEqual(len(manifest_execution_records), len(resolved_manifest["executions"]))
            self.assertEqual(manifest_execution_records, expected_execution_records)

            recorded_summary = json.loads(
                (report.run_dir / "summary.json").read_text(encoding="utf-8")
            )["summary"]
            self.assertEqual(recorded_summary, dict(report.summary))
            recorded_outcomes = {
                outcome: sum(item["outcome"] == outcome for item in case_results)
                for outcome in ("victory", "death")
            }
            self.assertEqual(recorded_summary["outcome_counts"], recorded_outcomes)
            discoveries = [
                json.loads(line)
                for line in (report.run_dir / "manifest_updates.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(discoveries), 2)
            discovery_records = [record["record"] for record in discoveries]
            expected_build_info = {
                "extension": "spire-python",
                "revision": "4874fb398128c142bd56f10a12e6fca939a88f93",
                "rust_version": "1.98.0",
            }
            self.assertEqual(
                {record["execution_id"] for record in discovery_records},
                {item["execution"]["execution_id"] for item in case_results},
            )
            for item in case_results:
                self.assertEqual(item["observed_descriptor"]["schema_id"], schema_id)
                self.assertEqual(item["observed_descriptor"]["fingerprint"], fingerprint)
                self.assertEqual(item["native_build_info"], expected_build_info)
            for record in discovery_records:
                self.assertEqual(record["observed_descriptor"]["schema_id"], schema_id)
                self.assertEqual(record["observed_descriptor"]["fingerprint"], fingerprint)
                self.assertEqual(record["native_build_info"], expected_build_info)

            evidence = {
                "run_dir": str(report.run_dir),
                "counts": {**counts, "actions_total": counts["step"]},
                "per_case_steps": per_case_steps,
                "summary": dict(report.summary),
                "native_build_info": expected_build_info,
                "schema_ids": sorted({item["observed_descriptor"]["schema_id"] for item in case_results}),
            }
            print("EVALUATION_EVIDENCE " + json.dumps(evidence, sort_keys=True))
