from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

from spire_learn.contracts import CaseExecution
from spire_learn.experiments.cases import CaseManifest, load_cases
from spire_learn.experiments.config import (
    bind_case_expectations,
    load_config,
    to_probe_config,
    validate_expectations,
)


def config_text(**overrides):
    fields = {
        "mode": '"evaluate"', "profile": '"smoke"', "agent": '"random"',
        "cases": '"cases.json"', "output_root": '"runs"',
        "max_wall_seconds": "10", "traces": "false", "per_case_decisions": "5",
    }
    fields.update(overrides)
    return "\n".join(f"{key} = {value}" for key, value in fields.items())


class ConfigTests(unittest.TestCase):
    def load(self, text, *, cwd=None, expected_mode="evaluate"):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "settings.toml"
            path.write_text(text)
            original = Path.cwd()
            try:
                if cwd:
                    os.chdir(cwd)
                return load_config(path, expected_mode=expected_mode)
            finally:
                os.chdir(original)

    def test_loads_evaluate_and_resolves_paths_from_config(self):
        cfg = self.load(config_text())
        self.assertEqual(cfg.cases.name, "cases.json")
        self.assertEqual(cfg.cases.parent, cfg.output_root.parent)
        self.assertEqual(cfg.max_wall_seconds, 10.0)
        self.assertEqual(cfg.per_case_decisions, 5)

    def test_authored_evaluation_profiles_use_exact_mode_specific_caps(self):
        expected = (
            ("smoke", 128, 60.0),
            ("development", 1024, 600.0),
            ("overnight", 4096, 28800.0),
        )
        for profile, decisions, wall_seconds in expected:
            with self.subTest(profile=profile):
                config = load_config(
                    Path("configs/evaluation") / f"{profile}.toml",
                    expected_mode="evaluate",
                )
                self.assertEqual(
                    (config.mode, config.profile, config.agent,
                     config.per_case_decisions, config.max_wall_seconds),
                    ("evaluate", profile, "random", decisions, wall_seconds),
                )
                self.assertEqual(
                    config.cases,
                    Path("configs/cases/selection-smoke.json").resolve(),
                )

    def test_authored_probe_settings_derive_executions_from_selection_manifest(self):
        manifest_path = Path("configs/cases/selection-smoke.json").resolve()
        manifest = load_cases(manifest_path)
        for mode in ("probe-vector", "probe-torchrl"):
            with self.subTest(mode=mode):
                config = load_config(
                    Path("configs/probes") / f"{mode.removeprefix('probe-')}-smoke.toml",
                    expected_mode=mode,
                )
                probe = to_probe_config(config, manifest)
                self.assertIsNone(config.per_case_decisions)
                self.assertEqual(
                    (config.mode, config.profile, config.agent,
                     config.max_batch_steps, config.max_wall_seconds,
                     config.probe.num_envs, config.probe.workers),
                    (mode, "smoke", "random", 128, 60.0, 2, 1),
                )
                self.assertEqual(probe.executions, manifest.executions)
                self.assertEqual(
                    tuple(
                        (item.scenario_id, item.environment_seed, item.policy_seed)
                        for item in probe.executions
                    ),
                    (
                        ("starter-cultist-a0-v1", 7, 101),
                        ("starter-jaw-worm-a0-v1", 11, 103),
                    ),
                )

    def test_relative_paths_use_config_parent_not_current_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as cwd:
            path = Path(directory) / "nested" / "settings.toml"
            path.parent.mkdir()
            path.write_text(config_text())
            original = Path.cwd()
            try:
                os.chdir(cwd)
                cfg = load_config(path, expected_mode="evaluate")
            finally:
                os.chdir(original)
            self.assertEqual(cfg.cases, (path.parent / "cases.json").resolve())
            self.assertNotEqual(cfg.cases.parent, Path(cwd).resolve())

    def test_explicit_absolute_optional_paths_are_preserved_as_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory).resolve()
            text = config_text(gym_checkout=f'"{absolute}"', expected_schema_id='"schema-v1"')
            cfg = self.load(text)
            self.assertEqual(cfg.gym_checkout, absolute)

    def test_rejects_unknown_missing_wrong_mode_and_bad_profile_agent(self):
        for text in (config_text(extra="1"), config_text().replace('traces = false\n', ""),
                     config_text(mode='"probe-vector"'), config_text(profile='"other"'),
                     config_text(agent='"other"')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_numeric_boolean_nonfinite_zero_and_negative_values_rejected(self):
        for value in ("true", '"5"', "0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(config_text(max_wall_seconds=value))
        for value in ("true", '"5"', "0", "-1"):
            with self.subTest(decisions=value), self.assertRaises(ValueError):
                self.load(config_text(per_case_decisions=value))

    def test_evaluate_forbids_probe_and_batch_cap(self):
        for suffix in ("max_batch_steps = 1", "[probe]\nnum_envs=2\nworkers=1"):
            with self.assertRaises(ValueError):
                self.load(config_text() + "\n" + suffix)

    def test_probe_mode_matrix_and_config_fields(self):
        common = config_text(mode='"probe-vector"', max_wall_seconds="2.5", traces="true")
        probe = common.replace("per_case_decisions = 5", "") + "\nmax_batch_steps=3\n[probe]\nnum_envs=2\nworkers=1\n"
        cfg = self.load(probe, expected_mode="probe-vector")
        self.assertEqual((cfg.probe.num_envs, cfg.probe.workers, cfg.max_batch_steps), (2, 1, 3))
        for suffix in ("", "\nper_case_decisions=1", "\nmax_batch_steps=true", "\n[probe]\nnum_envs=1\nworkers=1", "\n[probe]\nnum_envs=2\nworkers=0", "\n[probe]\nnum_envs=2\nworkers=1\nextra=1"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.load(common.replace("per_case_decisions = 5", "") + "\nmax_batch_steps=3\n" + suffix, expected_mode="probe-vector")

    def test_probe_modes_reject_non_random_agents_during_strict_load(self):
        for mode in ("probe-vector", "probe-torchrl"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "probe.toml"
                path.write_text(
                    config_text(mode=f'"{mode}"', agent='"play-first"')
                    .replace("per_case_decisions = 5", "")
                    + "\nmax_batch_steps = 3\n[probe]\nnum_envs = 2\nworkers = 1\n"
                )
                with self.assertRaisesRegex(ValueError, "probe.*agent|agent.*random"):
                    load_config(path, expected_mode=mode)

    def test_evaluation_still_accepts_play_first_agent(self):
        config = self.load(config_text(agent='"play-first"'))
        self.assertEqual((config.mode, config.agent), ("evaluate", "play-first"))

    def test_probe_conversion_rejects_direct_non_random_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.toml"
            path.write_text(
                config_text(mode='"probe-vector"').replace("per_case_decisions = 5", "")
                + "\nmax_batch_steps = 4\n[probe]\nnum_envs = 2\nworkers = 1\n"
            )
            config = replace(load_config(path, expected_mode="probe-vector"), agent="play-first")
        executions = (
            self.execution(execution_id="probe-a"),
            self.execution(execution_id="probe-b", environment_seed=24),
        )
        manifest = CaseManifest(Path("cases.json"), "selection", executions)

        with self.assertRaisesRegex(ValueError, "probe.*agent|agent.*random"):
            to_probe_config(config, manifest)

    def test_trace_requires_exact_boolean_and_optional_split_guards_resolve(self):
        with self.assertRaises(ValueError):
            self.load(config_text(traces="1"))
        cfg = self.load(config_text(split_guard_manifests='["guards/a.json"]'))
        self.assertEqual(cfg.split_guard_manifests[0].name, "a.json")

    def execution(self, **changes):
        values = dict(execution_id="exec", case_id="case", split="selection", partition="native",
                      scenario_id="scenario", environment_seed=23, policy_seed=41)
        values.update(changes)
        return CaseExecution(**values)

    def test_binding_fills_missing_expected_schema_and_fingerprint_preserving_seeds(self):
        cfg = self.load(config_text(expected_schema_id='"schema-v1"', expected_fingerprint='"' + 'a'*64 + '"'))
        original = self.execution()
        bound = bind_case_expectations(cfg, original)
        self.assertEqual((bound.expected_schema_id, bound.expected_fingerprint), ("schema-v1", "a" * 64))
        self.assertEqual((bound.execution_id, bound.environment_seed, bound.policy_seed), ("exec", 23, 41))
        self.assertIsNone(original.expected_schema_id)

    def test_binding_preserves_case_claims_and_rejects_each_conflict(self):
        cfg = self.load(config_text(expected_schema_id='"schema-v1"', expected_partition='"native"'))
        existing = self.execution(expected_schema_id="schema-v1", expected_fingerprint="b" * 64)
        self.assertEqual(bind_case_expectations(cfg, existing).expected_fingerprint, "b" * 64)
        for execution in (self.execution(expected_schema_id="other"),
                          self.execution(partition="other")):
            with self.assertRaises(ValueError):
                bind_case_expectations(cfg, execution)

    def test_validate_expectations_checks_manifest_static_assertions(self):
        cfg = self.load(config_text(expected_partition='"native"', expected_schema_id='"schema-v1"'))
        manifest = CaseManifest(Path("cases.json"), "selection", (self.execution(),))
        validate_expectations(cfg, manifest)
        conflict = CaseManifest(Path("cases.json"), "selection", (self.execution(partition="other"),))
        with self.assertRaises(ValueError):
            validate_expectations(cfg, conflict)

    def test_validate_expectations_rejects_execution_split_mismatch(self):
        cfg = self.load(config_text())
        manifest = CaseManifest(Path('cases.json'), 'selection',
                                (self.execution(split='final'),))
        with self.assertRaisesRegex(ValueError, 'split'):
            validate_expectations(cfg, manifest)

    def test_probe_conversion_requires_selection_exact_count_and_single_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.toml"
            path.write_text(config_text(mode='"probe-vector"').replace("per_case_decisions = 5", "") + "\nmax_batch_steps=4\n[probe]\nnum_envs=2\nworkers=1\n")
            cfg = load_config(path, expected_mode="probe-vector")
        executions = (self.execution(execution_id="a"), self.execution(execution_id="b", environment_seed=24))
        manifest = CaseManifest(Path("cases.json"), "selection", executions)
        converted = to_probe_config(cfg, manifest)
        self.assertEqual(converted.partition, "native")
        self.assertEqual(tuple((e.environment_seed, e.policy_seed, e.scenario_id) for e in converted.executions),
                         tuple((e.environment_seed, e.policy_seed, e.scenario_id) for e in executions))
        for bad in (CaseManifest(Path("x"), "train", executions), CaseManifest(Path("x"), "selection", executions[:1]),
                    CaseManifest(Path("x"), "selection", (executions[0], self.execution(execution_id="b", partition="other")))):
            with self.assertRaises(ValueError):
                to_probe_config(cfg, bad)

    def test_probe_conversion_rejects_execution_split_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.toml'
            path.write_text(config_text(mode='"probe-vector"').replace(
                'per_case_decisions = 5', '') +
                '\nmax_batch_steps=4\n[probe]\nnum_envs=2\nworkers=1\n')
            cfg = load_config(path, expected_mode='probe-vector')
        executions = (self.execution(execution_id='a', split='final'),
                      self.execution(execution_id='b', split='final', environment_seed=24))
        manifest = CaseManifest(Path('cases.json'), 'selection', executions)
        with self.assertRaisesRegex(ValueError, 'split'):
            to_probe_config(cfg, manifest)

    def test_import_does_not_load_runtime_probe_or_native_modules(self):
        import sys
        import spire_learn.experiments.config
        self.assertFalse(any(name in sys.modules for name in ("numpy", "torch", "torchrl", "gymnasium", "spire_lab", "spire_learn.probes.vector", "spire_learn.probes.torchrl")))


if __name__ == "__main__":
    unittest.main()
