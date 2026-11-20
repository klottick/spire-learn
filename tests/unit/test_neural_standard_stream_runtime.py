"""The curriculum stream in standard-standalone mode: recipe dispatch, config, sampler, resume, panels.

Host tests use synthetic run-state tables and fake Gym validators. The gated smoke test
(``SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1``) trains two updates at 16 rows against the real
Gym in the runtime container, via ``scripts/neural/standard_stream_smoke_entry.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spire_learn.neural import recipes
from spire_learn.neural import standard_stream as ss
from spire_learn.neural.cases import NeuralCases, load_neural_cases
from spire_learn.neural.config import load_neural_config
from spire_learn.neural.curriculum import CurriculumSampler
from spire_learn.neural.scenario_stream import ScenarioRecipe, generate_scenario_shard, load_scenario_recipe
from spire_learn.neural.scenario_support import canonical_bytes, content_sha256, json_value
from spire_learn.neural.standard_catalog import CatalogRejected
from tests.unit.test_neural_standard_stream import recipe_document, synthetic_states, write_table

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/development/neural-combat"
# Inside the runtime container only the runner output is writable; it lies under ARTIFACTS.
WORK = Path(os.environ.get("SPIRE_LEARN_RUNNER_OUTPUT") or ARTIFACTS)
V1 = ROOT / "configs/neural/curriculum-v1/recipe.json"
STANDARD_TEMPLATE = ROOT / "configs/neural/gym-1.1.1/standard-standalone.toml"
STANDARD_STREAM_TEMPLATE = ROOT / "configs/neural/gym-1.1.1/standard-stream.toml"
_RUN_NATIVE = os.environ.get("SPIRE_LEARN_RUN_NATIVE_INTEGRATION") == "1"

# The unmodified M10 pins (tests/unit/test_neural_scenario_stream_v2.py).
V1_RECIPE_SHA256 = "3279664729328894fc995c997ccc54282f8d33483459e91d4da1b7ed43a92f66"
V1_TRAIN_SHARD_0 = "d644d2a113034641bc12cd06e50c7817cad23b5268c3c9014ef3f3f4406c7909"
V1_TRAIN_SHARD_37 = "8df900ac9a7ec8375c39628675ee799d8def93d445f0fe70bdc0cdcdf20af7a7"
V1_PANEL_MANIFEST = "435a236e60ac54f0b04f742b6d019023f49670c7730ba47545b73b23d2fb620e"


class FakeGym:
    """A deterministic fake catalog validator: rejects any entry whose id hashes to 0 mod 4."""

    def __init__(self):
        self.calls = 0

    def __call__(self, catalog):
        self.calls += 1
        for entry in catalog["entries"]:
            if int(hashlib.sha256(entry["id"].encode()).hexdigest(), 16) % 4 == 0:
                raise CatalogRejected(f"fake rejection of {entry['id']}")


def bind(test: unittest.TestCase, validate) -> None:
    previous = recipes.bind_validator(validate)
    test.addCleanup(recipes.bind_validator, previous)
    ss.clear_shard_cache()
    test.addCleanup(ss.clear_shard_cache)


def standard_template_text(*, rows=4, steps=32, common=False, identity="standard") -> str:
    """The shipped Standard template with a Standard-only identity (and no common view)."""
    text = STANDARD_TEMPLATE.read_text()
    if not common:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("common_view_")) + "\n"
    if identity == "standard":
        text = text.replace('model_id = "spire-learn-combat-ff-v1"', 'model_id = "spire-learn-combat-ff-standard-v1"')
        text = text.replace('features_id = "spire-learn-combat-features-v1"',
                            'features_id = "spire-learn-combat-features-standard-v1"')
    return text.replace("rows = 4\nsteps = 32", f"rows = {rows}\nsteps = {steps}")


class _StandardFixture(unittest.TestCase):
    """A synthetic table (400 runs) and a schedule that turns fuzz on at shard 2."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(dir=WORK)
        cls.dir = Path(cls._tmp.name)
        _, cls.table_sha = write_table(cls.dir)
        cls.recipe_path = cls.dir / "recipe.json"
        cls.recipe_path.write_text(json.dumps(recipe_document(cls.table_sha)))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.gym = FakeGym()
        bind(self, self.gym)
        self.recipe = recipes.load_recipe(self.recipe_path)


class RecipeDispatchTests(_StandardFixture):
    def test_load_recipe_dispatch(self):
        m10 = recipes.load_recipe(V1)
        self.assertIsInstance(m10, ScenarioRecipe)
        self.assertEqual(m10.sha256, load_scenario_recipe(V1).sha256)
        self.assertIsInstance(self.recipe, ss.StandardRecipe)
        self.assertEqual(self.recipe.sha256, ss.load_standard_recipe(self.recipe_path).sha256)
        # Shards: M10 delegates unchanged; Standard binds the process validator only when fuzzed.
        self.assertEqual(recipes.generate_shard(m10, split="train", shard_index=3, arm="mixed").manifest_sha256,
                         generate_scenario_shard(m10, split="train", shard_index=3, arm="mixed").manifest_sha256)
        plain = recipes.generate_shard(self.recipe, split="train", shard_index=0, arm="mixed")
        self.assertEqual(self.gym.calls, 0)
        self.assertEqual(plain.manifest_sha256,
                         ss.generate_standard_shard(self.recipe, split="train", shard_index=0).manifest_sha256)
        fuzzed = recipes.generate_shard(self.recipe, split="train", shard_index=2, arm="mixed")
        self.assertGreater(self.gym.calls, 0)
        self.assertEqual(fuzzed.manifest_sha256, ss.generate_standard_shard(
            self.recipe, split="train", shard_index=2, validate=self.gym).manifest_sha256)
        # Panels: M10 is the development panel; Standard is the selection panel by default.
        panel = recipes.build_panel(self.recipe)
        self.assertEqual(panel["schema"], "spire-learn-standard-panel-v1")
        self.assertEqual(panel["split"], "selection")
        self.assertEqual(json_value(panel), json_value(ss.build_standard_panel(self.recipe, "selection",
                                                                              validate=self.gym)))
        self.assertEqual(recipes.build_panel(self.recipe, "final")["split"], "final")
        with self.assertRaises(ValueError):
            recipes.build_panel(m10, "final")
        # A recipe with an unknown generator fails at load.
        bad = self.dir / "unknown.json"
        bad.write_text(json.dumps({**recipe_document(self.table_sha), "generator_version": "nope"}))
        with self.assertRaises(ValueError):
            recipes.load_recipe(bad)

    def test_m10_paths_unchanged(self):
        recipe = recipes.load_recipe(V1)
        self.assertEqual(recipe.sha256, V1_RECIPE_SHA256)
        self.assertEqual(recipes.generate_shard(recipe, split="train", shard_index=0, arm="mixed").manifest_sha256,
                         V1_TRAIN_SHARD_0)
        self.assertEqual(recipes.generate_shard(recipe, split="train", shard_index=37, arm="mixed").manifest_sha256,
                         V1_TRAIN_SHARD_37)
        self.assertEqual(recipes.build_panel(recipe)["manifest_sha256"], V1_PANEL_MANIFEST)
        # The M10 sampler keeps its schema and state.
        shard = recipes.generate_shard(recipe, split="train", shard_index=0, arm="mixed")
        sampler = CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                                    categories=recipe.category_keys)
        self.assertEqual(sampler.snapshot()["schema"], "spire-learn-curriculum-v1")
        # The existing M10 stream pins and resume tests run unchanged.
        loader = unittest.defaultTestLoader
        suite = unittest.TestSuite([
            loader.loadTestsFromName("tests.unit.test_neural_scenario_stream_v2.ScenarioStreamV2Tests"),
            loader.loadTestsFromName("tests.unit.test_neural_stream_resume.StreamResumeTests"
                                     ".test_loader_binds_actual_manifest_recipe_and_preserves_real_hashes"),
        ])
        result = unittest.TestResult()
        cwd = os.getcwd()
        os.chdir(ROOT)  # those tests name the repository's configs by relative path
        try:
            suite.run(result)
        finally:
            os.chdir(cwd)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)


class StandardConfigTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            path = Path(folder) / "config.toml"
            path.write_text(text.replace('manifest_path = "../../cases/neural/standard-standalone-train.json"',
                                         f'manifest_path = "{ROOT / "configs/cases/neural/standard-standalone-train.json"}"')
                            .replace('output_root = "../../../artifacts/development/neural-combat/runs/gym-1.1.1/standard-standalone"',
                                     f'output_root = "{ARTIFACTS / "runs/standard-config-test"}"'))
            return load_neural_config(path)

    def test_standard_config_validates_without_common_fingerprint(self):
        config = self._load(standard_template_text())
        self.assertIsNone(config.common_view_id)
        self.assertIsNone(config.common_view_fingerprint)
        self.assertEqual(config.standard_view_fingerprint,
                         "5d8147e13ded030c7faa501a8202eb2588915c2e1330b30cac5b06afc9f9916e")
        identity = config.accepted_identities[0]
        self.assertEqual((identity.model_id, identity.features_id),
                         ("spire-learn-combat-ff-standard-v1", "spire-learn-combat-features-standard-v1"))
        # The shipped Standard templates (common view present, v1 identities) still load.
        self.assertEqual(self._load(STANDARD_TEMPLATE.read_text()).common_view_fingerprint,
                         "4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463")
        self.assertIsNone(self._load(standard_template_text(identity="v1")).common_view_fingerprint)
        # The Standard identity cannot claim an M10 mode, and an M10 config still needs the common view.
        claims_m10 = standard_template_text(common=True).replace(
            "[accepted_identities.accepted_raw_mode_mappings]\n",
            "[accepted_identities.accepted_raw_mode_mappings]\nstandalone = "
            '"spire-learn-standalone-common-v1@4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463+gym-1.1.1"\n')
        with self.assertRaises(ValueError):
            self._load(claims_m10)
        with self.assertRaises(ValueError):
            self._load(standard_template_text().replace(
                'standard_view_fingerprint', 'common_view_id = "spire-gym-combat-policy-view-v1"\nstandard_view_fingerprint'))
        m10 = (ROOT / "configs/neural/gym-1.1.1/standalone.toml").read_text()
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            path = Path(folder) / "standalone.toml"
            path.write_text("\n".join(line for line in m10.splitlines() if not line.startswith("common_view_")))
            with self.assertRaises(ValueError):
                load_neural_config(path)
        # The shipped Standard stream template loads with the Standard identity and no common view.
        shipped = load_neural_config(STANDARD_STREAM_TEMPLATE)
        self.assertEqual(shipped.accepted_identities[0].model_id, "spire-learn-combat-ff-standard-v1")
        self.assertIsNone(shipped.common_view_fingerprint)


class StandardSamplerTests(_StandardFixture):
    def test_sampler_accepts_standard_annotations_and_round_trips(self):
        shard = recipes.generate_shard(self.recipe, split="train", shard_index=2, arm="mixed")
        cases = load_cases(shard)
        sampler = CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                                    categories=self.recipe.category_keys)
        sampler.validate_cases(cases)
        rng = random.Random(5)
        chosen = [sampler.choose_case(cases, rng=rng) for _ in range(200)]
        self.assertGreater(len({case.case_id for case in chosen}), 20)
        for case in chosen[:50]:
            sampler.observe(case.case_id, ("victory", "death", "escaped")[len(case.case_id) % 3])
        state = sampler.snapshot()
        self.assertEqual(state["schema"], "spire-learn-curriculum-standard-v1")
        self.assertEqual(state["allocated"], 200)
        self.assertEqual(sum(state["category_stats"][key]["allocated"] for key in state["category_stats"]
                             if key.startswith("level:")), 200)
        restored = CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                                     categories=self.recipe.category_keys, state=state)
        self.assertEqual(restored.snapshot(), state)
        first, second = random.Random(9), random.Random(9)
        self.assertEqual([sampler.choose_case(cases, rng=first).case_id for _ in range(3)],
                         [restored.choose_case(cases, rng=second).case_id for _ in range(3)])
        # Annotation and catalog disagreements are rejected.
        key = next(iter(shard.annotations))
        broken = {**json_value(shard.annotations), key: {**json_value(shard.annotations[key]), "level": 3}}
        with self.assertRaises(ValueError):
            CurriculumSampler(broken, distribution_sha256=shard.distribution_sha256,
                              categories=self.recipe.category_keys).validate_cases(cases)
        with self.assertRaises(ValueError):
            CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                              categories=("room:1:monster", "card:bash"))
        with self.assertRaises(ValueError):
            CurriculumSampler(shard.annotations, distribution_sha256=shard.distribution_sha256,
                              categories=self.recipe.category_keys, state={**state, "allocated": 1})


def load_cases(shard) -> tuple:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "shard.json"
        path.write_text(json.dumps(json_value(shard.manifest)))
        return load_neural_cases(path).cases


class StandardStreamTests(_StandardFixture):
    def prepare(self, root: Path, index: int):
        from spire_learn.neural.stream_runner import prepare_stream
        template = root / "template.toml"
        if not template.exists():
            template.write_text(standard_template_text(rows=4, steps=8))
        destination = root / f"shard-{index}"
        result = prepare_stream(self.recipe_path, destination, arm="mixed", split="train", shard_index=index,
                                template=template)
        return result, load_neural_config(destination / "config.toml")

    def test_prepare_stream_writes_a_standard_segment(self):
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            result, config = self.prepare(Path(folder), 2)
            self.assertEqual(config.mode, "standard-standalone")
            self.assertEqual(config.scenario_stream["shard_index"], 2)
            self.assertEqual(result["recipe_sha256"], self.recipe.sha256)
            self.assertEqual(result["levels"], [0, 1])
            self.assertEqual(config.manifest_sha256, recipes.generate_shard(
                self.recipe, split="train", shard_index=2, arm="mixed").manifest_sha256)
            # The prepared recipe loads on its own and has the same identity.
            self.assertEqual(recipes.load_recipe(Path(folder) / "shard-2/recipe.json").sha256, self.recipe.sha256)
            from spire_learn.neural.stream_resume import stream_identity
            identity = stream_identity(config)
            self.assertEqual(identity["states_table_sha256"], self.table_sha)
            self.assertNotIn("support_sha256", identity)

    def test_standard_stream_requires_the_standard_learner_identity(self):
        from spire_learn.neural.stream_runner import prepare_stream
        standalone = ('standalone = "spire-learn-standalone-common-v1@'
                      '4c73f7de04d7b2875f473f8034c1cbacf984ea2d52d0a9428289c30cf1dff463+gym-1.1.1"\n')
        variants = {
            "old ids": standard_template_text(rows=4, steps=8, identity="v1"),
            "old ids, mixed mappings": standard_template_text(rows=4, steps=8, common=True, identity="v1").replace(
                "[accepted_identities.accepted_raw_mode_mappings]\n",
                "[accepted_identities.accepted_raw_mode_mappings]\n" + standalone),
        }
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            root = Path(folder)
            _, good = self.prepare(root, 0)
            for name, text in variants.items():
                with self.subTest(name):
                    template = root / f"{len(name)}.toml"
                    template.write_text(text)
                    # Preparation refuses the template, writing nothing.
                    with self.assertRaisesRegex(ValueError, "Standard learner identity"):
                        prepare_stream(self.recipe_path, root / f"bad-{len(name)}", arm="mixed", split="train",
                                       shard_index=0, template=template)
                    self.assertFalse((root / f"bad-{len(name)}").exists())
                    # Config admission refuses a hand-written stream config with that identity.
                    source = (root / "shard-0/config.toml").read_text()
                    pinned = {line.split(" = ")[0]: line for line in source.splitlines()
                              if line.startswith(("manifest_path = ", "output_root = "))}
                    body = "\n".join(pinned.get(line.split(" = ")[0], line) for line in text.splitlines())
                    tail = source[source.index("[scenario_stream]"):]
                    (root / "shard-0/bad.toml").write_text(body + "\n\n" + tail)
                    with self.assertRaisesRegex(ValueError, "Standard learner identity"):
                        load_neural_config(root / "shard-0/bad.toml")
            self.assertEqual(good.accepted_identities[0].model_id, "spire-learn-combat-ff-standard-v1")

    def test_checkpoint_preflight_rejects_cross_family_identities(self):
        from spire_learn.neural.checkpoint import CheckpointCompatibilityError, validate_checkpoint_metadata
        from spire_learn.neural.contracts import NeuralIdentity
        from spire_learn.neural.gym_contracts import contract_for_revision
        from tests.unit.test_neural_checkpoint import _metadata
        contract = contract_for_revision("e3c13836bd1458be8a87b93160f5c350f78fcd80")
        hashes = {"model": "a" * 64, "ppo": "b" * 64}

        def identity(ids, modes):
            return NeuralIdentity(*ids[:3], "spire-learn-combat-winloss-v1",
                                  {mode: contract.raw_mode_mappings[mode] for mode in modes}, hashes)
        v1 = ("spire-learn-combat-ff-v1", "spire-learn-combat-features-v1", "spire-learn-combat-dispatch-v1")
        standard = ("spire-learn-combat-ff-standard-v1", "spire-learn-combat-features-standard-v1",
                    "spire-learn-combat-dispatch-v1")
        m10_only = identity(v1, ("standalone",))
        mixed = identity(v1, ("standalone", "standard-standalone"))
        standard_v1 = identity(v1, ("standard-standalone",))
        standard_new = identity(standard, ("standard-standalone",))
        for saved, expected in ((mixed, m10_only), (m10_only, mixed), (standard_v1, m10_only),
                                (m10_only, standard_v1), (standard_new, standard_v1), (standard_v1, standard_new),
                                (mixed, standard_v1)):
            with self.subTest(saved=dict(saved.accepted_raw_mode_mappings), expected=expected.model_id):
                with self.assertRaises(CheckpointCompatibilityError):
                    validate_checkpoint_metadata(_metadata(identity=saved), expected, purpose="evaluate")
        for same in (m10_only, standard_new, standard_v1):
            validate_checkpoint_metadata(_metadata(identity=same), same, purpose="evaluate")

    def test_resume_across_schedule_boundary_matches_uninterrupted(self):
        from spire_learn.neural.stream_resume import stream_checkpoint_state, validate_stream_resume
        uninterrupted = [recipes.generate_shard(self.recipe, split="train", shard_index=i, arm="mixed")
                         for i in range(5)]
        self.assertGreater(self.gym.calls, 0)
        with tempfile.TemporaryDirectory(dir=WORK) as folder:
            _, before = self.prepare(Path(folder), 1)
            sampler = CurriculumSampler(uninterrupted[1].annotations,
                                        distribution_sha256=uninterrupted[1].distribution_sha256,
                                        categories=self.recipe.category_keys)
            saved = stream_checkpoint_state(before, sampler.snapshot())
            self.assertEqual(saved["next"]["manifest_sha256"], uninterrupted[2].manifest_sha256)
            # A fresh process: empty caches and a new validator object bound to the same Gym.
            ss.clear_shard_cache()
            fresh = FakeGym()
            recipes.bind_validator(fresh)
            _, after = self.prepare(Path(folder), 2)
            self.assertGreater(fresh.calls, 0)
            self.assertEqual(after.manifest_sha256, uninterrupted[2].manifest_sha256)
            metadata = {"scenario_stream": json_value(saved), "config_sha256": before.config_sha256,
                        "manifest_sha256": before.manifest_sha256, "controller_id": "neural-policy-train-v1"}
            admitted = validate_stream_resume(metadata, after)
            self.assertEqual(admitted["destination"]["shard_index"], 2)
        for index, shard in enumerate(uninterrupted):
            again = recipes.generate_shard(self.recipe, split="train", shard_index=index, arm="mixed")
            self.assertEqual(canonical_bytes(again.manifest), canonical_bytes(shard.manifest))
            self.assertEqual(canonical_bytes(again.annotations), canonical_bytes(shard.annotations))
            levels = {row["level"] for row in json_value(shard.annotations).values()}
            self.assertLessEqual(levels, set(ss.levels_for_shard(self.recipe, index)))
        self.assertEqual({row["level"] for row in json_value(uninterrupted[4].annotations).values()}, {0, 1, 2})


def _report(mode: str, *, outcome: str | None, censored: bool = False):
    from spire_learn.neural.runner import NeuralReport
    fights = [] if outcome is None else [{"outcome": outcome, "hp_before_healing": 40, "hp_after_cleanup": 30,
                                          "hp_retention": 0.5}]
    fight = {"reached_combat_denominator": 1, "completed": len(fights),
             "victories": int(outcome == "victory"), "deaths": int(outcome == "death"),
             "escaped": int(outcome == "escaped"), "fights": fights}
    if censored:
        return NeuralReport(summary={
            "status": "incomplete", "stop_reason": "fixed_case_budget_incomplete", "fault": None, "mode": mode,
            "case_accounting": {"cases": [{"case_id": "x", "status": "incomplete"}]},
            "collection": {"fault": None, "faults": 0, "uncertain_environment_slots": 0},
            "fights": {"standalone_fights": fight},
            "lifecycle": {"status": "incomplete", "exit_code": 1, "close_errors": []}}, exit_code=1, run_dir=None)
    return NeuralReport(summary={"status": "complete", "mode": mode, "fights": {"standalone_fights": fight},
                                 "case_accounting": {"cases": [{"case_id": "x", "status": "complete"}]}},
                        exit_code=0, run_dir=None)


def _bind_case(report, case_id):
    report.summary["case_accounting"]["cases"][0]["case_id"] = case_id
    return report


class StandardPanelOutcomeTests(_StandardFixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        previous = recipes.bind_validator(FakeGym())
        try:
            cls.panel = recipes.build_panel(recipes.load_recipe(cls.recipe_path))
        finally:
            recipes.bind_validator(previous)

    def test_escaped_reported_not_scored(self):
        from scripts.neural import baseline_entry
        cases = self.panel["manifest"]["cases"][:3]
        reports = {case["case_id"]: _bind_case(_report("standard-standalone", outcome=outcome), case["case_id"])
                   for case, outcome in zip(cases, ("victory", "death", "escaped"))}
        summary = recipes.summarize_panel_reports(self.panel, reports)
        self.assertEqual(summary["schema"], "spire-learn-standard-panel-summary-v1")
        total = summary["total"]
        self.assertEqual((total["wins"], total["deaths"], total["escaped"]), (1, 1, 1))
        self.assertEqual(total["win_rate"], 0.5)  # escapes are neither wins nor losses
        self.assertEqual(summary["status"], "incomplete")  # 253 cases not reported
        escaped_case = cases[2]["case_id"]
        level = str(self.panel["annotations"][escaped_case]["level"])
        self.assertGreaterEqual(summary["strata"]["level"][level]["escaped"], 1)
        self.assertEqual(baseline_entry._case_outcome(_report("standard-standalone", outcome="escaped")), "escaped")
        paired = baseline_entry._paired({"a": "win", "b": "escaped", "c": "loss"},
                                        {"a": "win", "b": "win", "c": "escaped"}, total=3, standard=True)
        self.assertEqual((paired["paired_cases"], paired["excluded_escaped"]), (1, 2))
        self.assertEqual(paired["excluded_failed_invalid_or_missing"], 0)
        self.assertEqual(paired["final_only_win"], 0)

    def test_m10_paired_dictionary_unchanged(self):
        """M10 pairing is exactly the original: escapes settle as non-wins and no escape key is added."""
        from scripts.neural import baseline_entry
        fresh = {"a": "win", "b": "escaped", "c": "loss", "d": "incomplete", "e": "failed", "f": "loss"}
        final = {"a": "win", "b": "win", "c": "escaped", "d": "win", "e": "win", "f": "win"}
        expected = {"paired_cases": 4, "excluded_censored": 1, "excluded_failed_invalid_or_missing": 2,
                    "both_win": 1, "fresh_only_win": 0, "final_only_win": 2, "neither_win": 1,
                    "fresh_wins": 1, "final_wins": 3, "exact_mcnemar_two_sided_p": 0.5}
        paired = baseline_entry._paired(fresh, final, total=7)
        self.assertEqual(canonical_bytes(paired), canonical_bytes(expected))
        self.assertEqual(list(paired), list(expected))

    def test_long_case_censored_not_failure(self):
        from scripts.neural import baseline_entry
        case = self.panel["manifest"]["cases"][0]
        report = _bind_case(_report("standard-standalone", outcome=None, censored=True), case["case_id"])
        from spire_learn.neural.stream_runner import is_censored_case_report
        self.assertTrue(is_censored_case_report(report))
        summary = recipes.summarize_panel_reports(self.panel, {case["case_id"]: report})
        self.assertEqual(summary["failures"], [])
        self.assertEqual(summary["censored_cases"], [case["case_id"]])
        self.assertEqual(summary["total"]["censored"], 1)
        self.assertNotEqual(summary["status"], "operational_failure")
        self.assertEqual(baseline_entry._case_outcome(report), "incomplete")
        # A faulted incomplete report is still a failure.
        faulted = _bind_case(_report("standard-standalone", outcome=None, censored=True), case["case_id"])
        faulted.summary["fault"] = {"type": "RuntimeError"}
        failed = recipes.summarize_panel_reports(self.panel, {case["case_id"]: faulted})
        self.assertEqual(failed["status"], "operational_failure")
        # An M10 report is not a Standard panel report.
        with self.assertRaises(ValueError):
            recipes.summarize_panel_reports(self.panel, {case["case_id"]: _bind_case(
                _report("standalone", outcome="victory"), case["case_id"])})

    def test_per_panel_case_actions(self):
        """The plan sets case_actions per panel; panels without one keep evaluation.case_actions."""
        from scripts.neural import baseline_entry
        from tests.unit.test_baseline_entry import _Clock, _Fakes
        plan = json.loads((ROOT / "configs/neural/baseline-8726d467/plan.json").read_text())
        plan["additional_panels"] = [{"recipe": "configs/neural/curriculum-v1/recipe.json", "case_actions": 1024,
                                      "case_seconds": 120}]
        fakes = _Fakes(decisions_per_segment=405504, panel_cases=1)
        limits = []
        evaluate = fakes.evaluate

        def recording(config, *, runtime, clock, allow_final):
            text = Path(config.path).read_text()
            limits.append((runtime.operation_limits.max_environment_actions,
                           "max_environment_actions = 1024" in text))
            return evaluate(config, runtime=runtime, clock=clock, allow_final=allow_final)

        fakes.evaluate = recording
        with tempfile.TemporaryDirectory() as folder:
            plan_path = Path(folder) / "plan.json"
            plan_path.write_text(json.dumps(plan))
            output = Path(folder) / "out"
            output.mkdir()
            # The runner contract binds the container's own output; this run is a fake, outside it.
            with patch.dict(os.environ):
                for name in ("SPIRE_LEARN_RUNNER_OUTPUT", "SPIRE_LEARN_RUNNER_TIMEOUT_SECONDS"):
                    os.environ.pop(name, None)
                summary = baseline_entry.run_baseline(ROOT, plan_path, output, dependencies=fakes.deps(),
                                                      clock=_Clock())
        self.assertEqual(summary["result"], "complete")
        self.assertEqual(limits[:4], [(256, False)] * 4)  # primary panel: final, fresh, two nears
        self.assertEqual(limits[4:], [(1024, True)] * 2)  # the additional panel: final, fresh
        self.assertIn("configs/neural/curriculum-v1/recipe.json", summary["additional_panels"])
        for bad in ([{"recipe": "x"}], [{"recipe": "x", "case_actions": 0}],
                    [{"recipe": "x", "case_actions": 8, "other": 1}], ["x", {"recipe": "x", "case_actions": 8}]):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "plan.json"
                path.write_text(json.dumps({**plan, "additional_panels": bad}))
                with self.assertRaises(ValueError):
                    baseline_entry.load_plan(path)


# ---------------------------------------------------------------- gated smoke (runtime container)

_SMOKE_ENCOUNTERS = ((1, "monster", "cultist"), (1, "monster", "jaw_worm"), (1, "elite", "gremlin_nob"),
                     (1, "elite", "lagavulin"), (1, "boss", "hexaghost"))
_SMOKE_CARDS = ("inflame", "shrug_it_off", "pommel_strike", "anger", "clothesline", "iron_wave")
_SMOKE_RELICS = ("vajra", "anchor", "lantern", "akabeko", "bag_of_marbles", "oddly_smooth_stone")


def smoke_states(runs: int = 160, seed: int = 11) -> list[dict]:
    """Conservative Ironclad A20 act-1 states: starter decks, a few commons, counterless relics."""
    rng = random.Random(seed)
    states = []
    for index in range(runs):
        run_id = str(1_700_000_000 + index * 613)
        for floor in (3, 6, 9):
            act, kind, encounter = _SMOKE_ENCOUNTERS[rng.randrange(len(_SMOKE_ENCOUNTERS))]
            deck = ([{"card": "strike", "upgrades": 0, "misc": 0}] * 5 + [{"card": "defend", "upgrades": 0, "misc": 0}] * 4
                    + [{"card": "bash", "upgrades": rng.randrange(2), "misc": 0}])
            deck += [{"card": rng.choice(_SMOKE_CARDS), "upgrades": rng.randrange(2), "misc": 0}
                     for _ in range(rng.randrange(5))]
            deck = sorted(deck, key=lambda r: (r["card"], r["upgrades"]))
            relics = ["burning_blood"] + rng.sample(_SMOKE_RELICS, rng.randrange(3))
            entry = {"id": f"{run_id}-{floor}", "character": "ironclad", "ascension": 20, "encounter": encounter,
                     "hp": 60, "max_hp": 80, "gold": 99, "deck": deck, "relics": relics, "relic_counters": {},
                     "potions": [None, None]}
            states.append({"run_id": run_id, "character": "ironclad", "floor": floor, "act": act,
                           "room_kind": kind, "enemies": encounter, "deck": deck, "relics": relics,
                           "relic_counters": {}, "bottled_cards": {}, "potions": [], "hp": 60, "max_hp": 80,
                           "gold": 99, "burning": False, "won": True, "tags": [], "entry": entry})
    return states


@unittest.skipUnless(_RUN_NATIVE, "set SPIRE_LEARN_RUN_NATIVE_INTEGRATION=1 in the approved runtime")
class StandardStreamSmokeTest(unittest.TestCase):
    def test_standard_stream_smoke(self):
        """Two PPO updates at 16 rows on a synthetic table: shard 0 (level 0), then resume on fuzzed shard 1."""
        from spire_learn.neural.runner import RuntimeOperationLimits
        from spire_learn.neural.standard_catalog import validate_entries
        from spire_learn.neural.stream_runner import prepare_stream, run_stream_segment
        previous = recipes.bind_validator(None)  # the real Gym, bound lazily in this process
        self.addCleanup(recipes.bind_validator, previous)
        ss.clear_shard_cache()
        root = WORK / f"standard-stream-smoke-{os.getpid()}"
        root.mkdir()
        _, table_sha = write_table(root, smoke_states())
        recipe_path = root / "recipe.json"
        recipe_path.write_text(json.dumps(recipe_document(
            table_sha, shard_size=32, redraw_bound=4,
            fuzz_schedule=[{"from_shard": 0, "levels": [0]}, {"from_shard": 1, "levels": [0, 1, 2]}])))
        recipe = recipes.load_recipe(recipe_path)
        # Every base state the shards draw from is accepted by the Gym.
        _, rejected = validate_entries([dict(json_value(s["entry"])) for s in recipe.states_for("train")],
                                       recipes.gym_validator())
        self.assertEqual(rejected, [])
        template = root / "template.toml"
        template.write_text(standard_template_text(rows=16, steps=8).replace(
            "max_training_decisions = 256", "max_training_decisions = 4096").replace(
            "max_environment_actions = 1024", "max_environment_actions = 4096").replace(
            "max_wall_seconds = 120.0", "max_wall_seconds = 1200.0"))
        evidence = {"recipe_sha256": recipe.sha256, "table_sha256": table_sha, "segments": []}
        parent = None
        for index, purpose in ((0, "train"), (1, "resume")):
            prepared = prepare_stream(recipe_path, root / f"shard-{index}", arm="mixed", split="train",
                                      shard_index=index, template=template)
            config = load_neural_config(root / f"shard-{index}/config.toml")
            started = time.monotonic()
            result = run_stream_segment(config, purpose=purpose, checkpoint=parent,
                                        operation_limits=RuntimeOperationLimits(
                                            max_updates=index + 1, max_environment_actions=10**6,
                                            monotonic_deadline=time.monotonic() + 1500))
            report = result["report"]
            self.assertEqual((report.exit_code, report.summary.get("status")), (0, "complete"), report.summary)
            self.assertIsNotNone(result["checkpoint"])
            self.assertEqual(result["saved_prefix"]["update_count"], index + 1)
            self.assertEqual(result["next_shard"]["shard_index"], index + 1)
            annotations = json_value(recipes.generate_shard(recipe, split="train", shard_index=index,
                                                            arm="mixed").annotations)
            evidence["segments"].append({
                "shard_index": index, "purpose": purpose, "levels": prepared["levels"],
                "level_counts": prepared["level_counts"], "skipped": prepared["skipped"],
                "manifest_sha256": prepared["manifest_sha256"], "seconds": round(time.monotonic() - started, 1),
                "saved_prefix": json_value(result["saved_prefix"]),
                "fuzzed_cases": sum(row["level"] > 0 for row in annotations.values()),
                "fights": json_value(report.summary.get("fights", {}).get("standalone_fights", {}).get(
                    "completed")),
                "checkpoint": str(result["checkpoint"])})
            parent = result["checkpoint"]
        self.assertGreater(evidence["segments"][1]["fuzzed_cases"], 0)
        (root / "smoke-evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
