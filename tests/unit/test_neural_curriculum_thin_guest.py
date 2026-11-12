"""Thin continuation admission, with only OS/optional-runtime boundaries faked."""
from contextlib import ExitStack, redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.neural import curriculum_controls as controls
from scripts.neural import curriculum_package as package
from scripts.neural import curriculum_profiles as profiles
from scripts.neural import curriculum_runtime as producer
from tests.unit.test_neural_curriculum_integration import blocked_optional
from tests.unit.test_neural_study_cdi import cdi_facts

ROOT = Path(__file__).resolve().parents[2]
CACHED = ROOT/'artifacts/development/neural-combat/twenty-hour-20261103-continuation-2'
PROFILE = profiles.TWENTY_HOUR_CONTINUATION_2.name


class AdmittedProducerBoundary(Exception):
    pass


class ThinGuestTests(unittest.TestCase):
    """A host-history read in cap math must fail this actual entry regression."""
    profile = PROFILE

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.base = Path(self.stack.enter_context(tempfile.TemporaryDirectory(
            dir=ROOT/'artifacts/development/neural-combat')))
        self.stack.enter_context(profiles.scope(self.profile))
        self.host, self.guest, self.control = (self.base/name for name in ('host', 'guest', 'control'))
        self.host.mkdir()
        self.package_path = CACHED/'inputs/package-a/package.json'
        self.package = package.read_json(self.package_path)
        if profiles.is_repaired_twenty_hour_continuation():
            # Rebind the descriptor and sole authorized-gap config, reusing
            # cached selected cases. Never generate/copy the full finite stream.
            from spire_learn.neural.stream_runner import _toml
            import tomllib
            campaign = package.read_json(ROOT/profiles.current().campaign_path)
            selected = self.base/'selected'
            names = {name for row in self.package['panel']['cases'] for name in row['inputs']}
            names.add(self.package['panel']['path'])
            names.update(self.package['training']['mixed'][0]['inputs'])
            for name in names:
                target = selected/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.package_path.parent/name, target)
            config_path = selected/'training/mixed/config-339.toml'
            raw = tomllib.loads(config_path.read_text())
            raw['scenario_stream']['resume_skip'] = json.dumps(
                campaign['continuation']['resume_skip_authorization'], sort_keys=True, separators=(',', ':'))
            config_path.write_text(_toml(raw))
            binding = package._config_row(selected, 'training/mixed/config-339.toml')
            self.package['training']['mixed'][0] = binding
            self.package['inventory']['training/mixed/config-339.toml'] = package.inventory(
                selected, ['training/mixed/config-339.toml'])['training/mixed/config-339.toml']
            self.package.update(campaign=campaign, campaign_id=campaign['campaign_id'],
                continuation=campaign['continuation'])
            self.package_path = selected/'package.json'
            controls._write(self.package_path, self.package)
        self.source_path = self.base/'source.json'
        snapshot = controls.capture_source(ROOT, self.source_path, learner_head='a'*40,
            dirty={'test': 'thin source projection of current checkout'})
        release = package.read_json(CACHED/'control/campaign-release.json')
        if profiles.is_repaired_twenty_hour_continuation():
            release.update(campaign_id=self.package['campaign_id'],
                campaign_raw_sha256=package.sha256(ROOT/profiles.current().campaign_path),
                package_sha256=package.sha256(self.package_path),
                caps=controls.release_caps(ROOT),
                qualification_input_inventory=controls.qualification_inventory(ROOT),
                user_approval=dict(path=profiles.current().user_approval_path,
                    sha256=package.sha256(ROOT/profiles.current().user_approval_path)))
            release['runtime_pins']['profile'] = profiles.current().name
        release['source_snapshot_sha256'] = package.sha256(self.source_path)
        self.release_path, self.claim_path = self.base/'release.json', self.host/'campaign-claim.json'
        controls._write(self.release_path, release)
        controls._write(self.claim_path, dict(schema=controls.CLAIM_SCHEMA,
            campaign_id=self.package['campaign_id'], source_sha256=package.sha256(self.source_path),
            release_sha256=package.sha256(self.release_path), package_sha256=package.sha256(self.package_path)))
        self.verified = dict(snapshot=snapshot, release=release, package=self.package,
            package_path=self.package_path, source_snapshot=self.source_path,
            release_path=self.release_path, claim_path=self.claim_path)
        controls.initialize_stage(ROOT, self.host, self.verified)
        self.mapping = {'/learner': self.guest, '/control': self.control,
            '/inputs/parent': self.base/'parent',
            '/runtime/venv-cuda': ROOT/controls.low.VENV,
            '/runtime/requirements-cuda.lock': ROOT/controls.low.LOCK,
            profiles.OVERLAY_DESTINATION: ROOT/profiles.current().package_overlay}
        self.stack.enter_context(patch.object(controls.low, '_guest_path', side_effect=self.mapped))
        self.budget = controls.CampaignBudget(release['caps']['global_effect_caps'], start=100.,
            deadline=66098., clock=lambda: 100.)

    def mapped(self, value):
        path = Path(value)
        for name in sorted(self.mapping, key=len, reverse=True):
            if path == Path(name) or path.is_relative_to(name):
                return self.mapping[name]/path.relative_to(name)
        return path

    def prepare(self, kind='qualification'):
        parent = self.package['original_parent']
        options = {}
        if kind == 'segment':
            lineage = self.package['continuation']
            retained = self.host/'lineage/receipt337.json'
            retained.parent.mkdir()
            shutil.copyfile(ROOT/lineage['parent_receipt_path'], retained)
            options = dict(arm='mixed', index=339, prefix=lineage['parent_prefix'],
                predecessor=dict(path=str(retained), sha256=lineage['parent_receipt_sha256']),
                source_transition=dict(schema=producer.SOURCE_TRANSITION_SCHEMA,
                    purpose='authorized-checkpoint-continuation',
                    from_source_sha256=lineage['parent_source_sha256'],
                    to_source_sha256=package.sha256(self.source_path),
                    predecessor_sha256=lineage['parent_receipt_sha256'],
                    checkpoint_inventory_sha256=lineage['parent_checkpoint_inventory_sha256'],
                    user_approval_sha256=self.verified['release']['user_approval']['sha256']))
        elif kind == 'panel':
            options = dict(slot='continuation-parent-shard337')
        self.prepared = controls.prepare_job(ROOT, self.host, self.verified, self.budget,
            kind=kind, job_id='curriculum-thin-'+kind,
            parent=dict(path=str(ROOT/parent['checkpoint_directory']), inventory=parent['inventory']), **options)
        self.context = self.prepared['context']
        shutil.copytree(self.host/'source', self.guest)
        for source, destination, readonly in self.prepared['spec'].mounts:
            path = Path(destination)
            if path == self.guest or not path.is_relative_to(self.base):
                continue
            if path.exists():
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, path) if Path(source).is_dir() else shutil.copyfile(source, path)
        self.assertFalse((self.guest/'artifacts/development/neural-combat/twenty-hour-20261103/campaign-release-approved.json').exists())
        return self.context

    def rewrite(self, name, value):
        (self.control/name).write_text(json.dumps(value, sort_keys=True)+'\n')

    def bind_control_chain(self):
        context = self.context
        claim = package.read_json(self.control/'claim.json')
        campaign_claim = package.read_json(self.control/'campaign-claim.json')
        for name, key in (('release.json', 'release_sha256'), ('snapshot.json', 'source_sha256')):
            digest = package.sha256(self.control/name)
            context[key] = claim[key] = campaign_claim[key] = digest
        self.rewrite('campaign-claim.json', campaign_claim)
        claim['campaign_claim_sha256'] = package.sha256(self.control/'campaign-claim.json')
        self.rewrite('claim.json', claim)
        context['claim_sha256'] = package.sha256(self.control/'claim.json')
        self.rewrite('job.json', context)

    def enter(self):
        admitted = []
        def stop(root, context, **kwargs):
            self.assertIs(producer._ADMITTED_CONTEXT, context)
            admitted.append(context)
            raise AdmittedProducerBoundary('verified entry reached producer')
        context_path = self.control/'job.json'
        layout = package.read_json(self.control/'layout.json')
        spec = controls._spec(self.context, context_path, layout)
        facts = dict(mounts=[('/', True)]+[(d, r) for _, d, r in spec.mounts],
            environment=dict(spec.environment), uid=1000, gid=1000,
            python=controls.low.CONTAINER_PYTHON, device_nodes=[], supplementary_group_ids=[1000])
        if spec.uses_gpu:
            facts = cdi_facts(facts)
        stderr = io.StringIO()
        with blocked_optional() as imports, redirect_stderr(stderr), \
                patch.dict(os.environ, dict(spec.environment), clear=True), \
                patch.object(controls.low, '_guest_facts', return_value=facts), \
                patch.object(profiles, 'verify_imported_overlay', return_value={'test_boundary': 'optional native import'}), \
                patch.object(producer, 'run_'+self.context['kind'], side_effect=stop):
            with self.assertRaises(AdmittedProducerBoundary):
                producer._guest_entry(self.guest, context_path, clock=lambda: 100.)
        self.assertEqual(imports, [])
        self.assertEqual(len(admitted), 1)
        self.assertIn(controls.transport.MARKER, stderr.getvalue())
        self.assertIsNone(producer._ADMITTED_CONTEXT)
        return admitted[0]

    def test_actual_qualification_entry_needs_only_declared_thin_mounts(self):
        self.prepare()
        self.assertEqual(self.enter()['kind'], 'qualification')

    def test_actual_resume339_entry_uses_checkpoint337(self):
        self.prepare('segment')
        admitted = self.enter()
        self.assertEqual(admitted['shard_index'], 339)
        self.assertEqual(admitted['parent']['path'], '/inputs/parent')
        self.assertEqual(admitted['prefix']['learned_decisions'], 335872)

    def test_actual_cpu_baseline_entry_is_thin_and_deterministic(self):
        self.prepare('panel')
        admitted = self.enter()
        self.assertEqual((admitted['runtime_class'], admitted['slot']), ('cpu', 'continuation-parent-shard337'))
        self.assertEqual(admitted['parent']['inventory'], self.package['original_parent']['inventory'])
        env = dict(self.prepared['spec'].environment)
        self.assertEqual((env['NVIDIA_VISIBLE_DEVICES'], env['CUDA_VISIBLE_DEVICES']), ('none', ''))
        self.assertEqual(admitted['reservation']['optimizer_attempts'], 0)

    def test_changed_released_cap_is_rejected_before_producer(self):
        self.prepare()
        release = package.read_json(self.control/'release.json')
        release['caps']['global_effect_caps']['sample_rows'] += 1
        self.rewrite('release.json', release)
        self.bind_control_chain()
        with self.assertRaisesRegex(ValueError, 'acyclic authority'):
            self.enter()

    def test_changed_source_is_rejected_before_producer(self):
        self.prepare()
        with (self.guest/'src/spire_learn/__init__.py').open('a') as stream:
            stream.write('\n# drift\n')
        with self.assertRaisesRegex(ValueError, 'inventory input changed'):
            self.enter()

    def test_changed_parent_is_rejected_before_producer(self):
        self.prepare()
        parent = Path(self.context['parent']['path'])
        (parent/'metadata.json').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'parent inventory'):
            self.enter()

    def test_changed_bridge_authority_is_rejected_before_producer(self):
        self.prepare('segment')
        self.context['source_transition']['user_approval_sha256'] = '0'*64
        claim = package.read_json(self.control/'claim.json')
        claim['source_transition'] = self.context['source_transition']
        self.rewrite('claim.json', claim)
        self.bind_control_chain()
        with self.assertRaisesRegex(ValueError, 'source transition binding'):
            self.enter()


class RepairedThinGuestTests(ThinGuestTests):
    profile = 'ironclad-twenty-hour-continuation-20261103-repaired'


class PredecessorProjectionTests(unittest.TestCase):
    def projection(self, root):
        for name in (profiles.current().campaign_path, profiles.current().predecessor_path):
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)

    def test_caps_derive_without_host_history_but_debit_and_lease_stay_exact(self):
        with profiles.scope(PROFILE), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.projection(root)
            caps = controls.release_caps(root)['global_effect_caps']
            self.assertEqual(caps['environment_attempts'], 4538880)
            path = root/profiles.current().predecessor_path
            pristine = package.read_json(path)
            for mutate in ('charged_or_reserved', 'failed_shard'):
                changed = copy.deepcopy(pristine)
                if mutate == 'charged_or_reserved':
                    changed[mutate]['sample_rows'] -= 512
                else:
                    changed[mutate]['lease']['sample_rows'] -= 1
                path.write_text(json.dumps(changed))
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    controls.release_caps(root)
            path.write_text(json.dumps(pristine))

    def test_host_release_admission_still_rejects_corrupted_physical_history(self):
        with profiles.scope(PROFILE), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.projection(root)
            old = root/controls.TWENTY_HOUR_CONTINUATION_ARTIFACT_PATHS['campaign_release_approved']
            old.parent.mkdir(parents=True)
            old.write_text('{}\n')
            release, snapshot = root/'release.json', root/'snapshot.json'
            release.write_text('{}\n')
            snapshot.write_text(json.dumps(dict(schema=controls.SOURCE_SCHEMA, state='source_only',
                learner_head='a'*40, dirty={}, files={'entry': {}}, modes={'entry': 0o644})))
            with self.assertRaisesRegex(ValueError, 'raw artifact changed: campaign_release_approved'):
                controls.validate_release(root, package_path=root/'package.json',
                    source_snapshot=snapshot, release_path=release)
