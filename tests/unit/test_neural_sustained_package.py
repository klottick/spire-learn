"""Pure fixed-package authoring, parser and selection scheduling contracts."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from spire_learn.neural.cases import load_neural_cases, validate_neural_case_lineage
from spire_learn.neural.config import load_neural_config

ROOT = Path(__file__).resolve().parents[2]

class SustainedPackageTests(unittest.TestCase):
    def api(self):
        try:
            from scripts.neural import sustained_package
        except ImportError as error:
            self.fail(f"sustained package API absent: {error}")
        return sustained_package

    def test_fixed_package_roundtrips_with_exact_replica_contracts(self):
        api = self.api()
        campaign = api.validate_campaign(ROOT, ROOT / api.CAMPAIGN_PATH)
        self.assertEqual(campaign['state'], 'prepared_not_released')
        self.assertEqual(campaign['diagnostic_checkpoints']['standalone-a']['inventory']['metadata.json']['sha256'],
                         '59e9cbd80d7696298759169640814756d8102b081b21d4c9495b913332ba6e2d')
        runs = campaign['training']['runs']
        self.assertEqual([r['policy_id'] for r in runs], ['standalone-1','run-1','standalone-2','run-2'])
        seeds = [(1101103,1102207,1103301,1104409),(1301103,1302207,1303301,1304409),
                 (1201103,1202207,1203301,1204409),(1401103,1402207,1403301,1404409)]
        for run, quartet in zip(runs, seeds):
            config = load_neural_config(ROOT / run['config']['path'])
            self.assertEqual(tuple(getattr(config.seed_streams,k) for k in ('policy','optimizer','curriculum','noncombat')), quartet)
            self.assertEqual((config.budgets.max_training_decisions,config.budgets.max_environment_actions,
                              config.budgets.max_optimizer_steps,config.budgets.max_wall_seconds),(500000,1000000,32768,6000.0))
            self.assertEqual(run['parent_id'], 'standalone-b' if config.mode == 'standalone' else 'run-b')
            self.assertEqual(config.output_root, ROOT/'artifacts/development/neural-combat/experiments/sustained-v1/training'/run['policy_id'])
            self.assertEqual(run['output_root'], '/learner/artifacts/development/neural-combat/experiments/sustained-v1/training/'+run['policy_id'])
            self.assertEqual(dict(config.accepted_identities[0].config_hashes), {
                'model':'4fb628e06bf4952847795ac5d8548d12dfeede4c97e874ab843857ae0f91cd2f',
                'ppo':'f08075e0b01a15095b542fc20cba11c3258381bcb5721bb5066a4448e43c9e2a'})
        train = load_neural_cases(ROOT/api.TRAINING_CASES)
        selection = load_neural_cases(ROOT/api.SELECTION_CASES)
        validate_neural_case_lineage((train, selection, load_neural_cases(ROOT/'configs/cases/neural/final.json')))
        self.assertEqual((len(train.for_mode('standalone')),len(train.for_mode('run'))),(78,1))
        self.assertEqual(len({(c.entry['encounter'], c.entry['hp'],
            tuple((r['card'],r['upgrades']) for r in c.entry['deck']))
            for c in train.for_mode('standalone')}),78)
        encounters = ('cultist','jaw_worm','two_louse','small_slimes','large_slime',
            'lots_of_slimes','three_louse','gremlin_nob','lagavulin','three_sentries','guardian','hexaghost','slime_boss')
        self.assertEqual({c.entry['encounter'] for c in train.for_mode('standalone')},set(encounters))
        self.assertEqual([c.entry['encounter'] for c in selection.cases[:16]],list(encounters)+list(encounters[:3]))
        self.assertEqual([c.entry['encounter'] for c in selection.cases[16:24]],
                         [encounters[i] for i in (0,2,4,6,8,10,11,12)])
        self.assertEqual([c.entry['encounter'] for c in selection.cases[24:32]],
                         [encounters[i] for i in (1,3,5,7,9,10,11,12)])
        self.assertEqual([c.entry['hp'] for c in selection.cases[24:32]],[56,72]*4)
        for case in selection.cases[16:24]:
            self.assertEqual(list(case.entry['deck'][-2:]),[{'card':'shrug_it_off','upgrades':0}]*2)
        self.assertEqual([c.environment_seed for c in selection.cases],list(range(2**40+2**20,2**40+2**20+64)))
        self.assertEqual([c.mode for c in selection.cases],['standalone']*32+['run']*32)
        self.assertEqual(campaign['selection']['sentry_case_ids'],[selection.cases[i].case_id for i in (0,4,8,12,16,20,24,28,32,33,34,35,36,37,38,39)])
        for binding in campaign['selection']['case_bindings']:
            config = load_neural_config(ROOT/binding['config_path'])
            self.assertEqual(len(config.cases.cases),1)
            self.assertEqual((config.operation,config.device,config.budgets.max_environment_actions),('evaluate','cpu',256))

    def test_schedule_retains_unavailable_rows_and_only_supplied_hashes(self):
        api = self.api()
        campaign = api.validate_campaign(ROOT, ROOT/api.CAMPAIGN_PATH)
        rows = api.evaluation_schedule(campaign,{})
        self.assertEqual(len(rows),704)
        self.assertEqual(len({r['operation_id'] for r in rows}),704)
        self.assertEqual(sum(r['availability']=='unavailable' for r in rows),448)
        self.assertTrue(all(r['state']=='not_started' for r in rows))
        self.assertTrue(all(r['checkpoint'] is None for r in rows if r['availability']=='unavailable'))
        binding = {'path':'actual/checkpoint','inventory':{name:{'sha256':'a'*64,'size_bytes':123} for name in ('metadata.json','model.pt','training.pt')}}
        bound = api.evaluation_schedule(campaign,{'standalone-1:25':binding})
        chosen = [r for r in bound if r['policy_id']=='standalone-1' and r['checkpoint_slot']=='25']
        self.assertEqual(len(chosen),16)
        self.assertTrue(all(r['checkpoint']==binding and r['availability']=='bound' for r in chosen))

    def test_schedule_accepts_immutable_inventory_mappings_and_owns_its_result(self):
        from types import MappingProxyType
        api = self.api()
        campaign = api.validate_campaign(ROOT,ROOT/api.CAMPAIGN_PATH)
        inventory = {name:MappingProxyType({'sha256':'a'*64,'size_bytes':123})
                     for name in ('metadata.json','model.pt','training.pt')}
        binding = MappingProxyType({'path':'actual/checkpoint','inventory':MappingProxyType(inventory)})
        rows = api.evaluation_schedule(campaign,MappingProxyType({'standalone-1:25':binding}))
        chosen = [row for row in rows if row['policy_id']=='standalone-1' and row['checkpoint_slot']=='25']
        chosen[0]['checkpoint']['inventory']['model.pt']['size_bytes'] = 999
        self.assertEqual(chosen[1]['checkpoint']['inventory']['model.pt']['size_bytes'],123)
        self.assertEqual(inventory['model.pt']['size_bytes'],123)

    def test_schedule_rejects_hashless_or_unknown_checkpoint_bindings(self):
        api = self.api()
        campaign = api.validate_campaign(ROOT,ROOT/api.CAMPAIGN_PATH)
        for bindings in ({'standalone-1:25':{}}, {'standalone-1:25':{'path':'x','inventory':{}}},
                         {'typo:25':None}):
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                api.evaluation_schedule(campaign,bindings)

    def test_author_refuses_unfamiliar_bytes_and_validator_refuses_mutation(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT/'configs',root/'configs')
            preparation = Path('artifacts/development/neural-combat/sustained-v1/preparation/parent-checkpoints.json')
            (root/preparation).parent.mkdir(parents=True)
            shutil.copy2(ROOT/preparation,root/preparation)
            diagnostic = api.validate_campaign(ROOT,ROOT/api.CAMPAIGN_PATH)['diagnostic_checkpoints']['standalone-a']
            for parent in (*json.loads((ROOT/preparation).read_text()).values(), diagnostic):
                shutil.copytree(ROOT/parent['path'],root/parent['path'])
            # The loader scopes output paths to its own checkout. Run the
            # copied source API in that checkout, as the guest projection does.
            import subprocess, os
            shutil.copytree(ROOT/'src',root/'src')
            shutil.copytree(ROOT/'scripts',root/'scripts')
            def invoke(operation):
                return subprocess.run(['python3.12','-B','scripts/neural/sustained_package.py',
                    '--root',str(root),operation],cwd=root,env={**os.environ,'PYTHONPATH':'src'},
                    capture_output=True,text=True)
            first = invoke('--author')
            self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual(invoke('--author').stdout,first.stdout)
            path = root/api.BASE_PACKAGE/'standalone-1.toml'
            path.write_text(path.read_text()+'\n# unfamiliar\n')
            self.assertNotEqual(invoke('--author').returncode,0)
            self.assertNotEqual(invoke('--validate').returncode,0)
