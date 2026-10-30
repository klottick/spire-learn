"""Pure fixed inputs for the sustained Ironclad campaign; no runtime dispatch.

Authoring refuses differing existing bytes. Validation regenerates the approved
inputs from bound historical templates and verifies real parser identities and
all six parent files. Execution release and operation consumption live elsewhere.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spire_learn.neural.cases import load_neural_cases, validate_neural_case_lineage
from spire_learn.neural.config import load_neural_config

BASE_PACKAGE = 'configs/neural/experiments/sustained-v1'
CAMPAIGN_PATH = BASE_PACKAGE + '/campaign.json'
TRAINING_CASES = 'configs/cases/neural/sustained-v1/train.json'
SELECTION_CASES = 'configs/cases/neural/sustained-v1/selection.json'
ARTIFACT_PREFIX = 'artifacts/development/neural-combat/experiments/sustained-v1'
PARENT_PATH = 'artifacts/development/neural-combat/sustained-v1/preparation/parent-checkpoints.json'
TRAINING_ORDER = ('standalone-1', 'run-1', 'standalone-2', 'run-2')
TRAINING_SEEDS = dict(zip(TRAINING_ORDER, (
    dict(zip(('policy','optimizer','curriculum','noncombat'), row)) for row in (
        (1101103,1102207,1103301,1104409), (1301103,1302207,1303301,1304409),
        (1201103,1202207,1203301,1204409), (1401103,1402207,1403301,1404409)))))
ENCOUNTERS = ('cultist','jaw_worm','two_louse','small_slimes','large_slime',
              'lots_of_slimes','three_louse','gremlin_nob','lagavulin','three_sentries',
              'guardian','hexaghost','slime_boss')
SENTRY_INDICES = (0,4,8,12,16,20,24,28,32,33,34,35,36,37,38,39)
# These inputs are historical, immutable sources, not editable campaign parameters.
SOURCE_HASHES = {'configs/cases/neural/train.json': '1bbb41954481cc411dfa9e5ecc5d1cd63d6ba568a81f256269a0bc4163640ed4',
 'configs/cases/neural/final.json': '5b300377431bf16821799bdbd0a706ed634d3c568323a8a2e2ac1c59e58572b9',
 'configs/neural/experiments/task9-v1/study.json': 'c4939fc7d882fadb265fbc5a4ced42c7809e6482720dc4bac7698188aa14f555',
 'configs/neural/experiments/task9-v1/standalone-b.toml': 'c1e3eda0e7bbe7140022d1832cb694455a17c6e007eae2935e6cec0e4cf8b3a4',
 'configs/neural/experiments/task9-v1/run-b.toml': 'a8c1412d4977c90645c2e122f43e4f178beb1524b02e0f82d73a23bb2ab1c989',
 'configs/neural/experiments/task9-v1/selection/case-00-standalone.toml': '29910b52ff03c81f55ec71dfcc2702b6ee74a0f42804baa95183b8d1b00f2430',
 'configs/neural/experiments/task9-v1/selection/case-00-run.toml': '987348367c6a286409a8f55b3bb79f46735edd41cd82239f9b87e1dc2b96ae6a',
 'artifacts/development/neural-combat/sustained-v1/preparation/parent-checkpoints.json': '72237b90c6211fb24900f7481aa7b101e9a8da208d16ba68b6fa7f180fb237e2'}

DIAGNOSTIC_CHECKPOINT = {'path': 'artifacts/development/neural-combat/experiments/task9-v1-attempt-b/training/standalone-a/20261030T204522Z-bb8ee549fe1a4817bdc444bf474327c9/checkpoint-final',
 'inventory': {'metadata.json': {'sha256': '59e9cbd80d7696298759169640814756d8102b081b21d4c9495b913332ba6e2d',
                                 'size_bytes': 17150},
               'model.pt': {'sha256': 'e75ce4a34a46e3bc076548b9ef8ca6caf80dc7e1a9e56c9ca8da41481b1ab557',
                            'size_bytes': 1089216},
               'training.pt': {'sha256': '7a887f788f7904c25445fb598360d224ff4e14527e449b7786b85a4888d81036',
                               'size_bytes': 2211765}}}

BUDGETS = {
    'campaign': {'whole_wall_seconds':28800, 'one_runtime_job_at_a_time':True},
    'profile': {'whole_wall_seconds':900, 'environment_attempts':8192,
                'sample_rows':16384, 'optimizer_attempts':256, 'trace_episodes':8,
                'trace_actions_per_episode':32, 'trace_decision_bytes':65536,
                'trace_total_bytes':16777216},
    'replica': {'whole_wall_seconds':6000, 'learned_decisions':500000,
                'environment_attempts':1000000, 'optimizer_attempts':32768,
                'max_segments':256, 'training_sample_attempts':500000},
    'segment': {'whole_wall_seconds':600, 'environment_attempts':4096,
                'standalone_updates':8, 'run_updates':32, 'initial_reset_slots':16,
                'reset_rule':'environment_attempts_plus_initial_slots_per_admitted_segment'},
    'evaluation': {'whole_wall_seconds':2700, 'operations':704,
                   'reset_slots_per_operation':1, 'actions_per_operation':256,
                   'reserved_action_slots':180224, 'optimizer_attempts':0},
    'shared': {'whole_wall_seconds':1200},
    'storage': {'new_bytes':12*1024**3, 'cumulative_bytes':40*1024**3,
                'minimum_free_bytes':30*1024**3, 'pruning':False},
    'unchanged_guards': {'consumed_ids':65536, 'event_log_bytes':64*1024**2},
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def _path(root: Path, relative: str) -> Path:
    from scripts.neural.task9_study_package import _bound_path
    return _bound_path(root, relative, 'campaign input')


def _read(root: Path, relative: str) -> bytes:
    return _path(root, relative).read_bytes()


def _sources(root: Path) -> dict[str, bytes]:
    result = {}
    for path, digest in SOURCE_HASHES.items():
        raw = _read(root, path)
        if _sha(raw) != digest:
            raise ValueError(f'historical input hash mismatch: {path}')
        result[path] = raw
    return result


def _manifest(split: str, cases: list[dict]) -> bytes:
    return _json({'schema':'spire-learn-neural-cases-v1', 'split':split,
                  'partition':split, 'cases':cases})


def _case(index: int, split: str, template: dict, entry: dict | None,
          cohort: str) -> dict:
    mode = 'standalone' if entry is not None else 'run'
    identifier = f'sustained-{split}-{mode}-{index:02d}'
    result = copy.deepcopy(template)
    result.update(case_id=identifier, lineage_id=f'lineage:sustained-v1:{split}:{mode}:{index:02d}',
                  environment_seed=(index if split=='train' else 2**40+2**20+index),cohort=cohort)
    if entry is not None:
        result['entry_id'] = identifier
        result['entry'] = copy.deepcopy(entry)
        result['entry']['id'] = identifier
    return result


def _case_inputs(source: dict[str, bytes]) -> tuple[list[dict], list[dict]]:
    old = json.loads(source['configs/cases/neural/train.json'])['cases']
    standalone = next(row for row in old if row['mode']=='standalone')
    run = next(row for row in old if row['mode']=='run')
    entries = [old[i]['entry'] for i in (0,2,4)]
    training = []
    for encounter in ENCOUNTERS:
        for template in entries:
            for hp in (64,80):
                entry = copy.deepcopy(template)
                entry.update(encounter=encounter,hp=hp)
                training.append(_case(len(training),'train',standalone,entry,'train_catalog'))
    training.append(_case(78,'train',run,None,'normal_run'))
    selection = []
    encounters = list(range(13))+[0,1,2]+[0,2,4,6,8,10,11,12]+[1,3,5,7,9,10,11,12]
    for i, encounter_index in enumerate(encounters):
        entry = copy.deepcopy(entries[0])
        entry.update(encounter=ENCOUNTERS[encounter_index], hp=64)
        cohort = 'familiar' if i<16 else 'deck' if i<24 else 'hp'
        if cohort=='deck':
            entry['deck'] += [{'card':'shrug_it_off','upgrades':0} for _ in range(2)]
        if cohort=='hp':
            entry['hp'] = 56 if i%2==0 else 72
        selection.append(_case(i,'selection',standalone,entry,cohort))
    selection.extend(_case(i,'selection',run,None,'normal_run') for i in range(32,64))
    return training, selection


def _config(source: bytes, *, manifest: str, output: str, training: bool,
            policy: str | None = None) -> bytes:
    import re
    text = source.decode()
    for key, value in (('manifest_path',manifest),('output_root',output)):
        text = re.sub(r'^'+key+r' = .*$', key+' = '+json.dumps(value), text, flags=re.M)
    if training:
        assert policy is not None
        for key, value in TRAINING_SEEDS[policy].items():
            text = re.sub(r'^'+key+r' = \d+$',f'{key} = {value}',text,flags=re.M)
        for key, value in (('max_optimizer_steps',32768),('max_training_decisions',500000),
                           ('max_environment_actions',1000000),('max_wall_seconds',6000.0)):
            text = re.sub(r'^'+key+r' = [0-9.]+$',f'{key} = {value}',text,flags=re.M)
    return text.encode()


def _expected(root: Path) -> tuple[dict[str, bytes], dict]:
    source = _sources(root)
    from scripts.neural.task9_study_package import _RUNTIME_PINS, _same_json_value
    study = json.loads(source['configs/neural/experiments/task9-v1/study.json'])
    if not _same_json_value(study['runtime_pins'], _RUNTIME_PINS):
        raise ValueError('qualified runtime pins differ')
    parents = json.loads(source[PARENT_PATH])
    if set(parents) != {'standalone-b','run-b'}:
        raise ValueError('parent inventory must contain exactly Standalone B and Run B')
    for parent in (*parents.values(), DIAGNOSTIC_CHECKPOINT):
        if set(parent) != {'path','inventory'} or set(parent['inventory']) != {'metadata.json','model.pt','training.pt'}:
            raise ValueError('parent inventory must bind all three checkpoint files')
        for name, binding in parent['inventory'].items():
            raw = _read(root,parent['path']+'/'+name)
            if binding != {'sha256':_sha(raw),'size_bytes':len(raw)}:
                raise ValueError('parent checkpoint hash/size mismatch')
    train, selection = _case_inputs(source)
    files = {TRAINING_CASES:_manifest('train',train), SELECTION_CASES:_manifest('selection',selection)}
    runs = []
    for policy in TRAINING_ORDER:
        mode = policy.split('-')[0]
        relative = f'{BASE_PACKAGE}/{policy}.toml'
        output = f'/learner/{ARTIFACT_PREFIX}/training/{policy}'
        files[relative] = _config(source[f'configs/neural/experiments/task9-v1/{mode}-b.toml'],
            manifest='../../../../'+TRAINING_CASES,output='../../../../'+ARTIFACT_PREFIX+'/training/'+policy,training=True,policy=policy)
        runs.append({'policy_id':policy,'mode':mode,'parent_id':mode+'-b',
                     'seeds':copy.deepcopy(TRAINING_SEEDS[policy]), 'config':{'path':relative},'output_root':output,
                     'rollout':{'rows':16 if mode=='standalone' else 1,'steps':32 if mode=='standalone' else 128}})
    bindings = []
    for i, case in enumerate(selection):
        manifest = f'configs/cases/neural/sustained-v1/selection/case-{i:02d}.json'
        relative = f'{BASE_PACKAGE}/selection/case-{i:02d}.toml'
        files[manifest] = _manifest('selection',[case])
        output = f'/learner/{ARTIFACT_PREFIX}/selection/{case["case_id"]}'
        files[relative] = _config(source[f'configs/neural/experiments/task9-v1/selection/case-00-{case["mode"]}.toml'],
            manifest='../../../../../'+manifest,output='../../../../../'+ARTIFACT_PREFIX+'/selection/'+case['case_id'],training=False)
        bindings.append({'case_index':i,'case_id':case['case_id'],'lineage_id':case['lineage_id'],
                         'mode':case['mode'],'cohort':case['cohort'],'environment_seed':case['environment_seed'],
                         'config_path':relative,'manifest_path':manifest,'output_root':output})
    campaign = {'schema':'spire-learn-sustained-campaign-v1','state':'prepared_not_released',
        'base_commit':'930951cb77329273e06ece73d5ba88a2f18def19',
        'runtime_pins':study['runtime_pins'],'budgets':copy.deepcopy(BUDGETS), 'parents':parents,
        'diagnostic_checkpoints':{'standalone-a':copy.deepcopy(DIAGNOSTIC_CHECKPOINT)},
        'sources':copy.deepcopy(SOURCE_HASHES), 'files':{path:{'raw_sha256':_sha(raw),'size_bytes':len(raw)} for path,raw in files.items()},
        'training':{'order':list(TRAINING_ORDER),'manifest':{'path':TRAINING_CASES},'runs':runs},
        'selection':{'manifest':{'path':SELECTION_CASES},'case_bindings':bindings,
                     'sentry_case_ids':[selection[i]['case_id'] for i in SENTRY_INDICES],
                     'milestones':{'25':1500,'50':3000,'75':4500},'operations':704},
        'interpretation':{'final_split':'not_used','environment_seed_policy':'environment-seed-for-replica0-v1',
            'matched_streams':True,'parental_exposure':True,'no_training_adaptation_from_selection':True,
            'future_checkpoint_hashes':'absent_until_actual_publication','native_episodes_resume':False,
            'fault_policy':'halt_no_retry_no_substitute',
            'qualifications':['RNG-001','TIME-001','finite supported content','simulator-only evidence',
                              'model-selection evidence','catalog-relative strata; no Run or parent non-exposure claim']}}
    # Canonical identities are computed from the exact authored inputs before
    # writes, then independently checked against real loaders by validation.
    import tomllib
    def canonical(raw):
        return _sha(json.dumps(raw, sort_keys=True, ensure_ascii=False,
                               separators=(",",":"), allow_nan=False).encode())
    for key in ('training','selection'):
        binding = campaign[key]['manifest']
        binding['canonical_sha256'] = canonical(json.loads(files[binding['path']]))
    for run in runs:
        path = run['config']['path']
        run['config'].update(config_sha256=canonical(tomllib.loads(files[path].decode())),
                             raw_sha256=_sha(files[path]))
    for binding in bindings:
        path = binding['config_path']
        manifest = binding['manifest_path']
        binding.update(config_sha256=canonical(tomllib.loads(files[path].decode())),
                       config_raw_sha256=_sha(files[path]),
                       manifest_sha256=canonical(json.loads(files[manifest])),
                       manifest_raw_sha256=_sha(files[manifest]))
    return files, campaign


def _bind_parsers(root: Path, campaign: dict) -> None:
    """Bind identities returned by actual loaders, retaining raw full config hashes."""
    from scripts.neural.task9_study_package import _validate_common_config
    for key in ('training','selection'):
        binding = campaign[key]['manifest']
        binding['canonical_sha256'] = load_neural_cases(_path(root,binding['path'])).sha256
    validate_neural_case_lineage(tuple(load_neural_cases(_path(root,p)) for p in
        (TRAINING_CASES,SELECTION_CASES,'configs/cases/neural/final.json')))
    verified_parents = set()
    for run in campaign['training']['runs']:
        config = load_neural_config(_path(root,run['config']['path']))
        _validate_common_config(config,run['policy_id'])
        if run['parent_id'] not in verified_parents:
            from spire_learn.neural.checkpoint import preflight_checkpoint
            preflight_checkpoint(_path(root,campaign['parents'][run['parent_id']]['path']),
                                 config.accepted_identities[0],purpose='warm_start')
            verified_parents.add(run['parent_id'])
        run['config'].update(config_sha256=config.config_sha256,
                             raw_sha256=_sha(_read(root,run['config']['path'])))
    preflight_checkpoint(_path(root,campaign['diagnostic_checkpoints']['standalone-a']['path']),
                         config.accepted_identities[0],purpose='evaluate')
    for binding in campaign['selection']['case_bindings']:
        config = load_neural_config(_path(root,binding['config_path']))
        _validate_common_config(config,binding['case_id'])
        binding.update(config_sha256=config.config_sha256,
                       config_raw_sha256=_sha(_read(root,binding['config_path'])),
                       manifest_sha256=config.cases.sha256,
                       manifest_raw_sha256=_sha(_read(root,binding['manifest_path'])))


def author_campaign(root: Path) -> dict:
    """Create fixed inputs, accepting existing files only when bytes are identical.

    All destinations are checked before any write. Exclusive creation preserves
    unfamiliar concurrent files. This never publishes an execution release.
    """
    root = Path(root).resolve()
    files, campaign = _expected(root)
    files[CAMPAIGN_PATH] = _json(campaign)
    for path, raw in files.items():
        destination = _path(root,path)
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError(f'refusing to overwrite unfamiliar authored file: {path}')
    for path, raw in files.items():
        destination = _path(root,path)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            with destination.open('xb') as stream:
                stream.write(raw)
    return validate_campaign(root, root/CAMPAIGN_PATH)


def validate_campaign(root: Path, campaign_path: Path) -> dict:
    """Reject altered bytes, identities, budgets, parents or future claims."""
    root = Path(root).resolve()
    candidate = Path(campaign_path)
    if not candidate.is_absolute():
        candidate = root/candidate
    if candidate != _path(root,CAMPAIGN_PATH):
        raise ValueError('campaign path must be its fixed authored slot')
    files, expected = _expected(root)
    for path, raw in files.items():
        if _read(root,path) != raw:
            raise ValueError(f'fixed campaign file bytes differ: {path}')
    parsed = copy.deepcopy(expected)
    _bind_parsers(root,parsed)
    if parsed != expected:
        raise ValueError("actual parser identities differ from authored bindings")
    if candidate.read_bytes() != _json(expected):
        raise ValueError('campaign bytes differ from fixed bindings/budgets')
    return expected


def evaluation_schedule(campaign: Mapping[str, object], checkpoints: Mapping[str, object]) -> tuple[dict, ...]:
    """Return all 704 planned operations without asserting any execution.

    Optional checkpoint bindings use '<policy>:<slot>' keys, slots 25/50/75/final.
    Parent anchors use their verified campaign inventories. Missing future slots
    remain unavailable, with no hash. Returned rows are independent deep copies.
    An operation consumer must persist/claim operation_id before any effect and
    reject a previously consumed ID; pure schedule regeneration grants no retry.
    """
    if not isinstance(checkpoints, Mapping):
        raise TypeError('checkpoints must be a mapping')
    allowed = {f'{policy}:{slot}' for policy in TRAINING_ORDER
               for slot in ('25','50','75','final')}
    if any(key not in allowed for key in checkpoints):
        raise ValueError('unknown checkpoint policy or slot')
    for binding in checkpoints.values():
        if binding is None:
            continue
        if (not isinstance(binding, Mapping) or not isinstance(binding.get('path'), str)
                or not binding['path'] or not isinstance(binding.get('inventory'), Mapping)
                or set(binding['inventory']) != {'metadata.json','model.pt','training.pt'}):
            raise ValueError('checkpoint binding requires path and actual three-file inventory')
        from scripts.neural.task9_study_package import _check_sha
        for item in binding['inventory'].values():
            if (not isinstance(item, Mapping) or set(item) != {'sha256','size_bytes'}
                    or type(item['size_bytes']) is not int or item['size_bytes'] <= 0):
                raise ValueError('checkpoint inventory requires digest and positive byte size')
            _check_sha(item['sha256'],'actual checkpoint inventory')
    cases = campaign['selection']['case_bindings']
    sentry = set(campaign['selection']['sentry_case_ids'])
    slots = [('25',[c for c in cases if c['case_id'] in sentry]),
             ('50',[c for c in cases if c['case_id'] in sentry]),
             ('75',[c for c in cases if c['case_id'] in sentry]),('final',cases)]
    rows = []
    from spire_learn.neural.checkpoint import _strict_json
    def append(policy, slot, case, checkpoint, reference=False):
        rows.append({'operation_id':f'sustained-v1:{policy}:{slot}:{case["case_id"]}',
            'policy_id':policy,'checkpoint_slot':slot,'case_id':case['case_id'],
            'case_index':case['case_index'],'mode':case['mode'],
            'config':{'path':case['config_path'],'config_sha256':case['config_sha256'],
                      'raw_sha256':case['config_raw_sha256']},
            'state':'not_started','availability':'reference' if reference else 'bound' if checkpoint is not None else 'unavailable',
            'checkpoint':_strict_json(checkpoint), 'reset_reservation':1,'action_reservation':256})
    for slot,panel in slots:
        for case in panel:
            for policy in TRAINING_ORDER:
                append(policy,slot,case,checkpoints.get(f'{policy}:{slot}'))
    for case in cases:
        for policy in ('standalone-b','run-b','legal-random','end-turn-last-v1'):
            reference = policy in ('legal-random','end-turn-last-v1')
            append(policy,'reference' if reference else 'anchor',case,
                   None if reference else campaign['parents'][policy],reference)
    if len(rows)!=704 or len({r['operation_id'] for r in rows})!=704:
        raise ValueError('evaluation schedule must have 704 unique operations')
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--author',action='store_true')
    operation.add_argument('--validate',action='store_true')
    args = parser.parse_args(argv)
    try:
        result = author_campaign(args.root) if args.author else validate_campaign(args.root,args.root/CAMPAIGN_PATH)
    except (ValueError,OSError,TypeError,KeyError) as error:
        print(f'sustained-package: {error}',file=sys.stderr)
        return 2
    print(json.dumps({'schema':result['schema'],'state':result['state'],
                      'training_runs':4,'selection_cases':64,'evaluation_operations':len(evaluation_schedule(result,{})),
                      'runtime_dispatch':False},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
