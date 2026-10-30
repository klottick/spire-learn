"""Authored CDI metadata fixtures; no Docker, GPU, native or optional backend."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HOOK = '/run/nvidia-ctk-hook12345678-1234-4123-8123-123456789abc'
SOCKET = '/run/nvidia-persistenced/socket'
PARAMS = '/proc/driver/nvidia/params'
DRIVER_FILES = [
    '/usr/bin/' + n for n in ('nvidia-smi', 'nvidia-debugdump', 'nvidia-persistenced',
                              'nvidia-cuda-mps-control', 'nvidia-cuda-mps-server')
] + ['/usr/lib/x86_64-linux-gnu/' + n for n in (
    'libcuda.so.595.91.07', 'libcudadebugger.so.595.91.07',
    'libnvidia-cfg.so.595.91.07', 'libnvidia-ml.so.595.91.07',
    'libnvidia-nvvm.so.595.91.07', 'libnvidia-nvvm70.so.4',
    'libnvidia-opencl.so.595.91.07', 'libnvidia-pkcs11-openssl3.so.595.91.07',
    'libnvidia-pkcs11.so.595.91.07', 'libnvidia-ptxjitcompiler.so.595.91.07',
    'libnvidia-tileiras.so.595.91.07')
] + ['/etc/OpenCL/vendors/nvidia.icd',
     '/usr/lib/firmware/nvidia/595.91.07/gsp_ga10x.bin',
     '/usr/lib/firmware/nvidia/595.91.07/gsp_tu10x.bin']


def file_fact(kind=stat.S_IFREG, mode=0o444, uid=0, gid=0, writable=False,
              device=41, inode=7):
    return {'mode': kind | mode, 'uid': uid, 'gid': gid, 'effective_write_access': writable,
            'device': device, 'inode': inode, 'size_bytes': 1000}


def cdi_facts(base=None):
    """Semantic fixture: representative UUID/inodes, canonical public paths."""
    facts = deepcopy(base) if base is not None else {
        'mounts': [('/', True)], 'environment': {}, 'uid': 1000, 'gid': 1000,
        'python': '/runtime/venv-cuda/bin/python3.12', 'device_nodes': ['/dev/nvidia0']}
    facts['environment']['NVIDIA_VISIBLE_DEVICES'] = 'void'
    facts['environment']['NVIDIA_CTK_LIBCUDA_DIR'] = '/usr/lib/x86_64-linux-gnu'
    facts['mounts'] += [(p, True) for p in DRIVER_FILES] + [(SOCKET, True), (HOOK, False), (PARAMS, False)]
    facts['regular_mount_files'] = set(DRIVER_FILES + [PARAMS])
    metadata = {p: file_fact() for p in DRIVER_FILES}
    metadata[SOCKET] = file_fact(stat.S_IFSOCK, 0o777, 105, 108, True)
    metadata[PARAMS] = file_fact()
    metadata[HOOK] = {**file_fact(stat.S_IFDIR, 0o1777, writable=True),
        'capacity_bytes': 4096, 'available_bytes': 0, 'inode_capacity': 100000,
        'free_inodes': 99999, 'children': {'nvct-params': file_fact()}, 'children_truncated': False}
    facts['runtime_mount_metadata'] = metadata
    facts['runtime_mount_types'] = {p: {'fstype': 'tmpfs', 'mount_options': ['ro'] if ro else ['rw'],
                                      'super_options': ['rw', 'size=4k']} for p, ro in facts['mounts']}
    facts['runtime_mount_types'][SOCKET]['mount_options']=['ro','nosuid','nodev','noexec']
    facts['supplementary_group_ids'] = [44, 991, 1000]
    return facts


class CdiIntegrationTests(unittest.TestCase):
    def test_actual_b_cli_accepts_cdi_first_phase_then_bounded_stop(self):
        from tests.unit.test_neural_study_controls import ControlFixture
        with tempfile.TemporaryDirectory() as d:
            fixture = ControlFixture(d)
            old=fixture.root/'artifacts/development/neural-combat/experiments/task9-v1/training/standalone-a'
            old.mkdir(parents=True)
            old_claim=fixture.root/fixture.c.CLAIMS_DOMAIN/'prior-consumed'
            old_claim.mkdir(parents=True);(old_claim/'campaign-claim.json').write_bytes(b'consumed historical fixture\n')
            calls = []
            def mutate(facts):
                calls.append(True)
                if len(calls) > 1:
                    raise RuntimeError('bounded fixture stop after completed first phase')
                self.assertEqual(list((fixture.root/fixture.c.EXPERIMENT/'training/standalone-a').iterdir()),[])
                facts.update(cdi_facts(facts))
                # Distinct from static profile and default fixture: retained facts
                # must carry these observed identifiers/accounting values.
                facts['runtime_mount_metadata'][HOOK]['inode_capacity']=77777
                facts['runtime_mount_metadata'][HOOK]['free_inodes']=77776
                for row in (facts['runtime_mount_metadata'][PARAMS],
                            facts['runtime_mount_metadata'][HOOK]['children']['nvct-params']):
                    row['device']=63;row['inode']=991
            fixture.guest_mutation = mutate
            result = fixture.campaign()
            self.assertEqual(result['phases'][0]['status'], 'complete', fixture.last_guest_stderr)
            self.assertEqual(len(fixture.factory_calls), 1)
            self.assertEqual(fixture.factory_calls[0][0], 'standalone-a')
            fixture.c.verify_closed_inventory(fixture.h)
            self.assertEqual(list(old.iterdir()), [])
            self.assertEqual((old_claim/'campaign-claim.json').read_bytes(), b'consumed historical fixture\n')
            self.assertTrue((fixture.root/fixture.c.EXPERIMENT/'training/standalone-a/operation-receipt.json').is_file())
            context=json.loads((fixture.h/'standalone-a/phase.json').read_text())
            self.assertEqual(context['runtime_profile']['profile_id'],'task9-nvidia-cdi-1.20.0-595.91.07-v1')
            retained=json.loads((fixture.h/'standalone-a-commands.json').read_text())
            start=next(row for row in retained['commands'] if 'start' in row['argv'])
            marker='SPIRE_LEARN_TASK9_ADMISSION_V1 '
            lines=[line for line in start['stderr'].splitlines() if line.startswith(marker)]
            self.assertEqual(len(lines),1,'actual admission record must survive B->host captured stderr')
            self.assertLessEqual(len((lines[0]+'\n').encode()),16384)
            admission=json.loads(lines[0][len(marker):])
            self.assertEqual(admission['schema'],'spire-learn-task9-guest-admission-v1')
            self.assertEqual(admission['phase'],'standalone-a')
            self.assertEqual(admission['campaign_id'],'fixture-a')
            self.assertEqual(admission['context_raw_sha256'],fixture.c.sha256_file(fixture.h/'standalone-a/phase.json'))
            self.assertEqual(admission['study_raw_sha256'],fixture.c.STUDY_SHA256)
            self.assertEqual(admission['profile_binding'],context['runtime_profile'])
            self.assertEqual(admission['process'],{'uid':1000,'gid':1000,
                'python':'/runtime/venv-cuda/bin/python3.12','supplementary_group_ids':[44,991,1000]})
            self.assertEqual(admission['observed_environment'],{
                'NVIDIA_VISIBLE_DEVICES':'void','NVIDIA_CTK_LIBCUDA_DIR':'/usr/lib/x86_64-linux-gnu'})
            observed=admission['runtime_profile']['runtime_exceptions']
            self.assertEqual(observed['scratch_path'],HOOK)
            self.assertEqual(observed['scratch']['inode_capacity'],77777)
            self.assertEqual(observed['scratch']['free_inodes'],77776)
            self.assertTrue(observed['scratch']['effective_write_access'])
            self.assertEqual(observed['scratch']['capacity_bytes'],4096)
            self.assertIn('no separate hard inode quota',observed['scratch']['inode_qualification'])
            self.assertEqual(observed['driver_params']['inode'],991)
            self.assertEqual(observed['scratch']['children']['nvct-params']['device'],63)
            self.assertTrue(observed['socket']['effective_write_access'])
            closed=fixture.c.verify_closed_inventory(fixture.h)
            self.assertTrue(any('standalone-a-commands.json' in str(row) for row in closed['entries']))
            stage=fixture.h/'source'
            self.assertTrue((stage/'scripts/neural/task9_nvidia_cdi.py').is_file())
            guard="""import importlib.abc,sys,json
from pathlib import Path
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','tensordict','torchrl','gymnasium','spire_lab'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from scripts.neural import task9_study_runtime as b,task9_study_controls as c,task9_nvidia_cdi as d
root=Path.cwd().resolve()
for module in (b,c,d):assert Path(module.__file__).resolve().is_relative_to(root),module.__file__
assert len(c._study(root)['selection']['schedule'])==192
assert d.load_runtime_profile(root)['profile_id']=='task9-nvidia-cdi-1.20.0-595.91.07-v1'
assert not any(n.split('.')[0] in {'torch','numpy','tensordict','torchrl','gymnasium','spire_lab'} for n in sys.modules)
print('actual staged B/C/sixth helper/profile validated under six-prefix guard')
"""
            result=subprocess.run([sys.executable,'-B','-c',guard],cwd=stage,
                env={'PATH':os.defpath,'PYTHONPATH':str(stage/'src'),'PYTHONNOUSERSITE':'1'},
                capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('actual staged',result.stdout)

    def test_actual_guest_cli_rejects_cdi_faults_before_factory(self):
        from tests.unit.test_neural_study_controls import ControlFixture
        for fault in ('hook_size','params_identity','extra_driver'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d)
                def mutate(facts):
                    facts.update(cdi_facts(facts))
                    if fault=='hook_size':facts['runtime_mount_metadata'][HOOK]['capacity_bytes']=8192
                    if fault=='params_identity':facts['runtime_mount_metadata'][PARAMS]['inode']=17
                    if fault=='extra_driver':facts['mounts'].append(('/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.08',True))
                f.guest_mutation=mutate;result=f.campaign()
                self.assertEqual(result['phases'][0]['status'],'operational_failure')
                self.assertEqual(f.factory_calls,[])
                self.assertIn('phase-context verification failed',f.last_guest_stderr)
                self.assertNotIn('SPIRE_LEARN_TASK9_ADMISSION_V1 ',f.last_guest_stderr)
                f.c.verify_closed_inventory(f.h)

    def test_admission_serialization_size_and_stderr_faults_never_enter_factory(self):
        from tests.unit.test_neural_study_controls import ControlFixture
        class Stream:
            def __init__(self,target,fault):self.target=target;self.fault=fault;self.seen=False
            def write(self,text):
                if text.startswith('SPIRE_LEARN_TASK9_ADMISSION_V1 '):
                    self.seen=True
                    if self.fault=='write':raise OSError('injected admission write failure')
                    if self.fault=='short':return len(text)-1
                    if self.fault=='control':raise KeyboardInterrupt('admission control')
                return self.target.write(text)
            def flush(self):
                if self.seen and self.fault=='flush':raise OSError('injected admission flush failure')
                return self.target.flush()
        for fault in ('oversize','unserializable','write','short','flush','control'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);calls=[]
                def mutate(facts):
                    calls.append(True)
                    if len(calls)>1:raise RuntimeError('bounded fixture stop after first phase')
                    facts.update(cdi_facts(facts))
                    if fault in ('oversize','unserializable'):
                        facts['runtime_mount_metadata'][HOOK]['children']['nvct-params']['extra_evidence']=(
                            'x'*17000 if fault=='oversize' else object())
                    else:sys.stderr=Stream(sys.stderr,fault)
                f.guest_mutation=mutate
                if fault=='control':
                    with self.assertRaisesRegex(KeyboardInterrupt,'admission control'):f.campaign()
                else:
                    result=f.campaign()
                    self.assertEqual(result['phases'][0]['status'],'operational_failure')
                    if fault=='oversize':self.assertIn('admission evidence exceeds',f.last_guest_stderr)
                    if fault=='unserializable':self.assertIn('serializable',f.last_guest_stderr)
                self.assertEqual(f.factory_calls,[])
                self.assertFalse((f.root/f.c.EXPERIMENT/'training/standalone-a/operation-reservation.json').exists())
                f.c.verify_closed_inventory(f.h)

    def test_actual_host_profile_refuses_before_claim_and_rechecks_failed_final_closure(self):
        from tests.unit.test_neural_study_controls import ControlFixture
        from scripts.neural import task9_nvidia_cdi as api
        for drift in ('preclaim','failed_closure'):
            with self.subTest(drift=drift),tempfile.TemporaryDirectory() as d:
                f=ControlFixture(d);real_runtime=f.c._verify_runtime;real_sha=f.c.sha256_file
                for path in (f.c.LOCK,f.c.NATIVE):
                    (f.root/path).parent.mkdir(parents=True,exist_ok=True);(f.root/path).write_bytes(b'authored runtime placeholder')
                qualification=f.root/f.c.RUNTIME/'device-check-work/device-check.json'
                qualification.parent.mkdir(parents=True);qualification.write_text(json.dumps({
                    'status':'qualified','gpu_uuid':f.c.GPU_UUID,'visible_gpu_uuid':f.c.GPU_UUID,
                    'torch':'2.14.1+cu130','torch_cuda':'13.0','compute_capability':[12,0]}))
                f.runtime=lambda root,approval:real_runtime(root,approval)
                profile=api.load_runtime_profile(f.root)
                hashes={row['path']:row['sha256'] for row in profile['host_inputs'].values()};calls=[]
                def host_hash(path,maximum):
                    calls.append(path)
                    bad=drift=='preclaim' or (f.h/'standalone-a-result.json').exists()
                    return '0'*64 if bad else hashes[path]
                def sha(path):
                    return {f.root/f.c.LOCK:f.c.LOCK_SHA256,f.root/f.c.NATIVE:f.c.NATIVE_SHA256}.get(Path(path),real_sha(path))
                if drift=='failed_closure':f.failure='start-timeout'
                with patch.object(api,'_host_file_sha256',side_effect=host_hash), \
                     patch.object(f.c,'sha256_file',side_effect=sha), \
                     patch.object(f.c,'_runtime_metadata',return_value={'torch':'2.14.1+cu130','numpy':'2.5.4','gymnasium':'1.4.0','spire-lab':'0.1.0'}):
                    if drift=='preclaim':
                        result=f.campaign()
                        self.assertEqual(result['campaign_claim_state'],'unclaimed')
                        self.assertFalse((f.root/f.c.CLAIMS_DOMAIN/'fixture-a').exists())
                        self.assertFalse(any('create' in argv for argv,kw in f.commands))
                    else:
                        with self.assertRaises(ValueError) as caught:f.campaign()
                        self.assertIn('bounded command timeout',str(caught.exception))
                        self.assertTrue(any('host cdi_spec hash' in note for note in caught.exception.__notes__))
                        failure=json.loads((f.h/'closure-failure.json').read_text())
                        self.assertTrue(any('host cdi_spec hash' in note for note in failure['error']['notes']))
                        self.assertGreaterEqual(calls.count('/var/run/cdi/nvidia.yaml'),4)
                self.assertEqual(f.factory_calls,[])

    def test_package_routes_all_outputs_to_fresh_namespace(self):
        from scripts.neural.task9_study_package import validate_study
        study = validate_study(ROOT, ROOT/'configs/neural/experiments/task9-v1/study.json')
        for row in study['training']['runs'] + study['selection']['case_bindings']:
            self.assertTrue(row['output_root'].startswith(
                'artifacts/development/neural-combat/experiments/task9-v1-attempt-b/'), row['output_root'])


class CdiProfileTests(unittest.TestCase):
    def api(self):
        path = ROOT/'scripts/neural/task9_nvidia_cdi.py'
        self.assertTrue(path.exists(), 'CDI profile helper is not implemented')
        from scripts.neural import task9_nvidia_cdi
        return task9_nvidia_cdi

    def validate(self, facts=None, phase='standalone-a'):
        api = self.api()
        requested = {'NVIDIA_VISIBLE_DEVICES': 'GPU-c05e0ce3-c1c2-0c54-14e6-8a4a2eccaa1d',
                     'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '4'}
        actual = dict(requested, NVIDIA_VISIBLE_DEVICES='void',
                      NVIDIA_CTK_LIBCUDA_DIR='/usr/lib/x86_64-linux-gnu')
        facts = cdi_facts() if facts is None else facts
        actual.update(facts.get('environment', {}))
        return api.validate_guest_profile(api.load_runtime_profile(ROOT), phase=phase,
            requested_environment=requested, actual_environment=actual, facts=facts)

    def test_exact_cdi_records_actual_projection_and_explicit_runtime_exceptions(self):
        result = self.validate()
        self.assertEqual(result['mode'], 'nvidia-cdi')
        self.assertEqual(result['accepted_environment_changes']['NVIDIA_VISIBLE_DEVICES'], 'void')
        self.assertEqual(set(result['admitted_mount_destinations']), set(DRIVER_FILES+[SOCKET,HOOK,PARAMS]))
        self.assertTrue(result['runtime_exceptions']['scratch']['effective_write_access'])
        self.assertEqual(result['runtime_exceptions']['scratch']['capacity_bytes'], 4096)
        self.assertIn('no separate hard inode quota', result['runtime_exceptions']['scratch']['inode_qualification'])

    def test_unrecognized_controlled_or_extra_environment_is_rejected(self):
        for key,value in [('NVIDIA_VISIBLE_DEVICES','all'), ('NVIDIA_CTK_LIBCUDA_DIR','/tmp'),
                          ('OMP_NUM_THREADS','8'), ('LD_PRELOAD','/tmp/evil'),
                          ('NVIDIA_UNREVIEWED','yes')]:
            with self.subTest(key=key):
                facts=cdi_facts();facts['environment'][key]=value
                with self.assertRaisesRegex(ValueError, key):self.validate(facts)

    def test_profile_runtime_classes_fail_closed_on_metadata_tampering(self):
        mutations = [
            lambda f: f['mounts'].append(('/var/run/docker.sock',True)),
            lambda f: f['mounts'].append((HOOK+'/nested',True)),
            lambda f: f['mounts'].append((HOOK.replace('12345678','22345678',1),False)),
            lambda f: f['runtime_mount_metadata'][HOOK].update(capacity_bytes=8192),
            lambda f: f['runtime_mount_metadata'][HOOK].update(mode=stat.S_IFDIR|0o777),
            lambda f: f['runtime_mount_metadata'][HOOK].update(uid=1000),
            lambda f: f['runtime_mount_metadata'][HOOK].update(children_truncated=True),
            lambda f: f['runtime_mount_metadata'][HOOK]['children'].update(extra=file_fact()),
            lambda f: f['runtime_mount_metadata'][HOOK]['children'].clear(),
            lambda f: f['runtime_mount_metadata'][HOOK].update(free_inodes=100001),
            lambda f: f['runtime_mount_metadata'][HOOK].update(inode_capacity=0),
            lambda f: f['runtime_mount_metadata'][HOOK]['children']['nvct-params'].update(mode=stat.S_IFLNK|0o777),
            lambda f: f['runtime_mount_metadata'][HOOK]['children']['nvct-params'].update(effective_write_access=True),
            lambda f: f['runtime_mount_metadata'][PARAMS].update(inode=8),
            lambda f: f['runtime_mount_metadata'][PARAMS].update(device=42),
            lambda f: f['runtime_mount_metadata'][HOOK]['children']['nvct-params'].update(mode=stat.S_IFREG|0o644),
            lambda f: f['runtime_mount_metadata'][PARAMS].update(effective_write_access=True),
            lambda f: f['runtime_mount_metadata'][SOCKET].update(mode=stat.S_IFREG|0o777),
            lambda f: f['runtime_mount_metadata'][SOCKET].update(uid=0),
            lambda f: f['runtime_mount_metadata'][SOCKET].update(effective_write_access=False),
            lambda f: f['runtime_mount_types'][HOOK].update(fstype='ext4'),
            lambda f: f.update(supplementary_group_ids=[1000]),
            lambda f: f['runtime_mount_metadata'].pop(HOOK),
        ]
        for i,mutate in enumerate(mutations):
            with self.subTest(i=i):
                facts=cdi_facts();mutate(facts)
                with self.assertRaises(ValueError):self.validate(facts)

    def test_boolean_owner_or_available_space_and_wrong_socket_options_are_not_facts(self):
        for destination,key,value in [(HOOK,'uid',False),(SOCKET,'uid',True),
                                      (PARAMS,'gid',False),(HOOK,'available_bytes',False)]:
            with self.subTest(destination=destination,key=key):
                f=cdi_facts();f['runtime_mount_metadata'][destination][key]=value
                with self.assertRaises(ValueError):self.validate(f)
        f=cdi_facts();f['runtime_mount_types'][SOCKET]['mount_options']=['ro']
        with self.assertRaises(ValueError):self.validate(f)

    def test_collector_gets_identity_access_and_bounded_directory_metadata_without_reading_contents(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);directory=root/'hook';directory.mkdir();child=directory/'nvct-params'
            child.write_bytes(b'authored params fixture');params=root/'params';os.link(child,params)
            paths={HOOK:directory,PARAMS:params}
            raw=''.join(f'1 0 0:41 / {p} rw,relatime - tmpfs tmpfs rw,size=4k\n' for p in paths)
            actual_path=Path
            with patch.object(api,'Path',side_effect=lambda p:paths.get(str(p),actual_path(p))), \
                 patch.object(api.os,'access',return_value=False) as access:
                result=api.collect_runtime_mount_facts(raw)
            observed=result['runtime_mount_metadata']
            self.assertEqual(observed[HOOK]['children']['nvct-params']['inode'],observed[PARAMS]['inode'])
            self.assertEqual(observed[HOOK]['children']['nvct-params']['device'],observed[PARAMS]['device'])
            self.assertEqual(result['runtime_mount_types'][HOOK]['fstype'],'tmpfs')
            self.assertGreater(observed[HOOK]['inode_capacity'],0)
            self.assertFalse(observed[HOOK]['children_truncated'])
            self.assertTrue(all(c.kwargs=={'effective_ids':True} for c in access.call_args_list))
            for i in range(17):(directory/str(i)).touch()
            with patch.object(api,'Path',side_effect=lambda p:paths.get(str(p),actual_path(p))):
                result=api.collect_runtime_mount_facts(raw)
            self.assertTrue(result['runtime_mount_metadata'][HOOK]['children_truncated'])
            self.assertEqual(len(result['runtime_mount_metadata'][HOOK]['children']),16)

    def test_each_exact_driver_file_requires_readonly_regular_nonwritable_evidence(self):
        for path in DRIVER_FILES:
            for change in ('rw','directory','writable','missing'):
                with self.subTest(path=path,change=change):
                    f=cdi_facts()
                    if change=='rw':f['mounts']=[(p,False if p==path else ro) for p,ro in f['mounts']]
                    if change=='directory':f['runtime_mount_metadata'][path]['mode']=stat.S_IFDIR|0o555
                    if change=='writable':f['runtime_mount_metadata'][path]['effective_write_access']=True
                    if change=='missing':f['runtime_mount_metadata'].pop(path)
                    with self.assertRaises(ValueError):self.validate(f)

    def test_cpu_refuses_cdi_paths_even_in_otherwise_allowed_proc_namespace(self):
        api=self.api();profile=api.load_runtime_profile(ROOT)
        requested={'NVIDIA_VISIBLE_DEVICES':'none','CUDA_VISIBLE_DEVICES':''}
        for dest in DRIVER_FILES+[SOCKET,HOOK,PARAMS]:
            with self.subTest(dest=dest):
                with self.assertRaises(ValueError):
                    api.validate_guest_profile(profile,phase='selection',requested_environment=requested,
                        actual_environment=requested,facts={'mounts':[(dest,True)],'device_nodes':[]})
        with self.assertRaises(ValueError):
            api.validate_guest_profile(profile,phase='selection',requested_environment=requested,
                actual_environment={**requested,'NVIDIA_CTK_LIBCUDA_DIR':'/usr/lib/x86_64-linux-gnu'},
                facts={'mounts':[],'device_nodes':[]})

    def test_raw_profile_and_host_hash_drift_refuse_admission(self):
        api=self.api();profile=api.load_runtime_profile(ROOT)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'configs/runtime/task9-nvidia-cdi-profile.json'
            path.parent.mkdir(parents=True);path.write_bytes((ROOT/path.relative_to(root)).read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'profile.*hash'):api.load_runtime_profile(root)
        expected={item['path']:item['sha256'] for item in profile['host_inputs'].values()}
        with patch.object(api,'_host_file_sha256',side_effect=lambda path,maximum:expected[path]):
            result=api.verify_host_profile(profile)
            self.assertEqual(result['profile_id'],profile['profile_id'])
        for bad_path in expected:
            with self.subTest(path=bad_path), patch.object(api,'_host_file_sha256',
                    side_effect=lambda path,maximum: '0'*64 if path==bad_path else expected[path]):
                with self.assertRaisesRegex(ValueError,'hash'):api.verify_host_profile(profile)


if __name__ == '__main__':
    unittest.main()
