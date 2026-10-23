"""CPU-only two-GPU ownership, independent failure and lifecycle audit tests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from unittest.mock import Mock, patch
import unittest

from ditserve.evaluation import workflow_overlap_bundle as bundle
from ditserve.evaluation import workflow_overlap_isolated as isolated
from ditserve.evaluation.workflow_budget import atomic_json
from test_workflow_overlap_isolated import config as isolated_config, completed_child, boundary


def config():
    cases = []
    for name, ordinal, pair in bundle.LAYOUT:
        child = isolated_config()
        child.update(device='cuda:0', cpu_masks={'cuda:0': bundle.MASKS[ordinal]},
                     gpu_uuid='GPU-12345678-1234-1234-1234-123456789ab'+str(ordinal),
                     frame_pairs=[pair], warmup_pairs=2, arm_timeout_seconds=900)
        cases.append(dict(id=name, device_ordinal=ordinal, gpu_uuid=child['gpu_uuid'], config=child))
    return bundle.normalize_config(dict(cases=cases))


def child_result(value, directory, *, diagnostic_capacity=False):
    arms = []
    for arm in isolated.ARMS:
        child = completed_child(capacity=diagnostic_capacity if arm['id']=='streams-current-decode' else True,
                                warmup_pairs=2)
        child['cpu_affinity'] = dict(bound_cpus=value['cpu_masks']['cuda:0'])
        child.update(schema=isolated.SCHEMA, configuration=isolated.child_config(value,arm,value['frame_pairs'][0]),
                     process=dict(pid=51000+len(arms),start_ticks=111),performance_evidence=False,
                     startup_cache_trims=1, baseline=boundary(),stream_ids=[1,1] if arm['mode']=='serial' else [1,2])
        child['baseline']['nvml']['uuid']=value['gpu_uuid']
        for round_ in child['rounds']:
            round_['stream_ids']=child['stream_ids'][:]
            round_['after_release']['nvml']['uuid']=value['gpu_uuid']
            for key in isolated.OUTPUTS:
                round_['measurement'][key]['sha256'] = hashlib.sha256(key.encode()).hexdigest()
            measured=round_['measurement'];index=round_['index']
            samples=[dict(timestamp_ns=index*10+offset,uuid=value['gpu_uuid'],memory_free_bytes=2*1024**3,
                          memory_total_bytes=24*1024**3,memory_used_bytes=22*1024**3) for offset in (1,2)]
            measured.update(mode=arm['mode'],repeat=index,purpose='lifecycle_warmup' if index<2 else 'lifecycle',
                a_decode_stream='b_stream' if arm['decode_a_on_b_stream'] else 'current_stream',
                started_monotonic_ns=index*10,finished_monotonic_ns=index*10+3,
                sample_interval_seconds=value['sample_interval_seconds'],nvml_samples=samples,nvml=dict(errors=[]))
            measured['memory_margin_gate']=isolated.overlap.memory_margin_gate(samples,[],index*10,index*10+3)
            if arm['id']=='streams-current-decode' and not diagnostic_capacity:
                measured['memory_margin_gate']['eligible']=False
        child['references']=[dict(role=role,frames=frames,**{kind:child['rounds'][0]['measurement'][role.lower()+'_'+kind]
                             for kind in ('latent','video')})
                             for role,frames in zip(('A','B'),value['frame_pairs'][0])]
        isolated._finish_child_result(child,child['configuration'])
        path=directory/arm['id']/'run.json';path.parent.mkdir(parents=True);atomic_json(path,child)
        arms.append(dict(id=arm['id'], frame_pair=value['frame_pairs'][0],
                         launch=dict(reaped=True, returncode=0, timed_out=False,pid=child['process']['pid'],
                                     result_path=str(path),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                         result=isolated._compact_child(child)))
    return dict(schema=isolated.SCHEMA, status='completed', configuration=value, arms=arms,
                qualification=isolated.aggregate(arms, value), performance_evidence=False)


class BundleTests(unittest.TestCase):
    def run_mock(self, *, failed_case=None, bad_json=None, timeout_case=None):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)/'bundle'
        events, launches = [], []
        class Process:
            def __init__(self, name, number):
                self.pid = 41000+number; self.name=name; self.returncode=None
            def poll(self):
                if self.name!=timeout_case or self.returncode is not None:
                    self.returncode = 1 if self.name==failed_case else (self.returncode if self.returncode is not None else 0)
                return self.returncode
            def wait(self, timeout=None):
                events.append(('wait',self.name)); self.poll(); return self.returncode
        def popen(command, **kwargs):
            directory = Path(command[-1]); name=directory.name
            value=json.loads((directory/'config.json').read_text())
            if name==bad_json:
                (directory/'run.json').write_text('{broken')
            else:
                atomic_json(directory/'run.json',child_result(value,directory))
            process=Process(name,len(launches)); launches.append((process,kwargs))
            events.append(('start',name)); return process
        original=bundle._start
        def start(case,directory):
            process,launch=original(case,directory)
            if case['id']==timeout_case: launch['started_monotonic']-=2701
            return process,launch
        def stop(process):
            events.append(('stop',process.name)); process.returncode=-15; process.wait();return True
        with patch.object(bundle.subprocess,'Popen',side_effect=popen), \
             patch.object(bundle,'_identity',side_effect=lambda pid:dict(pid=pid,pgid=pid,sid=pid,start_ticks=123)), \
             patch.object(bundle,'_group_members',return_value=[]),patch.object(bundle,'_start',side_effect=start), \
             patch.object(bundle,'_stop',side_effect=stop):
            result=bundle.run_bundle(config(),root)
        self.assertEqual(json.loads((root/'run.json').read_text()),result)
        return root,result,events,launches

    def test_fixed_waves_cpu_uuid_and_idempotent_config(self):
        value=config();self.assertEqual(bundle.normalize_config(value),value)
        for change in ('duplicate_uuid','wrong_cpu','wrong_pair','wrong_wave','short_lifecycle','timeout'):
            bad=copy.deepcopy(value)
            if change=='duplicate_uuid':
                for c in bad['cases']:
                    c['gpu_uuid']=bad['cases'][0]['gpu_uuid'];c['config']['gpu_uuid']=c['gpu_uuid']
            if change=='wrong_cpu':bad['cases'][0]['config']['cpu_masks']={'cuda:0':list(range(60,68))}
            if change=='wrong_pair':bad['cases'][0]['config']['frame_pairs']=[[17,17]]
            if change=='wrong_wave':bad['waves'].reverse()
            if change=='short_lifecycle':bad['cases'][0]['config']['lifecycle_pairs']=2
            if change=='timeout':bad['case_timeout_seconds']=None
            with self.subTest(change=change),self.assertRaises(ValueError):bundle.normalize_config(bad)

    def test_both_wave_cases_start_before_wait_and_have_one_owner(self):
        _,result,events,launches=self.run_mock()
        self.assertEqual(result['status'],'completed')
        self.assertTrue(bundle.validate_result(result)['passed'])
        self.assertEqual(events,[('start','a17-b33'),('start','a33-b17'),('wait','a17-b33'),('wait','a33-b17'),
                                 ('start','a33-b33'),('start','a17-b17'),('wait','a33-b33'),('wait','a17-b17')])
        self.assertTrue(all(kwargs['start_new_session'] for _,kwargs in launches))
        self.assertEqual([kwargs['env']['CUDA_VISIBLE_DEVICES'] for _,kwargs in launches],
                         [case['gpu_uuid'] for case in config()['cases']])
        self.assertTrue(all(r['launch']['container_owner_pid']==os.getpid() for r in result['cases']))
        self.assertFalse(result['performance_evidence'])
        for row in result['cases']:
            self.assertEqual(row['launch']['command'][:2], ['taskset', '-c'])
            payload=Path(row['launch']['result_path']).read_bytes()
            self.assertEqual(row['launch']['result_sha256'], hashlib.sha256(payload).hexdigest())
            self.assertTrue(row['required_scope']['passed'])
            diagnostic=row['result']['arms'][1]['result']['qualification']['scopes']['capacity']
            self.assertEqual(diagnostic['status'],'failed')

    def test_failed_case_or_bad_json_does_not_cancel_independent_cases(self):
        for controls in (dict(failed_case='a17-b33'),dict(bad_json='a17-b33'),dict(timeout_case='a17-b33')):
            with self.subTest(controls=controls):
                _,result,events,_=self.run_mock(**controls)
                checked=bundle.validate_result(result)
                self.assertEqual(len([x for x in events if x[0]=='start']),4)
                self.assertFalse(checked['passed'])
                self.assertTrue(checked['structure_valid'])
                self.assertFalse(checked['cases']['a17-b33']['passed'])
                self.assertTrue(all(checked['cases'][name]['passed'] for name in ['a33-b17','a33-b33','a17-b17']))
                self.assertEqual(result['status'],'qualification_failed')

    def test_hash_configuration_owner_and_reap_tampering_fail_closed(self):
        _,result,_,_=self.run_mock()
        for field,value in [('result_payload_sha256','0'*64),('configuration_payload_sha256','0'*64),
                            ('gpu_uuid','GPU-wrong'),('reaped',False),('cleanup_verified',False),
                            ('container_owner_pid',-1)]:
            bad=copy.deepcopy(result);bad['cases'][0]['launch'][field]=value
            with self.subTest(field=field):self.assertFalse(bundle.validate_result(bad)['passed'])
        bad=copy.deepcopy(result);bad['cases'][0]['result']['qualification']['passed']=False
        self.assertFalse(bundle.validate_result(bad)['passed'])

    def test_raw_required_scope_rejects_rebound_bad_evidence(self):
        _,result,_,_=self.run_mock()
        def rebind(row,arm,evidence):
            raw=evidence['result'];payload=json.dumps(raw,sort_keys=True).encode()
            evidence.update(payload_sha256=bundle.digest(raw),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
            arm['launch']['result_sha256']=evidence['sha256']
            arm['result']=isolated._compact_child(raw)
            row['result']['qualification']=isolated.aggregate(row['result']['arms'],row['configuration'])
            row['launch']['result_payload_sha256']=bundle.digest(row['result'])
        for change in ('unreleased','stream_changed','bad_reference','missing_hot_sample','wrong_hot_uuid',
                       'wrong_boundary_uuid','bad_warmup_index','growth','absent_raw','reused_window','sample_interval'):
            bad=copy.deepcopy(result);row=bad['cases'][0]
            arm=row['result']['arms'][0];evidence=row['launch']['required_arm_evidence']['serial'];raw=evidence['result']
            if change=='unreleased':raw['rounds'][0]['after_release']['state_count']=1
            if change=='stream_changed':raw['rounds'][4]['stream_ids']=[1,2]
            if change=='bad_reference':raw['references'][0]['latent']['sha256']='0'*64
            if change=='missing_hot_sample':raw['rounds'][1]['measurement']['nvml_samples'].pop()
            if change=='wrong_hot_uuid':raw['rounds'][1]['measurement']['nvml_samples'][0]['uuid']='GPU-wrong'
            if change=='wrong_boundary_uuid':raw['rounds'][1]['after_release']['nvml']['uuid']='GPU-wrong'
            if change=='bad_warmup_index':raw['rounds'][0]['measured_index']=0
            if change=='growth':raw['rounds'][4]['after_release']['reserved_bytes']+=65*1024**2
            if change=='reused_window':raw['rounds'][1]['measurement']['started_monotonic_ns']=0
            if change=='sample_interval':raw['rounds'][1]['measurement']['sample_interval_seconds']*=2
            rebind(row,arm,evidence)
            if change=='absent_raw':row['launch'].pop('required_arm_evidence')
            with self.subTest(change=change):
                checked=bundle.validate_result(bad)
                self.assertTrue(checked['structure_valid'])
                self.assertFalse(checked['cases']['a17-b33']['passed'])
                self.assertTrue(checked['cases']['a33-b17']['passed'])

    def test_process_group_timeout_kills_descendants_and_waits(self):
        process=Mock(pid=12345,returncode=-9)
        process.wait.side_effect=[subprocess.TimeoutExpired('case',10),-9]
        process.poll.side_effect=[None,-9]
        with patch.object(bundle.os,'killpg') as kill,patch.object(bundle,'_group_members',return_value=[]):
            self.assertTrue(bundle._stop(process))
        self.assertEqual([call.args for call in kill.call_args_list],[(12345,signal.SIGTERM),(12345,signal.SIGKILL)])
        self.assertEqual(process.wait.call_count,2)

    def test_unverified_cleanup_stops_before_next_wave(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(bundle, '_start', side_effect=bundle.CleanupError('still live')) as start:
            result=bundle.run_bundle(config(),Path(temporary)/'bundle')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(start.call_count, 1)
        self.assertEqual(len(result['waves']), 1)
        self.assertFalse(bundle.validate_result(result)['passed'])

    def test_existing_result_never_overwritten(self):
        root,_,_,_=self.run_mock()
        before=(root/'run.json').read_bytes()
        with self.assertRaises(FileExistsError):bundle.run_bundle(config(),root)
        self.assertEqual((root/'run.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
