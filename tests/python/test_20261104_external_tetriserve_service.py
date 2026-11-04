"""Fail closed when native service/calibration evidence is incomplete or altered."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks'))
import external_tetriserve_service as service


def requests(mixed=False):
    protocol = json.loads((ROOT/'configs/benchmarks/formal-mixed-v1.json').read_bytes())
    return service.request_plans(protocol, ['red fox on snow', 'blue boat on a lake'], 101, mixed)


def terminal_events():
    return [dict(rank=r,event=event,sender_threads_joined=True) for r in (0,1)
            for event in ('native_serve_returned','distributed_closed')]


def service_evidence():
    offered = requests()
    dispatches, events, responses = [], [], []
    for index, row in enumerate(offered):
        request = row['request']
        # Deliberately reverse HTTP-assigned IDs. Order is not a binding.
        native_id, sid = 42-index, index+1
        for start, end, ranks in ((0,10,[0]),(10,20,[0,1]),(20,28,[1])):
            dispatch = dict(request_ids=[native_id],sub_request_ids=[sid],ranks=ranks,
                start=start,end=end,height=512,width=512,batch_size=1,
                inputs=dict(prompt=[request['prompt']],negative_prompt=request['negative_prompt'],
                    seed=request['seed'],guidance_scale=request['guidance_scale'],
                    max_sequence_length=request['max_sequence_length'],origin_steps=[28]))
            dispatches.append(dispatch)
            events.extend(dict(rank=rank,event='round',**deepcopy(dispatch),transformer_calls=end-start,
                output_kinds=['image' if end==28 else 'latent_future'],continuation_count=int(start>0)) for rank in ranks)
        responses.append(dict(request_id=row['request_id'],native_request_id=native_id,http_status=200,
            image_count=1,size=[512,512],mode='RGB',pixels_sha256='a'*64,elapsed_seconds=4.0))
    events.extend(terminal_events())
    return offered, dispatches, events, responses, [0,0]


def calibration_evidence():
    cells = service.calibration_cells(requests())
    events = []
    for index, cell in enumerate(cells):
        for rank in range(cell['sp']):
            events.append(dict(event='round',rank=rank,request_ids=[1000+index],ranks=list(range(cell['sp'])),
                height=cell['height'],width=cell['width'],batch_size=cell['bs'],start=0,end=28,
                transformer_calls=28,output_kinds=['image']*cell['bs'],
                prologue_seconds=[.2 + rank*.1]*cell['bs'],denoising_seconds=1.5+rank*.1))
    return cells, events + terminal_events()


class TetriServiceTests(unittest.TestCase):
    def test_formal_shapes_and_real_profile_denominator(self):
        low, mixed = requests(), requests(True)
        self.assertEqual([r['request']['height'] for r in low],[512,512])
        self.assertEqual([r['request']['height'] for r in mixed],[512,1024])
        self.assertEqual([r['request']['seed'] for r in low],[101,102])
        self.assertTrue(all(r['request']['steps']==28 and r['request']['guidance_scale']==4.5 for r in mixed))
        self.assertEqual(len(service.calibration_cells(low)),4)
        self.assertEqual(len(service.calibration_cells(mixed)),8)
        self.assertEqual(sum(c['sp'] for c in service.calibration_cells(low)),6)
        bad = json.loads((ROOT/'configs/benchmarks/formal-mixed-v1.json').read_bytes())
        bad['shapes']['sd35-512']['steps'] = 27
        with self.assertRaises(ValueError): service.request_plans(bad,['a','b'],101,False)
        with self.assertRaises(ValueError): service.request_plans(bad,['a','a'],101,False)

    def test_cost_table_uses_actual_slowest_rank_and_every_native_batch_form(self):
        cells, events = calibration_evidence()
        rows = service.measured_profile(cells,events)
        self.assertEqual([(r['bs'],r['ulysses_degree']) for r in rows],[(1,1),(1,2),(2,1),(2,2)])
        self.assertAlmostEqual(rows[-1]['avg_text_encoding_time'],.6)
        self.assertAlmostEqual(rows[-1]['avg_denoising_time'],1.6)
        self.assertEqual(rows[-1]['steps'],28)
        service.validate_exits(events)
        for kind in ('missing_rank','duplicate','zero','nan','bool','wrong_steps','wrong_shape','wrong_batch','wrong_output'):
            value = deepcopy(events)
            if kind == 'missing_rank': value.pop(2)
            if kind == 'duplicate': value.insert(0,deepcopy(value[0]))
            if kind == 'zero': value[0]['denoising_seconds']=0
            if kind == 'nan': value[0]['denoising_seconds']=float('nan')
            if kind == 'bool': value[0]['denoising_seconds']=True
            if kind == 'wrong_steps': value[0]['transformer_calls']=27
            if kind == 'wrong_shape': value[0]['height']=1024
            if kind == 'wrong_batch': value[0]['batch_size']=2
            if kind == 'wrong_output': value[0]['output_kinds']=['latent_future']
            with self.subTest(kind=kind), self.assertRaises(ValueError): service.measured_profile(cells,value)

    def test_original_dispatch_all_rank_execution_and_http_identity_close(self):
        result = service.assess_service(*service_evidence())
        self.assertEqual(result['requests_completed'],2)
        self.assertEqual(result['rank_executions'],8)
        self.assertEqual(result['native_rounds'],6)
        self.assertEqual(result['continuation_rounds'],4)
        self.assertEqual(result['transformer_steps_per_request'],[28,28])

    def test_incomplete_replayed_mutated_or_failed_service_never_passes(self):
        for kind in ('missing_rank','repeat','cursor','steps','seed','prompt','cfg','future','http_count','http_size',
                     'http_id','worker_exit','cleanup','sender_thread','worker_error','wrong_dispatch'):
            offered, dispatches, events, responses, exits = deepcopy(service_evidence())
            if kind == 'missing_rank': events.pop(2)
            if kind == 'repeat': events.insert(0,deepcopy(events[0]))
            if kind == 'cursor': dispatches[1]['start']=0
            if kind == 'steps': events[0]['transformer_calls']=9
            if kind == 'seed': events[0]['inputs']['seed']=999
            if kind == 'prompt': dispatches[0]['inputs']['prompt']=['different']
            if kind == 'cfg': dispatches[0]['inputs']['guidance_scale']=1.
            if kind == 'future': events[1]['continuation_count']=0
            if kind == 'http_count': responses[0]['image_count']=0
            if kind == 'http_size': responses[0]['size']=[1024,1024]
            if kind == 'http_id': responses[1]['native_request_id']=responses[0]['native_request_id']
            if kind == 'worker_exit': exits[1]=1
            if kind == 'cleanup': events.pop()
            if kind == 'sender_thread': events[-4]['sender_threads_joined']=False
            if kind == 'worker_error': events.append(dict(event='worker_failed',rank=0))
            if kind == 'wrong_dispatch': events[0]['ranks']=[0,1]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                service.assess_service(offered,dispatches,events,responses,exits)

    def test_storage_uses_current_800_gib_or_80_percent_boundary(self):
        for used,percent,allowed in ((800,79,True),(799,80,False),(801,79,False)):
            process = SimpleNamespace(returncode=0,stdout='Filesystem 1G-blocks Used Available Use% Mounted\nx 1024G '+str(used)+'G 200G '+str(percent)+'% /shared\n')
            with patch.object(service.subprocess,'run',return_value=process):
                if allowed:
                    self.assertEqual(service.check_storage()['used_gib'],800)
                else:
                    with self.assertRaises(RuntimeError): service.check_storage()

    def test_cli_help_is_model_free_and_names_real_service(self):
        proc = subprocess.run([sys.executable,'-B',str(ROOT/'benchmarks/external_tetriserve_service.py'),'--help'],
                              capture_output=True,text=True,timeout=15)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertIn('--mixed-resolution',proc.stdout)
        self.assertIn('original TetriServe HTTP/service/worker loop',' '.join(proc.stdout.split()))

    def test_actual_round_module_rebinds_to_identical_derived_file_and_rejects_tamper(self):
        # Real round adapter import (no Torch import/model construction). This
        # catches the prior planning-path vs derived-path integration failure.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = (ROOT/'benchmarks/external_sd35_round.py').read_bytes()
            (root/'external_sd35_round.py').write_bytes(raw)
            manifest = {'files':{'external_sd35_round.py':service.sha(ROOT/'benchmarks/external_sd35_round.py')}}
            (root/'PORT_MANIFEST.json').write_text(json.dumps(manifest))
            code = '''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import external_tetriserve_service as service
import external_sd35_round as planning
bound=service.bind_derived_adapters({'root':sys.argv[2]})
assert Path(bound.__file__).resolve()==Path(sys.argv[2])/'external_sd35_round.py'
assert bound is not planning
print('actual-derived-import-passed')
'''
            command = [sys.executable,'-B','-c',code,str(ROOT/'benchmarks'),str(root)]
            proc = subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertIn('actual-derived-import-passed',proc.stdout)
            (root/'external_sd35_round.py').write_bytes(raw+b'\n# changed\n')
            proc = subprocess.run(command,capture_output=True,text=True,timeout=15)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn('derived execution adapter bytes differ',proc.stderr)


if __name__ == '__main__':
    unittest.main()
