"""Raw numerical evidence contracts; these tests make no CUDA execution claim."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
import external_tetriserve_service_numerics as numerics
import external_tetriserve_service as service
from external_sd35_round import FROZEN_WORKER_SHA256
from test_20261104_external_tetriserve_service import service_evidence


def plan():
    requests = service_evidence()[0]
    sources = {key:'0'*64 for key in numerics.SOURCE_KEYS}
    sources.update(shared_worker_sha256=FROZEN_WORKER_SHA256,
        service_adapter_sha256=service.sha(service.__file__),numerics_adapter_sha256=service.sha(numerics.__file__))
    return dict(schema=numerics.PLAN_SCHEMA,requests=requests,sources=sources,
        tolerance={'atol':1e-5,'rtol':1e-5},max_raw_bytes=numerics.MAX_RAW_BYTES,
        expected_requests=2,qualification_eligible=False)


def tensor_fixture(path,request,request_sha,plan_sha):
    # Real safetensors format fixture, explicitly synthetic zero tensors. The
    # production writer uses safetensors.torch.save with actual captured tensors.
    header = {'__metadata__':{'schema':numerics.CAPTURE_SCHEMA,'request_sha256':request_sha,'plan_sha256':plan_sha}}
    offset = 0
    for name,shape in numerics.raw_shapes(request).items():
        size = int(np.prod(shape))*2
        header[name] = dict(dtype='BF16',shape=shape,data_offsets=[offset,offset+size])
        offset += size
    raw = json.dumps(header,separators=(',',':')).encode()
    raw += b' '*((-len(raw))%8)
    path.write_bytes(struct.pack('<Q',len(raw))+raw+bytes(offset))
    return header


def captured_fixture(directory):
    root = Path(directory)
    (root/'raw').mkdir()
    planned = plan()
    plan_sha = numerics.canonical(planned)
    offered,dispatches,events,responses,exits = service_evidence()
    summary = dict(requests=offered,dispatches=dispatches,responses=responses,worker_exitcodes=exits)
    for row,response in zip(offered,responses):
        rid,request = row['request_id'],row['request']
        path = root/'raw'/f'request-{rid}.safetensors'
        request_sha = numerics.canonical(request)
        tensor_fixture(path,request,request_sha,plan_sha)
        tensors = numerics.inspect_safetensors(path,request,request_sha,plan_sha)
        final = next(d for d in dispatches if d['request_ids']==[response['native_request_id']] and d['end']==28)
        record = dict(schema=numerics.CAPTURE_SCHEMA,plan_sha256=plan_sha,request_id=rid,
            request=request,request_sha256=request_sha,native_request_id=response['native_request_id'],
            sub_request_id=final['sub_request_ids'][0],owner_rank=final['ranks'][0],final_ranks=final['ranks'],
            final_interval=[final['start'],28],actual_final_sync_calls=1,actual_raw_decode_calls=1,
            capture_before_postprocess=True,tensor_file=dict(file=path.name,sha256=service.sha(path),
                bytes=path.stat().st_size,tensors=tensors))
        path.with_suffix('.capture.json').write_text(json.dumps(record))
    return planned,plan_sha,summary


class TetriServiceNumericsTests(unittest.TestCase):
    def test_plan_binds_fixed_tolerance_independent_requests_and_formal_shapes(self):
        numerics.validate_plan(plan())
        for kind in ('atol','rtol','denominator','seed','prompt','backend','steps','shape'):
            value = deepcopy(plan())
            if kind=='atol': value['tolerance']['atol']=.01
            if kind=='rtol': value['tolerance']['rtol']=.01
            if kind=='denominator': value['expected_requests']=1
            if kind=='seed': value['requests'][1]['request']['seed']=101
            if kind=='prompt': value['requests'][1]['request']['prompt']=value['requests'][0]['request']['prompt']
            if kind=='backend': value['sources']['shared_worker_sha256']='f'*64
            if kind=='steps': value['requests'][0]['request']['steps']=27
            if kind=='shape': value['requests'][0]['request']['height']=256
            with self.subTest(kind=kind),self.assertRaises(ValueError): numerics.validate_plan(value)

    def test_reference_relative_metrics_compare_all_raw_elements(self):
        reference = np.asarray([0.,1.,-2.,3.],dtype=np.float32)
        result = numerics.compare_arrays(reference,reference.copy())
        self.assertTrue(result['passed'])
        self.assertTrue(result['exactly_equal'])
        self.assertEqual(result['elements'],4)
        self.assertEqual(result['max_abs_error'],0.)
        near = reference.copy(); near[0]=.000001
        self.assertTrue(numerics.compare_arrays(reference,near)['passed'])
        far = reference.copy(); far[-1]+=.001
        mismatch = numerics.compare_arrays(reference,far)
        self.assertFalse(mismatch['passed'])
        self.assertGreater(mismatch['max_abs_error'],.0009)
        self.assertGreater(mismatch['rmse'],0.)
        for actual in (np.asarray([np.nan,1.,2.,3.],dtype=np.float32),reference[:3],reference.astype(np.float64)):
            with self.assertRaises(ValueError): numerics.compare_arrays(reference,actual)

    def test_original_final_round_identity_includes_seed_prompt_absolute_cursor(self):
        request = plan()['requests'][0]['request']
        config = SimpleNamespace(**{k:v for k,v in request.items() if k!='steps'},
                                 batch_size=1,num_inference_steps=8)
        config.prompt = [request['prompt']]
        req = SimpleNamespace(req_ids=[42],sub_req_ids=[3],parallel_ranks=[0,1],origin_steps=[28],
                              start_step=20,end_step=28,is_last_steps=[True],input_config=config)
        row = numerics.scheduled_identity(req)
        self.assertEqual(row['request'],request)
        self.assertEqual(row['final_interval'],[20,28])
        for attr,value in (('end_step',27),('origin_steps',[27]),('is_last_steps',[False]),('req_ids',[42,43])):
            changed = deepcopy(req); setattr(changed,attr,value)
            with self.assertRaises(ValueError): numerics.scheduled_identity(changed)

    def test_previously_executed_native_scheduler_final_chunks(self):
        default = ROOT.parent/'.local/ditserve-kairos-rewrite-runtime/evidence/d20261103-tetri-logic-19c47ea-r2/summary.json'
        path = Path(os.environ.get('DITSERVE_TETRI_LOGIC_SUMMARY',str(default)))
        if not path.is_file():
            self.skipTest('previously executed native scheduler evidence is not available in this checkout')
        evidence = json.loads(path.read_bytes())['execution']['child_summary']['scheduler']['evidence']
        self.assertEqual(len(evidence['records']),2)
        for record in evidence['records']:
            for final in record['rounds'][-1]:
                self.assertEqual(final['origin_steps'],[28])
                self.assertEqual((final['start'],final['end'],final['steps']),(20,28,8))
                request = plan()['requests'][0]['request']
                config = SimpleNamespace(**{k:v for k,v in request.items() if k!='steps'},
                    batch_size=final['batch'],num_inference_steps=final['steps'])
                config.prompt=final['prompts']
                req = SimpleNamespace(req_ids=[42],sub_req_ids=final['sub_req_ids'],parallel_ranks=final['ranks'],
                    origin_steps=final['origin_steps'],start_step=final['start'],end_step=final['end'],
                    is_last_steps=final['final'],input_config=config)
                self.assertEqual(numerics.scheduled_identity(req)['final_interval'],[20,28])

    def test_actual_safetensors_header_bytes_and_raw_tensor_digests(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'raw.safetensors'
            request = plan()['requests'][0]['request']
            request_sha = numerics.canonical(request)
            tensor_fixture(path,request,request_sha,'a'*64)
            record = numerics.inspect_safetensors(path,request,request_sha,'a'*64)
            self.assertEqual(record['latents']['shape'],[1,16,64,64])
            self.assertEqual(record['decoded']['shape'],[1,3,512,512])
            self.assertEqual(record['latents']['tensor_sha256'],hashlib.sha256(bytes(131072)).hexdigest())
            with self.assertRaises(ValueError): numerics.inspect_safetensors(path,request,'b'*64,'a'*64)
            with path.open('ab') as stream: stream.write(b'appended')
            with self.assertRaises(ValueError): numerics.inspect_safetensors(path,request,request_sha,'a'*64)

    def test_malformed_header_duplicate_keys_and_huge_lengths_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'raw.safetensors'
            request = plan()['requests'][0]['request']
            for raw in (struct.pack('<Q',2**50),struct.pack('<Q',13)+b'{"x":1,"x":2}',b'bad'):
                path.write_bytes(raw)
                with self.assertRaises((ValueError,KeyError)):
                    numerics.inspect_safetensors(path,request,'a'*64,'b'*64)
        with self.assertRaises(ValueError): numerics.read_json('{"x": 1, "x": 2}')

    def test_captures_close_to_real_http_final_owner_and_reject_mutations(self):
        with TemporaryDirectory() as directory:
            planned,plan_sha,summary = captured_fixture(directory)
            captures = numerics.bound_captures(directory,planned,plan_sha,summary)
            self.assertEqual(sorted(captures),[100,101])
            path = Path(directory)/'raw/request-100.capture.json'
            original = path.read_bytes()
            for kind in ('seed','native_id','owner','cursor','decode','postprocess','sha','tensor_sha'):
                record = json.loads(original)
                if kind=='seed': record['request']['seed']=999
                if kind=='native_id': record['native_request_id']=999
                if kind=='owner': record['owner_rank']=0
                if kind=='cursor': record['final_interval']=[0,28]
                if kind=='decode': record['actual_raw_decode_calls']=0
                if kind=='postprocess': record['capture_before_postprocess']=False
                if kind=='sha': record['tensor_file']['sha256']='f'*64
                if kind=='tensor_sha': record['tensor_file']['tensors']['latents']['tensor_sha256']='e'*64
                path.write_text(json.dumps(record))
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    numerics.bound_captures(directory,planned,plan_sha,summary)
            path.write_bytes(original)
            (Path(directory)/'raw/request-101.capture.json').unlink()
            with self.assertRaises(ValueError): numerics.bound_captures(directory,planned,plan_sha,summary)

    def test_service_exit_and_full_rank_evidence_are_required_before_reference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); events_dir=root/'service-events'; events_dir.mkdir()
            offered,dispatches,events,responses,exits = service_evidence()
            planned = plan()
            for rank in (0,1):
                (events_dir/f'rank-{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events if r['rank']==rank))
            summary = dict(status='service_passed',serving_executed=True,http_zmq_executed=True,
                model_weights_verified=True,requests=offered,dispatches=dispatches,responses=responses,worker_exitcodes=exits,
                service_events_sha256={p.name:service.sha(p) for p in events_dir.iterdir()},
                **planned['sources'])
            path=root/'service-completion.json'; path.write_text(json.dumps(summary))
            numerics.closed_service(path,service.sha(path),planned)
            summary['worker_exitcodes']=[0,None]; path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError): numerics.closed_service(path,service.sha(path),planned)
            summary['worker_exitcodes']=[0,0]; path.write_text(json.dumps(summary))
            (events_dir/'rank-1.jsonl').write_text('')
            with self.assertRaises(ValueError): numerics.closed_service(path,service.sha(path),planned)


if __name__=='__main__': unittest.main()
