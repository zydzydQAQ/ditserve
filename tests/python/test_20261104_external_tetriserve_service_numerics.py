"""Raw numerical evidence contracts; these tests make no CUDA execution claim."""
from copy import deepcopy
import ast
import asyncio
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


def official_http_ids():
    # Execute the unmodified official counter definitions, not a substitute
    # generator. Extract just these CPU definitions to avoid importing Torch.
    source = ROOT.parent/'.local/ditserve-kairos-rewrite-runtime/external-systems/20261103/TetriServe/tetriserve/utils.py'
    if not source.is_file():
        raise unittest.SkipTest('frozen official TetriServe source is not available in this checkout')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '815b8aea9ffb88df066fbd6fe3ae58e910c06ec6c4f7026e4727281e5bee95f6':
        raise ValueError('official HTTP counter source changed')
    wanted = {'Counter','gen_req_id','gen_sub_req_id','get_gen_req_id','get_gen_sub_req_id'}
    nodes = [n for n in ast.parse(raw).body if
        (isinstance(n,(ast.ClassDef,ast.AsyncFunctionDef)) and n.name in wanted) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets))]
    if len(nodes) != 5: raise ValueError('official counter definitions changed')
    namespace = {'asyncio':asyncio}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    async def allocate():
        return [(await namespace['get_gen_req_id'](),await namespace['get_gen_sub_req_id']()) for _ in range(2)]
    return asyncio.run(allocate())


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

    def test_real_http_counter_negative_subrequests_reach_final_capture(self):
        ids = official_http_ids()
        self.assertEqual(ids,[(0,-1),(1,-2)])
        protocol = json.loads((ROOT/'configs/benchmarks/formal-mixed-v1.json').read_bytes())
        offered = service.request_plans(protocol,
            ['A red fox standing beside a mountain lake.','A white rabbit in a field of blue flowers.'],101,True)
        # Reproduce both final dispatch identities observed in GPU r1. Native
        # HTTP ordering put the 1024 request first: (0,-1,26:28,rank1), followed
        # by the 512 request: (1,-2,20:28,rank0). The final tensor execution had
        # not occurred when the old observer incorrectly rejected these IDs.
        for (parent,sub),index,start,rank in ((ids[0],1,26,1),(ids[1],0,20,0)):
            request = offered[index]['request']
            config = SimpleNamespace(**{k:v for k,v in request.items() if k!='steps'},
                batch_size=1,num_inference_steps=28-start)
            config.prompt = [request['prompt']]
            req = SimpleNamespace(req_ids=[parent],sub_req_ids=[sub],parallel_ranks=[rank],origin_steps=[28],
                start_step=start,end_step=28,is_last_steps=[True],input_config=config)
            row = numerics.scheduled_identity(req)
            self.assertEqual((row['native_request_id'],row['sub_request_id']),(parent,sub))
            self.assertEqual(row['request'],request)
            self.assertEqual(row['final_interval'],[start,28])
            for field,bad in [('req_ids',[-1]),('req_ids',[True]),('req_ids',[1.0]),
                              ('sub_req_ids',[True]),('sub_req_ids',[-1.0]),('sub_req_ids',['-1'])]:
                altered = deepcopy(req);setattr(altered,field,bad)
                with self.subTest(field=field,bad=bad),self.assertRaises(ValueError):
                    numerics.scheduled_identity(altered)

    def test_signed_subrequests_remain_exactly_bound_and_unique_through_raw_receipts(self):
        with TemporaryDirectory() as directory:
            planned,plan_sha,summary = captured_fixture(directory)
            ids = official_http_ids()
            for response,(parent,sub) in zip(summary['responses'],ids):
                old_parent = response['native_request_id'];response['native_request_id']=parent
                for dispatch in summary['dispatches']:
                    if dispatch['request_ids']==[old_parent]:
                        dispatch['request_ids']=[parent];dispatch['sub_request_ids']=[sub]
                path=Path(directory)/'raw'/('request-'+str(response['request_id'])+'.capture.json')
                row=json.loads(path.read_bytes());row.update(native_request_id=parent,sub_request_id=sub)
                path.write_text(json.dumps(row))
            captured=numerics.bound_captures(directory,planned,plan_sha,summary)
            self.assertEqual([(r['native_request_id'],r['sub_request_id']) for r in captured.values()],ids)
            path=Path(directory)/'raw/request-100.capture.json';original=path.read_bytes()
            for bad in (1,-2,True,-1.0,'-1'):
                row=json.loads(original);row['sub_request_id']=bad;path.write_text(json.dumps(row))
                with self.subTest(sub_id=bad),self.assertRaises(ValueError):
                    numerics.bound_captures(directory,planned,plan_sha,summary)
            path.write_bytes(original)
            # Even if a poisoned second dispatch and capture agree, reused
            # global sub-request identity cannot qualify a distinct HTTP output.
            duplicate=deepcopy(summary)
            for dispatch in duplicate['dispatches']:
                if dispatch['request_ids']==[ids[1][0]]:dispatch['sub_request_ids']=[ids[0][1]]
            second=Path(directory)/'raw/request-101.capture.json'
            row=json.loads(second.read_bytes());row['sub_request_id']=ids[0][1];second.write_text(json.dumps(row))
            with self.assertRaises(ValueError):numerics.bound_captures(directory,planned,plan_sha,duplicate)

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
