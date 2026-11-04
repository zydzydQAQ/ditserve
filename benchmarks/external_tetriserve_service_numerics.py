"""Raw-tensor capture and independent reference for original TetriServe service.

Not connected to the service entry point until an explicit versioned integration.
Capture wrappers return the original results unchanged. References run only after
both original workers have exited, using the normal shared pipeline for 28 steps.
Raw service and reference tensors remain protected; this module never deletes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import threading
import traceback

import external_tetriserve_service as service

ATOL = 1e-5
RTOL = 1e-5
MAX_RAW_BYTES = 64 * 1024 * 1024
PLAN_SCHEMA = 'ditserve.tetriserve-service-numerics-plan.v1'
CAPTURE_SCHEMA = 'ditserve.tetriserve-service-raw-capture.v1'
RESULT_SCHEMA = 'ditserve.tetriserve-service-raw-comparison.v1'
SOURCE_KEYS = ('source_manifest_sha256','derived_manifest_sha256','model_manifest_sha256',
               'protocol_sha256','shared_worker_sha256','service_adapter_sha256','numerics_adapter_sha256')
SHA = re.compile('[0-9a-f]{64}')


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read_json(raw):
    def unique(pairs):
        result = {}
        for name,value in pairs:
            if name in result:
                raise ValueError('duplicate JSON field: '+name)
            result[name] = value
        return result
    return json.loads(raw,object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def check_sha(value):
    if not isinstance(value,str) or SHA.fullmatch(value) is None:
        raise ValueError('invalid SHA256 identity')


def validate_plan(plan):
    from external_sd35_round import ImageRequest, FROZEN_WORKER_SHA256
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('tolerance') != {'atol':ATOL,'rtol':RTOL}
            or plan.get('max_raw_bytes') != MAX_RAW_BYTES or plan.get('expected_requests') != 2
            or plan.get('qualification_eligible') is not False):
        raise ValueError('numerical protocol/tolerance/denominator differs')
    sources = plan.get('sources',{})
    if set(sources) != set(SOURCE_KEYS):
        raise ValueError('complete numerical source binding required')
    for value in sources.values(): check_sha(value)
    if sources['shared_worker_sha256'] != FROZEN_WORKER_SHA256:
        raise ValueError('shared backend is not frozen 51740c0')
    rows = plan.get('requests',[])
    if len(rows) != 2 or len({r['request_id'] for r in rows}) != 2:
        raise ValueError('two distinct request IDs required')
    if len({r['request']['seed'] for r in rows}) != 2 or len({r['request']['prompt'] for r in rows}) != 2:
        raise ValueError('two independent prompts and seeds required')
    for row in rows:
        if type(row['request_id']) is not int or row['request_id'] < 0:
            raise ValueError('request IDs must be nonnegative integers')
        request = ImageRequest(**row['request'])
        size = {'sd35-512':512,'sd35-1024':1024}.get(row.get('shape_id'))
        if (size is None or request.height != size or request.width != size or request.steps != 28
                or request.guidance_scale != 4.5 or request.max_sequence_length != 256
                or request.negative_prompt != 'blurry, low quality'):
            raise ValueError('request differs from frozen formal image profile')
    return plan


def create_plan(root, requests, sources, protocol_path):
    """Called before service execution; never infer tolerances from observations."""
    from external_sd35_round_check import request_plan
    root = Path(root).resolve()
    if not root.is_relative_to(Path('/shared/output/DITSERVE')):
        raise ValueError('numerical artifacts must stay in project shared output')
    if service.sha(protocol_path) != sources['protocol_sha256']:
        raise ValueError('protocol bytes differ')
    protocol = read_json(Path(protocol_path).read_bytes())
    for row in requests:
        r = row['request']
        if vars(request_plan(protocol,r['prompt'],r['seed'],row['shape_id'])) != r:
            raise ValueError('offered request does not match bound formal protocol')
    if sources['numerics_adapter_sha256'] != service.sha(__file__):
        raise ValueError('numerics implementation differs')
    plan = validate_plan(dict(schema=PLAN_SCHEMA,requests=requests,sources=sources,
        tolerance={'atol':ATOL,'rtol':RTOL},max_raw_bytes=MAX_RAW_BYTES,expected_requests=2,
        qualification_eligible=False,reference='Independent normal shared SP1 pipeline, full 28 steps',
        raw_retention='Protected through independent comparison/review; no automatic deletion'))
    service.check_storage()
    root.mkdir(parents=True,exist_ok=False)
    (root/'raw').mkdir()
    (root/'reference').mkdir()
    service.write_json(root/'numerics-plan.json',plan)
    service.write_json(root/'PROTECTED.json',dict(schema='ditserve.protected-numerical-work.v1',
        plan_sha256=service.sha(root/'numerics-plan.json'),dependencies=['raw service/reference numerical comparison','independent acceptance'],
        automatic_cleanup_allowed=False))
    return root/'numerics-plan.json',service.sha(root/'numerics-plan.json')


def load_plan(path, expected_sha):
    path = Path(path).absolute()
    check_sha(expected_sha)
    if path.resolve(strict=True) != path or service.sha(path) != expected_sha:
        raise ValueError('numerical plan identity differs')
    plan = validate_plan(read_json(path.read_bytes()))
    if plan['sources']['numerics_adapter_sha256'] != service.sha(__file__):
        raise ValueError('executing numerical implementation differs from plan')
    return plan


def raw_shapes(request):
    h,w = request['height'],request['width']
    return {'latents':[1,16,h//8,w//8], 'decoded':[1,3,h,w]}


def scheduled_identity(reqs):
    """Capture scientific identity at the original worker's final round."""
    if (len(reqs.req_ids) != 1 or len(reqs.sub_req_ids) != 1 or reqs.input_config.batch_size != 1
            or reqs.origin_steps != [28] or reqs.end_step != 28 or reqs.is_last_steps != [True]
            or type(reqs.start_step) is not int or not 0 <= reqs.start_step < 28
            or reqs.input_config.num_inference_steps != reqs.end_step-reqs.start_step
            or reqs.parallel_ranks not in ([0],[1],[0,1])
            or not isinstance(reqs.input_config.prompt,list) or len(reqs.input_config.prompt) != 1):
        raise ValueError('capture requires one actual final 28-step service output')
    for identifier in [*reqs.req_ids,*reqs.sub_req_ids]:
        if type(identifier) is not int or identifier < 0:
            raise ValueError('native request IDs must be nonnegative integers')
    config = reqs.input_config
    request = dict(prompt=config.prompt[0],negative_prompt=config.negative_prompt,
        height=config.height,width=config.width,steps=28,seed=config.seed,
        guidance_scale=config.guidance_scale,max_sequence_length=config.max_sequence_length)
    return dict(native_request_id=reqs.req_ids[0],sub_request_id=reqs.sub_req_ids[0],
                final_ranks=list(reqs.parallel_ranks),final_interval=[reqs.start_step,28],request=request)


def inspect_safetensors(path, request, request_sha, plan_sha):
    """Bounded, non-executing header validation before loading tensor payloads."""
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file() or path.stat().st_size > MAX_RAW_BYTES:
        raise ValueError('raw tensor file is missing, linked or oversized')
    with path.open('rb') as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError('missing safetensors header')
        length, = struct.unpack('<Q',length_bytes)
        if not 2 <= length <= 1024*1024:
            raise ValueError('unbounded safetensors header')
        header = read_json(stream.read(length))
    if set(header) != {'__metadata__','latents','decoded'}:
        raise ValueError('only complete latent and raw decoded tensors are allowed')
    if header['__metadata__'] != {'schema':CAPTURE_SCHEMA,'request_sha256':request_sha,'plan_sha256':plan_sha}:
        raise ValueError('raw tensor metadata identity differs')
    descriptors,offsets = {},[]
    for name,shape in raw_shapes(request).items():
        row = header[name]
        dtype = row.get('dtype')
        if row.get('shape') != shape or dtype not in ('BF16','F32'):
            raise ValueError('raw tensor shape/dtype differs')
        start,end = row.get('data_offsets',[None,None])
        size = math.prod(shape)*{'BF16':2,'F32':4}[dtype]
        if type(start) is not int or type(end) is not int or start < 0 or end-start != size:
            raise ValueError('raw tensor byte denominator differs')
        offsets.append((start,end))
        descriptors[name] = dict(shape=shape,dtype=dtype,bytes=size)
    cursor = 0
    for start,end in sorted(offsets):
        if start != cursor:
            raise ValueError('raw tensor offsets overlap or leave gaps')
        cursor = end
    if path.stat().st_size != 8+length+cursor:
        raise ValueError('truncated or appended raw tensor payload')
    with path.open('rb') as stream:
        for name in ('latents','decoded'):
            start,end = header[name]['data_offsets']
            stream.seek(8+length+start)
            digest = hashlib.sha256()
            remaining = end-start
            while remaining:
                chunk = stream.read(min(1024*1024,remaining))
                if not chunk:
                    raise ValueError('raw tensor changed during inspection')
                digest.update(chunk)
                remaining -= len(chunk)
            descriptors[name]['tensor_sha256'] = digest.hexdigest()
    return descriptors


def save_raw(path, tensors, request, request_sha, plan_sha):
    import torch
    from safetensors.torch import save
    if set(tensors) != {'latents','decoded'}:
        raise ValueError('both actual latent and pre-postprocess decode are required')
    cpu = {}
    for name,shape in raw_shapes(request).items():
        tensor = tensors[name]
        if (not isinstance(tensor,torch.Tensor) or list(tensor.shape) != shape
                or tensor.dtype not in (torch.bfloat16,torch.float32) or not torch.isfinite(tensor).all().item()):
            raise ValueError('actual finite full raw tensor required: '+name)
        cpu[name] = tensor.detach().to('cpu',copy=True).contiguous()
    raw = save(cpu,metadata={'schema':CAPTURE_SCHEMA,'request_sha256':request_sha,'plan_sha256':plan_sha})
    path = Path(path)
    if path.absolute() != path.resolve():
        raise ValueError('raw output path cannot traverse symlinks or aliases')
    current = sum(p.stat().st_size for p in path.parent.parent.rglob('*.safetensors'))
    if current+len(raw) > MAX_RAW_BYTES:
        raise ValueError('protected raw tensor budget exceeded')
    service.check_storage()
    with path.open('xb') as stream:
        stream.write(raw)
    descriptors = inspect_safetensors(path,request,request_sha,plan_sha)
    return dict(file=path.name,sha256=service.sha(path),bytes=path.stat().st_size,tensors=descriptors)


class CaptureSession:
    """Use around the original Runner.generate; disabled for calibration."""
    def __init__(self, plan_path, plan_sha, rank):
        self.plan_path,self.plan_sha = Path(plan_path).resolve(strict=True),plan_sha
        self.plan = load_plan(self.plan_path,plan_sha)
        if type(rank) is not int or rank not in (0,1):
            raise ValueError('capture requires a real allocated worker rank')
        self.rank = rank
        self.busy = False

    @contextmanager
    def observe_round(self, pipeline, reqs):
        if reqs.is_last_steps != [True] or self.rank != reqs.parallel_ranks[0]:
            yield
            return
        if self.busy:
            raise RuntimeError('overlapping capture on one model instance')
        identity = scheduled_identity(reqs)
        request_sha = canonical(identity['request'])
        matches = [r for r in self.plan['requests'] if canonical(r['request']) == request_sha]
        if len(matches) != 1:
            raise ValueError('actual final round is not a uniquely planned request')
        identity.update(request_id=matches[0]['request_id'],owner_rank=self.rank,request_sha256=request_sha)
        payload,counts = {},{'sync':0,'decode':0}
        original_sync = pipeline.pipe._sync_pipeline
        original_decode = pipeline.raw.vae.decode
        import torch

        def sync(*args,**kwargs):
            value = original_sync(*args,**kwargs)
            counts['sync'] += 1
            if not isinstance(value,torch.Tensor) or not value.is_cuda or not torch.isfinite(value).all().item():
                raise ValueError('actual final CUDA latent is not finite')
            payload['latents'] = value.detach().to('cpu',copy=True).contiguous()
            return value

        def decode(*args,**kwargs):
            output = original_decode(*args,**kwargs)
            counts['decode'] += 1
            if not isinstance(output,tuple) or len(output) != 1:
                raise ValueError('actual VAE return_dict=False output required')
            value = output[0]
            if not isinstance(value,torch.Tensor) or not value.is_cuda or not torch.isfinite(value).all().item():
                raise ValueError('actual pre-postprocess CUDA decode is not finite')
            payload['decoded'] = value.detach().to('cpu',copy=True).contiguous()
            return output

        self.busy = True
        pipeline.pipe._sync_pipeline,pipeline.raw.vae.decode = sync,decode
        try:
            yield
            if counts != {'sync':1,'decode':1}:
                raise ValueError('exactly one actual final sync/decode must be captured')
            stem = 'request-'+str(identity['request_id'])
            path = self.plan_path.parent/'raw'/(stem+'.safetensors')
            tensor_file = save_raw(path,payload,identity['request'],request_sha,self.plan_sha)
            service.write_json(path.with_suffix('.capture.json'),dict(schema=CAPTURE_SCHEMA,
                plan_sha256=self.plan_sha,**identity,tensor_file=tensor_file,
                actual_final_sync_calls=1,actual_raw_decode_calls=1,
                capture_before_postprocess=True,capture_adds_synchronization=True))
        finally:
            pipeline.pipe._sync_pipeline,pipeline.raw.vae.decode = original_sync,original_decode
            self.busy = False


def closed_service(summary_path, summary_sha, plan):
    path = Path(summary_path).resolve(strict=True)
    if service.sha(path) != summary_sha:
        raise ValueError('service summary identity differs')
    summary = read_json(path.read_bytes())
    if (summary.get('status') != 'service_passed' or summary.get('serving_executed') is not True
            or summary.get('http_zmq_executed') is not True or summary.get('model_weights_verified') is not True
            or summary.get('requests') != plan['requests'] or summary.get('worker_exitcodes') != [0,0]):
        raise ValueError('original service workers/requests have not completed')
    for name in SOURCE_KEYS:
        if summary.get(name) != plan['sources'][name]:
            raise ValueError('service source differs from predeclared numerical plan: '+name)
    expected_files = {'rank-0.jsonl','rank-1.jsonl'}
    hashes = summary.get('service_events_sha256',{})
    if set(hashes) != expected_files:
        raise ValueError('both final rank evidence files required')
    directory = path.parent/'service-events'
    for filename,digest in hashes.items():
        target = directory/filename
        if target.is_symlink() or service.sha(target) != digest:
            raise ValueError('final worker evidence differs')
    events = service.read_events(directory)
    service.assess_service(plan['requests'],summary['dispatches'],events,summary['responses'],summary['worker_exitcodes'])
    return summary


def bound_captures(root, plan, plan_sha, summary):
    root = Path(root)
    if (root/'raw').resolve() != root/'raw':
        raise ValueError('raw capture directory must not be linked')
    expected_ids = {r['request_id']:r for r in plan['requests']}
    files = sorted((root/'raw').glob('*.capture.json'))
    if len(files) != 2:
        raise ValueError('both actual raw request captures are required')
    captures = {}
    for path in files:
        row = read_json(path.read_bytes())
        rid = row.get('request_id')
        if rid not in expected_ids or rid in captures or path.name != 'request-'+str(rid)+'.capture.json':
            raise ValueError('duplicate/unoffered raw capture')
        request = expected_ids[rid]['request']
        if (row.get('schema') != CAPTURE_SCHEMA or row.get('plan_sha256') != plan_sha
                or row.get('request') != request or row.get('request_sha256') != canonical(request)
                or row.get('actual_final_sync_calls') != 1 or row.get('actual_raw_decode_calls') != 1
                or row.get('capture_before_postprocess') is not True):
            raise ValueError('capture source/request/raw execution identity differs')
        response = next(r for r in summary['responses'] if r['request_id'] == rid)
        dispatches = [d for d in summary['dispatches'] if d['request_ids'] == [response['native_request_id']] and d['end'] == 28]
        if len(dispatches) != 1:
            raise ValueError('capture requires exactly one actual final scheduler dispatch')
        dispatch = dispatches[0]
        if (row['native_request_id'] != response['native_request_id'] or dispatch['sub_request_ids'] != [row['sub_request_id']]
                or dispatch['ranks'] != row['final_ranks'] or row['owner_rank'] != dispatch['ranks'][0]
                or row['final_interval'] != [dispatch['start'],28]):
            raise ValueError('raw tensor does not belong to the completed HTTP output')
        record = row['tensor_file']
        expected_filename = 'request-'+str(rid)+'.safetensors'
        if record['file'] != expected_filename:
            raise ValueError('raw capture path differs')
        tensor_path = root/'raw'/expected_filename
        descriptors = inspect_safetensors(tensor_path,request,canonical(request),plan_sha)
        if (service.sha(tensor_path) != record['sha256'] or tensor_path.stat().st_size != record['bytes']
                or record['tensors'] != descriptors):
            raise ValueError('raw tensor file differs from actual capture receipt')
        captures[rid] = row
    actual_files = {p.name for p in (root/'raw').iterdir()}
    expected_files = {f'request-{rid}{suffix}' for rid in expected_ids for suffix in ('.safetensors','.capture.json')}
    if actual_files != expected_files or any(p.is_symlink() for p in (root/'raw').iterdir()):
        raise ValueError('raw capture inventory differs')
    return captures


def compare_arrays(reference, actual):
    """Reference-relative predeclared allclose, with full raw-element metrics."""
    import numpy as np
    if reference.shape != actual.shape or reference.dtype != actual.dtype or reference.size == 0:
        raise ValueError('equal nonempty raw array shapes and dtypes required')
    a,b = reference.astype(np.float64),actual.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('raw arrays contain nonfinite values')
    delta = b-a
    norm_a,norm_b,norm_delta = float(np.linalg.norm(a.ravel())),float(np.linalg.norm(b.ravel())),float(np.linalg.norm(delta.ravel()))
    return dict(passed=bool(np.all(np.abs(delta) <= ATOL+RTOL*np.abs(a))),
        exactly_equal=bool(np.array_equal(reference,actual)),elements=int(a.size),atol=ATOL,rtol=RTOL,
        max_abs_error=float(np.abs(delta).max()),rmse=float(np.sqrt(np.mean(delta*delta))),
        relative_l2_error=norm_delta/max(norm_a,1e-12),
        cosine_similarity=float(np.dot(a.ravel(),b.ravel())/(norm_a*norm_b)) if norm_a*norm_b else (1.0 if norm_delta==0 else 0.0))


def independent_reference(args, plan, captures, plan_sha):
    """Fresh process only: original service models and groups must be gone."""
    import torch
    from safetensors.torch import load_file
    from external_sd35_round import SD35RoundPipeline
    if torch.distributed.is_initialized() or torch.cuda.device_count() < 1:
        raise ValueError('reference requires a fresh process and a real GPU')
    if service.sha(args.model_manifest) != plan['sources']['model_manifest_sha256']:
        raise ValueError('reference model manifest differs')
    torch.cuda.set_device(0)
    rounder = SD35RoundPipeline.from_pretrained(args.model_root,shared_python_root=args.shared_python_root,
        model_manifest_path=args.model_manifest,model_manifest_sha256=plan['sources']['model_manifest_sha256'],
        device='cuda:0',attention='math',guidance_scale=4.5,max_sequence_length=256)
    root = args.plan.parent
    rows = []
    try:
        for offered in plan['requests']:
            service.check_storage()
            request,rid = offered['request'],offered['request_id']
            calls = [0]
            def count_forward(*unused): calls[0] += 1
            handle = rounder.pipe.transformer.register_forward_pre_hook(count_forward)
            try:
                with torch.inference_mode():
                    # Normal complete pipeline; no prepare/advance/round slicing.
                    latents = rounder.pipe(prompt=request['prompt'],negative_prompt=request['negative_prompt'],
                        height=request['height'],width=request['width'],num_inference_steps=28,
                        guidance_scale=4.5,max_sequence_length=256,
                        generator=torch.Generator(device='cpu').manual_seed(request['seed']),output_type='latent').images
                    if calls[0] != 28:
                        raise ValueError('independent normal reference did not execute exactly 28 Transformer calls')
                    decoded = rounder.raw.vae.decode(latents/rounder.raw.vae.config.scaling_factor
                        + rounder.raw.vae.config.shift_factor,return_dict=False)[0]
                    torch.cuda.synchronize()
            finally:
                handle.remove()
            reference_file = save_raw(root/'reference'/('request-'+str(rid)+'.safetensors'),
                {'latents':latents,'decoded':decoded},request,canonical(request),plan_sha)
            actual = load_file(str(root/'raw'/captures[rid]['tensor_file']['file']),device='cpu')
            comparisons = {}
            for name,value in {'latents':latents,'decoded':decoded}.items():
                if value.dtype != actual[name].dtype:
                    raise ValueError('service and normal reference raw tensor dtypes differ')
                comparisons[name] = compare_arrays(value.detach().float().cpu().numpy(),actual[name].float().numpy())
                comparisons[name]['raw_dtype'] = str(value.dtype)
                comparisons[name]['bitwise_equal'] = (
                    reference_file['tensors'][name]['tensor_sha256'] == captures[rid]['tensor_file']['tensors'][name]['tensor_sha256'])
            row = dict(request_id=rid,request_sha256=canonical(request),reference_transformer_calls=calls[0],
                normal_pipeline_output_type='latent',generator_device='cpu',reference_file=reference_file,
                service_file_sha256=captures[rid]['tensor_file']['sha256'],comparisons=comparisons,
                numerical_passed=all(c['passed'] for c in comparisons.values()))
            service.write_json(root/'reference'/('request-'+str(rid)+'.comparison.json'),row)
            rows.append(row)
            del latents,decoded,actual
    finally:
        rounder.runtime.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify-captures','reference'])
    for name in ('plan','service-summary'):
        parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
    for name in ('shared-python-root','model-root','model-manifest'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args(argv)
    result = dict(schema=RESULT_SCHEMA,status='failed',numerical_check_passed=False,
                  qualification_eligible=False,allows=[],raw_artifacts_protected=True,
                  tolerance={'atol':ATOL,'rtol':RTOL},scope='Raw numerical diagnostic; not a paper quality qualification')
    stop = threading.Event()
    guard = None
    validated_output = False
    try:
        args.plan = args.plan.absolute()
        if not args.plan.parent.is_relative_to(Path('/shared/output/DITSERVE')):
            raise ValueError('numerical input/output must be in project shared output')
        plan = load_plan(args.plan,args.plan_sha256)
        validated_output = True
        if service.sha(service.__file__) != plan['sources']['service_adapter_sha256']:
            raise ValueError('service verifier differs from captured execution')
        summary = closed_service(args.service_summary,args.service_summary_sha256,plan)
        captures = bound_captures(args.plan.parent,plan,args.plan_sha256,summary)
        result.update(status='captures_verified',captured_requests=2,plan_sha256=args.plan_sha256,
                      service_summary_sha256=args.service_summary_sha256)
        if args.command == 'reference':
            if any(getattr(args,name) is None for name in ('shared_python_root','model_root','model_manifest')):
                raise ValueError('reference requires all verified model/shared paths')
            if (args.plan.parent/'summary.json').exists() or any((args.plan.parent/'reference').iterdir()):
                raise ValueError('reference outputs already exist; preserve prior execution')
            service.check_storage()
            def monitor():
                while not stop.wait(5):
                    try:
                        service.check_storage()
                    except BaseException:
                        os._exit(75)  # Only this fresh reference process; all raw evidence remains.
            guard = threading.Thread(target=monitor,daemon=True)
            guard.start()
            rows = independent_reference(args,plan,captures,args.plan_sha256)
            passed = len(rows)==2 and all(r['numerical_passed'] for r in rows)
            result.update(status='numerical_passed' if passed else 'numerical_failed',
                          numerical_check_passed=passed,comparisons=rows)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc()[-14000:])
    finally:
        stop.set()
        if guard is not None:
            guard.join(timeout=25)
    if args.command == 'reference' and validated_output:
        if args.plan.parent.is_dir() and not (args.plan.parent/'summary.json').exists():
            service.write_json(args.plan.parent/'summary.json',result)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0 if result['status'] in ('captures_verified','numerical_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
