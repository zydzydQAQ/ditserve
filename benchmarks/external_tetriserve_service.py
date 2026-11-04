"""Two-request diagnostic through the original TetriServe HTTP/service/worker loop.

Calibrates real original workers before constructing the native scheduler. The
only worker wrapper observes calls and guarantees process-group cleanup; routing,
batching, scheduling, state transfer, and inference dispatch remain upstream.
This is an SD3.5 integration diagnostic, not a qualified performance comparison.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import asdict
import functools
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

import external_tetriserve_port as port

SCHEMA = 'ditserve.tetriserve-native-service-check.v1'
_OWNED_WORKERS = []
SERVICE_SOURCES = (
    'tetriserve/engine/runtime.py', 'tetriserve/engine/proc_launcher.py',
    'tetriserve/zmq_utils/comm.py', 'tetriserve/light_ray/latent_manager.py',
    'tetriserve/light_ray/latent_future.py', 'tetriserve/distributed/globals.py',
    'tetriserve/distributed/distributed.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_storage():
    proc = subprocess.run(['df', '-BG', '/shared'], capture_output=True, text=True, timeout=20)
    try:
        fields = proc.stdout.splitlines()[-1].split()
        used, percent = int(fields[2].rstrip('G')), int(fields[4].rstrip('%'))
    except (IndexError, ValueError) as exc:
        raise RuntimeError('cannot verify shared storage') from exc
    if proc.returncode or used > 800 or percent >= 80:
        raise RuntimeError('shared storage stop line: used > 800 GiB or usage >= 80%')
    return {'used_gib': used, 'percent': percent}


def write_json(path, value):
    check_storage()
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def request_plans(protocol, prompts, seed, mixed_resolution):
    from external_sd35_round_check import request_plan
    if len(prompts) != 2 or len(set(prompts)) != 2:
        raise ValueError('exactly two distinct explicit prompts are required')
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    shapes = ['sd35-512', 'sd35-1024' if mixed_resolution else 'sd35-512']
    return [dict(request_id=100 + index, shape_id=shape,
                 request=asdict(request_plan(protocol, prompt, seed + index, shape)))
            for index, (prompt, shape) in enumerate(zip(prompts, shapes))]


def calibration_cells(requests):
    sizes = sorted({(row['request']['height'], row['request']['width']) for row in requests})
    return [dict(height=h, width=w, bs=bs, sp=sp, steps=28)
            for h, w in sizes for bs in (1, 2) for sp in (1, 2)]


def measured_profile(cells, events):
    """No interpolated/default/synthetic costs: every cell needs real rank events."""
    rows = []
    for index, cell in enumerate(cells):
        selected = [r for r in events if r.get('event') == 'round' and r.get('request_ids') == [1000 + index]]
        if sorted(r['rank'] for r in selected) != list(range(cell['sp'])):
            raise ValueError('calibration requires exactly one event from every participating rank')
        for row in selected:
            if (row['ranks'] != list(range(cell['sp'])) or row['height'] != cell['height']
                    or row['width'] != cell['width'] or row['batch_size'] != cell['bs']
                    or row['start'] != 0 or row['end'] != 28 or row['transformer_calls'] != 28
                    or row['output_kinds'] != ['image'] * cell['bs']
                    or len(row['prologue_seconds']) != cell['bs']):
                raise ValueError('calibration shape/steps/batch execution differs')
            times = row['prologue_seconds'] + [row['denoising_seconds']]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in times):
                raise ValueError('calibration GPU durations must be finite and positive')
        rows.append(dict(height=cell['height'], width=cell['width'], bs=cell['bs'],
                         ulysses_degree=cell['sp'], ring_degree=1, mlp_world_size=cell['sp'], steps=28,
                         avg_text_encoding_time=max(sum(r['prologue_seconds']) for r in selected),
                         avg_denoising_time=max(r['denoising_seconds'] for r in selected)))
    if len([r for r in events if r.get('event') == 'round']) != sum(c['sp'] for c in cells):
        raise ValueError('unbound or duplicate calibration round')
    return rows


def validate_exits(events):
    for rank in (0, 1):
        terminal = [row['event'] for row in events if row['rank'] == rank and row['event'] != 'round']
        if terminal != ['native_serve_returned', 'distributed_closed']:
            raise ValueError('native serve return and distributed cleanup must be observed on both ranks')
        returned = next(r for r in events if r['rank'] == rank and r['event'] == 'native_serve_returned')
        if returned.get('sender_threads_joined') is not True:
            raise ValueError('original latent sender thread did not finish before distributed teardown')


def assess_service(requests, dispatches, events, responses, exits):
    """Reconcile original scheduler dispatch, all workers, and actual HTTP outputs."""
    if exits != [0, 0] or len(responses) != 2 or len(dispatches) == 0:
        raise ValueError('both original workers and both HTTP requests must finish')
    validate_exits(events)
    request_map = {r['request_id']: r['request'] for r in requests}
    if len(request_map) != 2 or sorted(r['request_id'] for r in responses) != sorted(request_map):
        raise ValueError('HTTP request denominator differs')
    # API assigns IDs itself; explicitly bind each returned completion to its
    # input via observation of CServeRuntime.generate, never assume ID order.
    api_ids = {r['native_request_id']: request_map[r['request_id']] for r in responses}
    if len(api_ids) != 2:
        raise ValueError('native HTTP request IDs are not unique')
    coverage = {rid: [] for rid in api_ids}
    expected_rounds = []
    for dispatch in dispatches:
        ranks = dispatch['ranks']
        if ranks not in ([0], [1], [0, 1]):
            raise ValueError('unsupported native rank assignment')
        if len(dispatch['request_ids']) != 1 or dispatch['request_ids'][0] not in api_ids:
            raise ValueError('dispatch must belong to an actual HTTP request')
        rid = dispatch['request_ids'][0]
        request = api_ids[rid]
        if (dispatch['height'], dispatch['width'], dispatch['batch_size']) != (request['height'], request['width'], 1):
            raise ValueError('service request shape or output count changed')
        if dispatch['inputs'] != dict(prompt=[request['prompt']], negative_prompt=request['negative_prompt'],
                seed=request['seed'], guidance_scale=request['guidance_scale'],
                max_sequence_length=request['max_sequence_length'], origin_steps=[request['steps']]):
            raise ValueError('native request prompt/seed/CFG/text/schedule identity differs')
        coverage[rid].append((dispatch['start'], dispatch['end']))
        for rank in ranks:
            expected_rounds.append((rank, tuple(dispatch['sub_request_ids']), dispatch['start'], dispatch['end']))
    actual_rounds = []
    for row in events:
        if row.get('event') != 'round':
            continue
        key = (row['rank'], tuple(row['sub_request_ids']), row['start'], row['end'])
        matches = [d for d in dispatches if (tuple(d['sub_request_ids']), d['start'], d['end']) == key[1:]]
        if len(matches) != 1:
            raise ValueError('worker round lacks exactly one native dispatch')
        dispatch = matches[0]
        if any(row[k] != dispatch[k] for k in ('request_ids','ranks','height','width','batch_size','inputs')):
            raise ValueError('worker execution differs from scheduler dispatch')
        if row['transformer_calls'] != row['end'] - row['start']:
            raise ValueError('actual Transformer forward count differs')
        expected_kind = 'image' if row['end'] == 28 else 'latent_future'
        if row['output_kinds'] != [expected_kind]:
            raise ValueError('native intermediate/final output type differs')
        if row['start'] and row['continuation_count'] != 1:
            raise ValueError('continued round must restore its original LatentFuture')
        actual_rounds.append(key)
    if sorted(expected_rounds) != sorted(actual_rounds):
        raise ValueError('all scheduled rank executions must be observed exactly once')
    for rid, intervals in coverage.items():
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor or not start < end <= api_ids[rid]['steps']:
                raise ValueError('native schedule repeats/skips a denoising interval')
            cursor = end
        if cursor != 28:
            raise ValueError('request did not execute all 28 steps')
    for response in responses:
        request = request_map[response['request_id']]
        if (response['http_status'] != 200 or response['image_count'] != 1
                or response['size'] != [request['width'], request['height']]
                or response['mode'] != 'RGB' or len(response['pixels_sha256']) != 64
                or not math.isfinite(response['elapsed_seconds']) or response['elapsed_seconds'] <= 0):
            raise ValueError('HTTP response is not one real correctly sized RGB image')
    return dict(requests_completed=2, transformer_steps_per_request=[28, 28],
                native_rounds=len(dispatches), rank_executions=len(actual_rounds),
                continuation_rounds=sum(d['start'] > 0 for d in dispatches),
                observed_rank_groups=sorted({tuple(d['ranks']) for d in dispatches}))


def append_event(rank, value):
    directory = Path(os.environ['DITSERVE_TETRI_SERVICE_EVENTS'])
    check_storage()
    path = directory / ('rank-' + str(rank) + '.jsonl')
    if path.exists() and path.stat().st_size > 2 * 1024 * 1024:
        raise RuntimeError('diagnostic event size bound exceeded')
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(dict(rank=rank, **value), sort_keys=True, allow_nan=False) + '\n')


def verify_native_imports(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'PORT_MANIFEST.json').read_bytes())
    for name, module in sys.modules.items():
        if name == 'tetriserve' or name.startswith('tetriserve.'):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(root) or sha(path) != manifest['files'].get(path.relative_to(root).as_posix()):
                    raise ValueError('native import escaped bound derived source: ' + name)


def observed_launch_worker(*args):
    """Spawn target delegates to the byte-identical native launcher and serve loop."""
    from external_tetriserve_migration import verify_derived
    config = json.loads(Path(os.environ['DITSERVE_TETRI_PORT_CONFIG']).read_bytes())
    verify_derived(Path(config['derived_manifest']).parent, config['derived_manifest_sha256'])
    import torch
    from tetriserve.engine.runner import FluidServeRunner
    from tetriserve.engine.proc_launcher import launch_worker
    from tetriserve.light_ray.latent_future import LatentFuture
    from tetriserve.light_ray.latent_manager import LatentManager
    from PIL.Image import Image
    verify_native_imports(Path(config['derived_manifest']).parent)
    rank = args[0]
    original = FluidServeRunner.generate
    capture = None
    if os.environ['DITSERVE_TETRI_SERVICE_PHASE'] == 'service':
        from external_tetriserve_service_numerics import CaptureSession
        capture = CaptureSession(os.environ['DITSERVE_TETRI_NUMERICS_PLAN'],
                                 os.environ['DITSERVE_TETRI_NUMERICS_PLAN_SHA256'],rank)
    original_stop = LatentManager.stop_sender_thread
    stopped_managers = []
    def stop_sender(manager):
        original_stop(manager)
        # Upstream sends the sentinel but leaves join commented out. Complete
        # that lifecycle before destroying groups, without changing transport.
        if manager.sender_thread is not None:
            manager.sender_thread.join(timeout=10)
            if manager.sender_thread.is_alive():
                raise RuntimeError('native latent sender did not stop')
        stopped_managers.append(len(manager.latent_store))
    LatentManager.stop_sender_thread = stop_sender

    @functools.wraps(original)
    def generate(self, reqs, *rest, **kwargs):
        pipeline = self.pipeline
        count = [0]
        def count_forward(*unused):
            count[0] += 1
        handle = pipeline.pipe.transformer.register_forward_pre_hook(count_forward)
        timings = {'prologue_seconds': [], 'denoising_seconds': 0.0}
        calibration = os.environ['DITSERVE_TETRI_SERVICE_PHASE'] == 'calibration'
        saved = {}
        if calibration:
            for name in ('inference_prologue', 'inference_denoising'):
                saved[name] = getattr(pipeline, name)
                def timed(*a, _name=name, **kw):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    out = saved[_name](*a, **kw)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    if _name == 'inference_prologue':
                        timings['prologue_seconds'].append(elapsed)
                    else:
                        timings['denoising_seconds'] += elapsed
                    return out
                setattr(pipeline, name, timed)
        try:
            with capture.observe_round(pipeline,reqs) if capture is not None else nullcontext():
                output = original(self, reqs, *rest, **kwargs)
            kinds = ['image' if isinstance(v, Image) else 'latent_future' if isinstance(v, LatentFuture) else type(v).__name__ for v in output]
            futures = kwargs.get('latent_future_dict', rest[0] if rest else {}) or {}
            append_event(rank, dict(event='round', **dispatch_record(reqs),
                         transformer_calls=count[0], output_kinds=kinds,
                         continuation_count=sum(isinstance(v, LatentFuture) for v in futures.values()), **timings))
            return output
        finally:
            handle.remove()
            for name, value in saved.items():
                setattr(pipeline, name, value)
    FluidServeRunner.generate = generate
    try:
        launch_worker(*args)
        if not stopped_managers:
            raise RuntimeError('native LatentManager cleanup did not execute')
        append_event(rank, dict(event='native_serve_returned', sender_threads_joined=True,
                               latent_store_entries_before_process_exit=stopped_managers))
    except BaseException:
        append_event(rank, dict(event='worker_failed', traceback=traceback.format_exc()[-12000:]))
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        append_event(rank, dict(event='distributed_closed'))


def dispatch_record(req):
    return dict(request_ids=list(req.req_ids), sub_request_ids=list(req.sub_req_ids),
                ranks=list(req.parallel_ranks), start=req.start_step, end=req.end_step,
                height=req.input_config.height, width=req.input_config.width,
                batch_size=req.input_config.batch_size,
                inputs=dict(prompt=req.input_config.prompt, negative_prompt=req.input_config.negative_prompt,
                    seed=req.input_config.seed, guidance_scale=req.input_config.guidance_scale,
                    max_sequence_length=req.input_config.max_sequence_length, origin_steps=list(req.origin_steps)))


def read_events(directory):
    result = []
    for rank in (0, 1):
        path = directory / ('rank-' + str(rank) + '.jsonl')
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if any(row.get('rank') != rank for row in rows):
                raise ValueError('event file rank identity differs')
            result.extend(rows)
    return result


async def stop_workers(processes, router, graceful=True):
    processes = [p for p in processes if p.pid is not None]
    exits = []
    try:
        if not graceful:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        elif len(processes) == 2 and all(p.is_alive() for p in processes):
            await router.send_to_workers(['Worker-0', 'Worker-1'], {'type': 'done'})
        for process in processes:
            await asyncio.to_thread(process.join, 45 if graceful else 10)
        exits = [p.exitcode for p in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()  # Only children created by this diagnostic.
                await asyncio.to_thread(process.join, 10)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
        router.router.close(linger=0)
        router.context.term()
    if graceful and exits != [0, 0]:
        raise RuntimeError('native workers did not exit successfully: ' + repr(exits))
    return exits


async def calibrate(args, requests, engine_config):
    from tetriserve.zmq_utils import Router, find_free_port
    from tetriserve.config.request import ScheduledRequests
    from tetriserve.xfuser.config import InputConfig
    context = mp.get_context('spawn')
    queues, sync = [context.Queue() for _ in range(2)], [context.Queue() for _ in range(2)]
    router_port, dist_port = find_free_port(), find_free_port()
    router = Router('localhost', router_port, 2)
    processes = [context.Process(target=observed_launch_worker,
                 args=(rank, 2, router_port, '127.0.0.1', dist_port, engine_config, False, False, queues, sync)) for rank in range(2)]
    _OWNED_WORKERS.extend(processes)
    cells = calibration_cells(requests)
    try:
        for process in processes:
            process.start()
        await router.init()
        for index, cell in enumerate(cells):
            request = requests[0]['request']
            config = InputConfig(batch_size=cell['bs'], height=cell['height'], width=cell['width'],
                prompt=[row['request']['prompt'] for row in requests[:cell['bs']]],
                negative_prompt=request['negative_prompt'], seed=request['seed'], num_inference_steps=28,
                guidance_scale=request['guidance_scale'], max_sequence_length=request['max_sequence_length'])
            req = ScheduledRequests(req_ids=[1000 + index], sub_req_ids=[10000 + index*2 + j for j in range(cell['bs'])],
                parallel_ranks=list(range(cell['sp'])), attn_ulysses_degree=cell['sp'], attn_ring_degree=1,
                sp_degree=cell['sp'], latent_length=cell['height']//8*(cell['width']//8), input_config=config,
                origin_steps=[28]*cell['bs'], start_step=0, end_step=28, is_last_steps=[True]*cell['bs'],
                arrival_time=0.0, latency_threshold=600.0, estimated_time=0.0)
            await router.send_to_workers(['Worker-'+str(r) for r in req.parallel_ranks],
                                        {'type':'generate', 'Requests':req, 'latent_future_dict':{}})
            identity, output = await router.recv_from_worker()
            if identity != 'Worker-0' or output.get('status') != 'success' or output['output'].sub_req_ids != req.sub_req_ids:
                raise ValueError('native calibration response identity differs')
        return cells
    finally:
        normal = sys.exc_info()[0] is None
        await stop_workers(processes, router, graceful=normal)
        for queue in queues + sync:
            if not normal:
                queue.cancel_join_thread()
            queue.close()
            if normal:
                queue.join_thread()


def http_request(url, payload):
    import base64
    import io
    from PIL import Image
    begin = time.perf_counter()
    request = urllib.request.Request(url, json.dumps(payload).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = response.read(24 * 1024 * 1024 + 1)
        status = response.status
    if len(body) > 24 * 1024 * 1024:
        raise ValueError('HTTP response bound exceeded')
    value = json.loads(body)
    if set(value) != {'images'} or len(value['images']) != 1:
        raise ValueError('expected exactly one HTTP PNG output')
    image = Image.open(io.BytesIO(base64.b64decode(value['images'][0], validate=True)))
    image.load()
    return dict(http_status=status, image_count=1, size=list(image.size), mode=image.mode,
                pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(), elapsed_seconds=time.perf_counter()-begin)


async def serve(args, requests, engine_config, profile_root):
    import uvicorn
    from tetriserve.config.args import ServerArgs
    from tetriserve.zmq_utils import find_free_port
    import tetriserve.engine.runtime as native
    from tetriserve.server import api_server
    config = json.loads(Path(os.environ['DITSERVE_TETRI_PORT_CONFIG']).read_bytes())
    verify_native_imports(Path(config['derived_manifest']).parent)
    native.launch_worker = observed_launch_worker
    server_args = ServerArgs(nproc_per_node=2, engine_config=engine_config,
        schedule_logic='dyn_slo_schedule_global_dp', perf_model_dir=str(profile_root), log_requests=True,
        fake_engine=False, skip_latent_transfer=False)
    runtime = api_server.create_runtime(server_args)
    _OWNED_WORKERS.extend(runtime.engine.engine_processes)
    dispatches, native_ids = [], {}
    original_dispatch = runtime.engine.handle_single_bunch_requests
    async def observe_dispatch(req, *a, **kw):
        dispatches.append(dispatch_record(req))
        return await original_dispatch(req, *a, **kw)
    runtime.engine.handle_single_bunch_requests = observe_dispatch
    original_generate = runtime.generate
    async def observe_generate(config, request_id=None, **kw):
        prompt = config.prompt[0] if isinstance(config.prompt, list) else config.prompt
        match = [r for r in requests if r['request']['prompt'] == prompt]
        if len(match) != 1 or match[0]['request_id'] in native_ids:
            raise ValueError('unexpected/duplicate HTTP input')
        native_ids[match[0]['request_id']] = request_id
        return await original_generate(config, request_id, **kw)
    runtime.generate = observe_generate
    port_number = find_free_port()
    server = uvicorn.Server(uvicorn.Config(api_server.app, host='127.0.0.1', port=port_number,
                            log_level='info', timeout_graceful_shutdown=5))
    http_task = None
    try:
        await runtime.init_zmq_router()
        http_task = asyncio.create_task(server.serve())
        while not server.started:
            if http_task.done():
                await http_task
                raise RuntimeError('HTTP server exited before ready')
            await asyncio.sleep(.05)
        payloads = []
        for row in requests:
            request = row['request']
            payloads.append(dict(prompt=request['prompt'], height=request['height'], width=request['width'],
                num_inference_steps=request['steps'], guidance_scale=request['guidance_scale'], seed=request['seed'],
                negative_prompt=request['negative_prompt'], max_sequence_length=request['max_sequence_length'],
                batch_size=1, output_type='pil', latency_threshold=args.slo_seconds))
        results = await asyncio.gather(*(asyncio.to_thread(http_request,
                  'http://127.0.0.1:'+str(port_number)+'/v1/generate', p) for p in payloads))
        responses = [dict(request_id=r['request_id'], native_request_id=native_ids[r['request_id']], **out)
                     for r, out in zip(requests, results)]
        if runtime.req_states or runtime.engine.scheduler.remaining_requests() or runtime.engine.scheduler.running_requests:
            raise ValueError('native request ledger is not drained')
    finally:
        normal = sys.exc_info()[0] is None
        server.should_exit = True
        server.force_exit = not normal
        if not normal:
            for process in runtime.engine.engine_processes:
                if process.is_alive():
                    process.terminate()
        if http_task is not None:
            await http_task
        exits = await stop_workers(runtime.engine.engine_processes, runtime.engine.zmq_router, graceful=normal)
        for queue in runtime.engine.latent_request_queues + runtime.engine.sync_queues:
            if not normal:
                queue.cancel_join_thread()
            queue.close()
            if normal:
                queue.join_thread()
    return dispatches, responses, exits


async def execution(args, requests, derived, config):
    from external_tetriserve_migration import upstream_engine_config
    from tetriserve.utils import get_gpu_name
    sys.path.insert(0, derived['root'])
    config_path = args.output_dir/'calibration-port-config.json'
    write_json(config_path, config)
    os.environ.update(DITSERVE_TETRI_PORT_CONFIG=str(config_path),
                      DITSERVE_TETRI_PORT_CONFIG_SHA256=sha(config_path))
    engine_config = upstream_engine_config(config['model'])
    calibration_dir = args.output_dir/'calibration-events'
    calibration_dir.mkdir()
    os.environ.update(DITSERVE_TETRI_SERVICE_EVENTS=str(calibration_dir), DITSERVE_TETRI_SERVICE_PHASE='calibration')
    cells = await calibrate(args, requests, engine_config)
    calibration_events = read_events(calibration_dir)
    validate_exits(calibration_events)
    profile = measured_profile(cells, calibration_events)
    profile_root = args.output_dir/'measured-profile'
    profile_path = profile_root/'sd35'/get_gpu_name()/'e2e_scaling_efficiency.json'
    profile_path.parent.mkdir(parents=True)
    write_json(profile_path, profile)
    config = dict(config, cost_profile_sha256=sha(profile_path), server_launch_allowed=True)
    config_path = args.output_dir/'service-port-config.json'
    write_json(config_path, config)
    events_dir = args.output_dir/'service-events'
    events_dir.mkdir()
    os.environ.update(DITSERVE_TETRI_PORT_CONFIG=str(config_path), DITSERVE_TETRI_PORT_CONFIG_SHA256=sha(config_path),
                      DITSERVE_TETRI_SERVICE_EVENTS=str(events_dir), DITSERVE_TETRI_SERVICE_PHASE='service')
    dispatches, responses, exits = await serve(args, requests, engine_config, profile_root)
    events = read_events(events_dir)
    counts = assess_service(requests, dispatches, events, responses, exits)
    return dict(counts=counts, dispatches=dispatches, responses=responses, worker_exitcodes=exits,
                cost_profile_sha256=sha(profile_path), calibration_cells=cells,
                calibration_timing_scope='One measured repeat per cell; denoising includes VAE decode; diagnostic costs only',
                service_events_sha256={p.name:sha(p) for p in events_dir.iterdir()},
                cost_calibration_sha256={p.name:sha(p) for p in calibration_dir.iterdir()})


async def guarded_execution(args, *rest):
    existing_children = {child.pid for child in mp.active_children()}
    task = asyncio.create_task(execution(args, *rest))
    async def monitor():
        while True:
            check_storage()
            failed = [(p.pid,p.exitcode) for p in _OWNED_WORKERS if p.exitcode not in (None,0)]
            if failed:
                raise RuntimeError('original worker failed: ' + repr(failed))
            await asyncio.sleep(5)
    guard = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait([task, guard], timeout=args.timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
        if guard in done:
            await guard
        if task not in done:
            raise TimeoutError('complete service diagnostic exceeded timeout')
        return await task
    finally:
        for pending in (task, guard):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, guard, return_exceptions=True)
        # Also covers upstream constructor failures after spawning workers but
        # before returning the engine object to our normal cleanup scope.
        for child in mp.active_children():
            if child.pid not in existing_children and child.is_alive():
                child.terminate()
                await asyncio.to_thread(child.join, 5)
                if child.is_alive():
                    child.kill()
                    await asyncio.to_thread(child.join, 5)


def bind_derived_adapters(derived):
    """Discard only the pure planning module before loading its identical copy.

    The bridge explicitly checks its own and the round adapter's execution path.
    No runtime/model instance exists at this point and no external package is
    impersonated. Spawned workers independently load the derived files.
    """
    root = Path(derived['root']).resolve()
    manifest = json.loads((root/'PORT_MANIFEST.json').read_bytes())
    if sha(root/'external_sd35_round.py') != manifest['files']['external_sd35_round.py']:
        raise ValueError('derived execution adapter bytes differ')
    for name in sys.modules:
        if name == 'tetriserve' or name.startswith('tetriserve.') or name == 'xfuser' or name.startswith('xfuser.'):
            raise ValueError('native service/backend must not already be imported')
    previous = sys.modules.get('external_sd35_round')
    if previous is not None:
        if sha(previous.__file__) != manifest['files']['external_sd35_round.py']:
            raise ValueError('planning adapter bytes differ from derived execution adapter')
        del sys.modules['external_sd35_round']
    sys.path.insert(0, str(root))
    module = importlib.import_module('external_sd35_round')
    if Path(module.__file__).resolve() != root/'external_sd35_round.py':
        raise ValueError('derived adapter import path differs')
    return module


def run_numerical_reference(args, result, plan_path, plan_sha):
    """Write immutable service completion first; compare in a fresh subprocess."""
    import external_tetriserve_service_numerics as numerics
    completion = args.output_dir/'service-completion.json'
    write_json(completion,result)
    command = [sys.executable,'-B',str(Path(numerics.__file__).resolve()),'reference',
        '--plan',str(plan_path),'--plan-sha256',plan_sha,
        '--service-summary',str(completion),'--service-summary-sha256',sha(completion),
        '--shared-python-root',str(args.shared_python_root),
        '--model-root',str(args.model_root),'--model-manifest',str(args.model_manifest)]
    env = os.environ.copy()
    # A detached SP1 reference process, after native distributed workers exited.
    # This does not alter the service allocation or the platform environment.
    env.update(WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',PYTHONDONTWRITEBYTECODE='1')
    process = subprocess.run(command,env=env,capture_output=True,text=True,
                             timeout=args.reference_timeout_seconds)
    summary_path = plan_path.parent/'summary.json'
    evidence = dict(returncode=process.returncode,stdout_tail=process.stdout[-12000:],
                    stderr_tail=process.stderr[-12000:],service_completion_sha256=sha(completion))
    if summary_path.is_file():
        summary = numerics.read_json(summary_path.read_bytes())
        evidence.update(summary=str(summary_path),summary_sha256=sha(summary_path),result=summary)
    else:
        summary = {}
    passed = (process.returncode == 0 and summary.get('status') == 'numerical_passed'
              and summary.get('numerical_check_passed') is True
              and summary.get('plan_sha256') == plan_sha
              and summary.get('service_summary_sha256') == sha(completion)
              and len(summary.get('comparisons',[])) == 2)
    return passed,evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','run'])
    for name in ('source-root','source-manifest','shared-python-root','model-root','model-manifest','protocol','output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    for name in ('source-manifest-sha256','model-manifest-sha256','protocol-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--mixed-resolution', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--reference-timeout-seconds', type=int, default=1800)
    parser.add_argument('--slo-seconds', type=float, default=600.)
    args = parser.parse_args(argv)
    result = dict(schema=SCHEMA, status='failed', qualification_eligible=False, allows=[],
                  http_zmq_executed=False, serving_executed=False, numerical_check_passed=False,
                  model_weights_verified=False,
                  performance_measurements=False, full_formal_baseline=False,
                  service_batching_covered=False, cross_rank_migration_required=False,
                  limitations=['Two independent HTTP requests, one image per request; native cross-request batching is not implemented upstream',
                      'BS2 is exercised only by calibration, not the HTTP workload',
                      'Native allocation is unrestricted within SP1/SP2; cross-rank migration is observed, not forced',
                      'Raw numerical tolerance is a diagnostic criterion, not a paper quality qualification',
                      'Capture adds CPU transfer synchronization; these service timings are instrumented'],
                  scope='Original TetriServe HTTP/runtime/worker service with shared SD3.5 model; two-request diagnostic')
    result['service_adapter_sha256'] = sha(__file__)
    created = False
    try:
        if (not 300 <= args.timeout_seconds <= 7200 or not 120 <= args.reference_timeout_seconds <= 3600
                or not math.isfinite(args.slo_seconds) or args.slo_seconds <= 0):
            raise ValueError('invalid diagnostic timeout/SLO')
        args.output_dir = args.output_dir.resolve()
        sys.dont_write_bytecode = True
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        for name in ('source_root','source_manifest','shared_python_root','model_root','model_manifest','protocol'):
            setattr(args, name, getattr(args, name).resolve(strict=True))
        if sha(args.protocol) != args.protocol_sha256:
            raise ValueError('frozen protocol SHA256 differs')
        requests = request_plans(json.loads(args.protocol.read_bytes()), args.prompt, args.seed, args.mixed_resolution)
        import external_sd35_round as round_adapter
        shared = round_adapter.verify_shared_sources(args.shared_python_root)
        model_manifest = json.loads(args.model_manifest.read_bytes())
        if (sha(args.model_manifest) != args.model_manifest_sha256 or model_manifest.get('model') != round_adapter.SD35_MODEL
                or model_manifest.get('revision') != round_adapter.SD35_REVISION):
            raise ValueError('frozen model manifest differs')
        source_plan, _, _ = port.plan(args.source_root, args.source_manifest, args.source_manifest_sha256)
        official = port.upstream.verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
        # Manifest formats are verified by upstream; explicit comparison below
        # establishes which service components execute byte-identically.
        preserved = {name:sha(args.source_root/name) for name in SERVICE_SOURCES}
        if any(source_plan['files'][name] != digest for name, digest in preserved.items()):
            raise ValueError('native service loop/transport was modified')
        result.update(requests=requests, native_service_sources=preserved,
                      source_manifest_sha256=args.source_manifest_sha256, protocol_sha256=args.protocol_sha256,
                      model_manifest_sha256=args.model_manifest_sha256, shared_worker_sha256=round_adapter.FROZEN_WORKER_SHA256,
                      calibration_cells=calibration_cells(requests),
                      source_verification=dict(commit=official['commit'], archive_sha256=official['archive_sha256'],
                                               verified_files=len(official['files'])))
        if args.command == 'plan':
            result['status'] = 'planned'
            print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
            return 0
        if not args.output_dir.is_relative_to(Path('/shared/output/DITSERVE')):
            raise ValueError('all GPU outputs must be under /shared/output/DITSERVE')
        for root in (args.source_root, shared, args.model_root, Path(__file__).resolve().parent):
            if args.output_dir.is_relative_to(root.resolve()) or root.resolve().is_relative_to(args.output_dir):
                raise ValueError('outputs and immutable inputs must be disjoint')
        check_storage()
        import torch
        if torch.cuda.device_count() != 2:
            raise ValueError('exactly two visible GPUs required; use one process, not torchrun')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created = True
        derived = port.apply(args.source_root, args.source_manifest, args.source_manifest_sha256, args.output_dir/'derived')
        sys.path[:0] = [derived['root'], str(shared)]
        round_adapter = bind_derived_adapters(derived)
        config = dict(schema='ditserve.tetriserve-sd35-port.v1', model=round_adapter.SD35_MODEL,
            revision=round_adapter.SD35_REVISION, shared_worker_sha256=round_adapter.FROZEN_WORKER_SHA256,
            shared_python_root=str(shared), model_path=str(args.model_root.resolve()),
            model_manifest=str(args.model_manifest.resolve()), model_manifest_sha256=args.model_manifest_sha256,
            guidance_scale=4.5, max_sequence_length=256, attention='math',
            resolutions=sorted({(r['request']['height'], r['request']['width']) for r in requests}),
            derived_manifest=str(Path(derived['root'])/'PORT_MANIFEST.json'),
            derived_manifest_sha256=derived['manifest_sha256'], diagnostic_only=True, server_launch_allowed=False)
        result.update(derived_manifest_sha256=derived['manifest_sha256'])
        import external_tetriserve_service_numerics as numerics
        sources = {name:result[name] for name in numerics.SOURCE_KEYS if name!='numerics_adapter_sha256'}
        sources['numerics_adapter_sha256'] = sha(numerics.__file__)
        plan_path,plan_sha = numerics.create_plan(args.output_dir/'numerics',requests,sources,args.protocol)
        result.update(numerics_plan=str(plan_path),numerics_plan_sha256=plan_sha,
                      numerics_adapter_sha256=sources['numerics_adapter_sha256'],raw_artifacts_protected=True)
        os.environ.update(DITSERVE_TETRI_NUMERICS_PLAN=str(plan_path),DITSERVE_TETRI_NUMERICS_PLAN_SHA256=plan_sha)
        # Upstream uses relative ipc://localhost:<port> sockets. Keep these tiny,
        # transient socket inodes on the container filesystem, not shared NFS.
        # All model data, measurements and failure artifacts use absolute /shared
        # paths. The directory contains only this process's IPC endpoints.
        before_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='ditserve-tetri-ipc-', dir='/tmp') as ipc:
            try:
                os.chdir(ipc)
                result.update(asyncio.run(guarded_execution(args, requests, derived, config)))
            finally:
                os.chdir(before_cwd)
        result['ipc_scope'] = 'Original relative ZMQ IPC endpoints in a private ephemeral container directory'
        result.update(status='service_passed', http_zmq_executed=True, serving_executed=True,
                      model_weights_verified=True)
        passed,reference = run_numerical_reference(args,result,plan_path,plan_sha)
        result.update(status='service_and_numerics_passed' if passed else 'numerical_failed',
                      numerical_check_passed=passed,independent_reference=reference)
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-16000:])
    if created:
        write_json(args.output_dir/'summary.json', result)
        print(json.dumps(dict(status=result['status'], summary=str(args.output_dir/'summary.json'),
                              sha256=sha(args.output_dir/'summary.json'), qualification_eligible=False)), flush=True)
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result['status'] == 'service_and_numerics_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
