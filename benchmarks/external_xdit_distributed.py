"""Official xDiT Wan SP2/SP4 diagnostic; no serving or shared-backend claim.

Launch with torchrun on one node. The official runner owns all model parallel
communication and returns full outputs on every SP rank when DP=1. Only small
source-bound receipts are written; decoded arrays are never transferred as Python
objects or saved. The outer launcher must additionally verify torchrun's exit.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

import external_xdit_native as native


def topology(world, environ):
    if type(world) is not int or world not in (2, 4):
        raise ValueError('only SP2/SP4 single-node diagnostics are supported')
    values = {}
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE'):
        value = environ.get(key, '')
        if not isinstance(value, str) or not value.isdecimal():
            raise ValueError('missing or invalid torchrun topology: ' + key)
        values[key] = int(value)
    rank = values['RANK']
    if (values['WORLD_SIZE'] != world or values['LOCAL_WORLD_SIZE'] != world
            or not 0 <= rank < world or values['LOCAL_RANK'] != rank):
        raise ValueError('requires one node with unique rank/local-rank in the declared world')
    return rank


def distributed_config(model, request, identity, output, world):
    if type(world) is not int or world not in (2, 4):
        raise ValueError('only SP2/SP4 are supported')
    config = native.runner_config(model, request, identity, output, cpu_offload=False)
    config.update(ulysses_degree=world, ring_degree=1, use_ray=False)
    return config


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def normalize_uuid(value):
    if not isinstance(value, str):
        raise ValueError('missing actual GPU UUID')
    try:
        return str(uuid.UUID(value[4:] if value.startswith('GPU-') else value))
    except (ValueError, AttributeError) as exc:
        raise ValueError('malformed actual GPU UUID') from exc


def publish_json(path, value):
    """Publish a complete receipt atomically within an exclusively owned directory."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('previous evidence must be preserved: ' + str(path))
    pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError('receipt already published: ' + str(path))
    # The rank directory is created exclusively before any writes. Only its
    # owning rank may publish here; readers never observe a partial JSON file.
    pending.rename(path)


def aggregate(rows, world):
    """Fail closed on incomplete, mismatched or failed rank receipts."""
    if len(rows) != world or {r.get('rank') for r in rows} != set(range(world)):
        raise ValueError('complete unique rank receipts are required')
    rows = sorted(rows, key=lambda r: r['rank'])
    uuids = [normalize_uuid(r.get('gpu_uuid')) for r in rows]
    if len(set(uuids)) != world:
        raise ValueError('actual GPU UUIDs must be present and distinct')
    first = rows[0]
    common = ('request', 'model_identity', 'source', 'protocol_sha256', 'adapter', 'native_dtype')
    for row in rows:
        if (row.get('status') != 'native_rank_completed' or row.get('cleanup_completed') is not True
                or row.get('planned_returncode') != 0 or row.get('world_size') != world
                or row.get('local_rank') != row['rank'] or row.get('actual_sp_degree') != world
                or row.get('actual_dp_degree') != 1 or row.get('qualification_eligible') is not False):
            raise ValueError('rank execution/topology/cleanup receipt is incomplete')
        if any(row.get(k) != first.get(k) for k in common):
            raise ValueError('rank source or request identities differ')
        request = row['request']; output = row.get('output', {})
        if (output.get('shape') != [request['num_frames'], request['height'], request['width'], 3]
                or output.get('finite') is not True or output.get('outputs') != 1
                or not output.get('floating_point_frames')):
            raise ValueError('every native rank must return a complete finite decoded video')
        native.check_effective_request(request, row['effective_request'])
        if not row.get('loaded_source_sha256'):
            raise ValueError('actual imported source inventory is missing')
    output_identity = ('shape', 'dtype', 'sha256')
    if any(any(row['output'].get(k) != first['output'].get(k) for k in output_identity) for row in rows):
        raise ValueError('SP ranks disagree on the complete output shape, dtype or digest')
    return dict(schema='ditserve.external-xdit-distributed.v1', status='native_sp_request_completed',
        world_size=world, requested_requests=1, completed_requests=1,
        selected_output_rank=0, output=first['output'], request=first['request'],
        source=first['source'], model_identity=first['model_identity'],
        rank_receipts=rows, normalized_gpu_uuids=uuids, all_rank_cleanup_completed=True,
        cross_rank_output_digest_equal=len({r['output']['sha256'] for r in rows}) == 1,
        torchrun_exit_verified=False,
        exit_scope='Rank receipts record intended return codes after cleanup; verify actual torchrun exit in the outer wrapper.',
        qualification_eligible=False, allows=[], serving_executed=False, quality_evaluated=False,
        formal_benchmark_executed=False, shared_backend_comparison=False,
        scope='One full native SP request at a formal shape; not serving, independent SP1 equivalence or comparative performance.')


def run_rank(model, request, identity, config, source_root, manifest, rank, world, result):
    if any(n == 'xfuser' or n.startswith('xfuser.') for n in sys.modules):
        raise ValueError('native distributed execution requires a fresh interpreter')
    sys.path.insert(0, str(Path(source_root).resolve()))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    import torch
    if torch.cuda.device_count() != world:
        raise ValueError('single-node GPU visibility differs from the declared world')
    torch.cuda.set_device(rank)
    from xfuser.runner import xFuserModelRunner
    from xfuser.core.distributed import get_world_group, get_sequence_parallel_world_size, get_data_parallel_world_size
    native.loaded_sources(source_root, manifest)

    class LocalCheckpointRunner(xFuserModelRunner):
        def _select_model(self, model_name, upstream_config):
            instance = super()._select_model(model_name, upstream_config)
            if (type(instance).__name__ != native.CLASSES[model]
                    or instance.settings.model_name != identity['model_id']):
                raise ValueError('official runner selected an unexpected checkpoint')
            instance.settings.model_name = identity['root']
            return instance

    runner = None
    initialized = False
    execution_error = None
    result['failed_stage'] = 'construct_runner'
    started = time.monotonic()
    try:
        runner = LocalCheckpointRunner(config)
        if (torch.distributed.get_rank() != rank or torch.distributed.get_world_size() != world
                or get_world_group().local_rank != rank):
            raise ValueError('actual initialized distributed topology differs')
        result['failed_stage'] = 'preprocess_request'
        effective = runner.preprocess_args(dict(vars(runner.config)))
        native.check_effective_request(request, effective)
        result['failed_stage'] = 'load_model'
        runner.initialize(effective)
        initialized = True
        torch.cuda.synchronize()
        pipeline = runner.model.pipe
        if type(pipeline).__name__ != 'WanPipeline' or getattr(pipeline, 'transformer_2', None) is not None:
            raise ValueError('expected native single-transformer WanPipeline')
        result['load_seconds'] = time.monotonic() - started
        result['components'] = {name: dict(class_name=type(component).__name__,
            dtype=str(getattr(component, 'dtype', None)), device=str(getattr(component, 'device', None)))
            for name, component in pipeline.components.items()}
        for name in ('transformer', 'text_encoder', 'vae'):
            if result['components'][name]['dtype'] != 'torch.bfloat16':
                raise ValueError('official component precision changed: ' + name)
        result.update(actual_sp_degree=get_sequence_parallel_world_size(), actual_dp_degree=get_data_parallel_world_size(),
            gpu=torch.cuda.get_device_name(rank), gpu_uuid=str(torch.cuda.get_device_properties(rank).uuid),
            torch=torch.__version__, cuda=torch.version.cuda)
        if result['actual_sp_degree'] != world or result['actual_dp_degree'] != 1:
            raise ValueError('official runner did not establish the requested SP/DP groups')
        torch.cuda.reset_peak_memory_stats(rank)
        result['failed_stage'] = 'native_inference'
        started = time.monotonic()
        output, timings = runner.run(effective)
        torch.cuda.synchronize()
        result['cold_rank_request_wall_seconds'] = time.monotonic() - started
        if len(timings) != 1 or any(not math.isfinite(float(v)) or float(v) < 0 for v in timings):
            raise ValueError('native runner returned invalid single-request timing')
        result['failed_stage'] = 'validate_output'
        # Official DP=1 returns the complete native DiffusionOutput on all SP ranks.
        # Never gather full videos through object collectives merely for reporting.
        result.update(output=native.summarize_video(output.videos, request),
            official_runner_timings_seconds=[float(v) for v in timings],
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(rank),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(rank),
            scheduler_config=dict(pipeline.scheduler.config),
            effective_request={k: effective[k] for k in request if k in effective},
            attention_backend_observation=native.attention_observation(pipeline, runner.config.attention_backend),
            loaded_source_sha256=native.loaded_sources(source_root, manifest))
        del output
        native.source.verify_source(source_root, result['source_manifest_path'], result['source']['manifest_sha256'])
        result['failed_stage'] = 'cleanup'
    except Exception as exc:
        execution_error = exc
        raise
    finally:
        try:
            if initialized:
                runner.cleanup()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            result['cleanup_completed'] = True
        except Exception as exc:
            result['cleanup_completed'] = False
            result['cleanup_error'] = repr(exc)
            if execution_error is None:
                raise
    result.pop('failed_stage', None)
    result.update(status='native_rank_completed', planned_returncode=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--world-size', type=int, choices=(2, 4), required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--models', type=Path, default=native.DEFAULT_MODELS)
    parser.add_argument('--models-sha256', default=native.DEFAULT_MODELS_SHA256)
    parser.add_argument('--model', choices=tuple(native.CLASSES), required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--shape', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    args = parser.parse_args(argv)
    if not 30 <= args.timeout_seconds <= 14400:
        parser.error('timeout must be30..14400 seconds')
    rank = topology(args.world_size, os.environ) if args.command == 'run' else 0
    output = args.output_dir.resolve()
    if args.command == 'run' and not output.is_relative_to(Path('/shared/output/DITSERVE')):
        raise ValueError('GPU evidence must be below the project output root')
    for path in (args.source_root, args.model_root):
        if output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output):
            raise ValueError('output must be disjoint from source and model inputs')
    stop = threading.Event()
    result = dict(schema='ditserve.external-xdit-rank.v1', status='native_rank_failed', rank=rank,
        local_rank=rank, world_size=args.world_size, planned_returncode=1, cleanup_completed=False,
        qualification_eligible=False, allows=[], native_dtype='bfloat16', source_manifest_path=str(args.source_manifest),
        adapter={'distributed_sha256': native.source.digest(Path(__file__).read_bytes()),
                 'native_helper_sha256': native.source.digest(Path(native.__file__).read_bytes()),
                 'source_verifier_sha256': native.source.digest(Path(native.source.__file__).read_bytes()),
                 'override': 'Only checkpoint path and requested Ulysses degree; original runner owns inference and communication.'})
    rank_dir = None
    deadline = time.monotonic() + args.timeout_seconds
    try:
        if args.command == 'run':
            result['initial_storage'] = native.check_storage()
            output.mkdir(parents=True, exist_ok=True)
            if (output/'summary.json').exists():
                raise FileExistsError('previous distributed summary must be preserved')
            rank_dir = output/f'rank-{rank}'
            rank_dir.mkdir(exist_ok=False)
            def watch():
                while not stop.wait(5):
                    try:
                        native.check_storage()
                        if time.monotonic() >= deadline:
                            raise TimeoutError('rank diagnostic deadline exceeded')
                    except Exception as exc:
                        print('xDiT rank guard stopped execution: ' + repr(exc), file=sys.stderr, flush=True)
                        os._exit(75)
            threading.Thread(target=watch, daemon=True).start()
        result['failed_stage'] = 'verify_source_and_model'
        manifest = native.source.verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
        protocol = native.bound_json(args.protocol, args.protocol_sha256)
        request = native.request_plan(protocol, args.model, args.shape, 'formal', args.prompt, args.seed)
        identity = native.verify_model(args.model, args.model_root, args.models, args.models_sha256)
        config = distributed_config(args.model, request, identity, rank_dir or output, args.world_size)
        result.update(request=request, model_identity=identity, runner_config=config, protocol_sha256=args.protocol_sha256,
            source=dict(commit=native.source.COMMIT, manifest_sha256=args.source_manifest_sha256,
                        archive_sha256=native.source.ARCHIVE_SHA256, files=len(manifest['files'])))
        if args.command == 'plan':
            result.update(status='planned', planned_returncode=0)
        else:
            publish_json(rank_dir/'started.json', result)
            with contextlib.redirect_stdout(sys.stderr):
                run_rank(args.model, request, identity, config, args.source_root, manifest, rank, args.world_size, result)
            result['runtime_packages'] = {}
            for name in ('torch', 'diffusers', 'transformers', 'cache-dit', 'numpy', 'yunchang'):
                try: result['runtime_packages'][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError: result['runtime_packages'][name] = None
    except Exception as exc:
        result.update(status='native_rank_failed', planned_returncode=1, error=repr(exc),
                      traceback=traceback.format_exc()[-10000:])
    if rank_dir is not None:
        native.check_storage()
        publish_json(rank_dir/'summary.json', result)
    if args.command == 'run' and rank == 0 and rank_dir is not None:
        try:
            paths = [output/f'rank-{i}'/'summary.json' for i in range(args.world_size)]
            while not all(p.is_file() for p in paths):
                if time.monotonic() >= deadline: raise TimeoutError('some ranks did not publish cleanup receipts')
                time.sleep(.25)
            rows = [native.source.read_json(p.read_bytes()) for p in paths]
            summary = aggregate(rows, args.world_size)
        except Exception as exc:
            summary = dict(schema='ditserve.external-xdit-distributed.v1', status='native_sp_request_failed',
                requested_requests=1, completed_requests=0, world_size=args.world_size,
                error=repr(exc), qualification_eligible=False, allows=[])
            result['planned_returncode'] = 1
        native.check_storage()
        publish_json(output/'summary.json', summary)
        print(json.dumps({'rank':rank,'status':summary['status'],'summary':str(output/'summary.json')}),flush=True)
    elif args.command == 'plan':
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    stop.set()
    return result['planned_returncode']


if __name__ == '__main__':
    raise SystemExit(main())
