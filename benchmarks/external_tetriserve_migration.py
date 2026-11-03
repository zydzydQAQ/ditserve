"""Real TetriServe five-tensor NCCL migration and shared SD3.5 continuation.

Two real GPU processes use the unchanged official LatentManager/LatentFuture,
request queues, sender threads and directed NCCL groups. Rank 0 executes steps
0:2; rank 1 receives all five tensors and executes 2:28. Its independent normal
shared-pipeline 28-step result is the numerical reference. This is SP1 changing
ranks, not SP2 inference, HTTP/ZMQ serving or full-system qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys
import time
import traceback

import external_tetriserve_port as port

NAMES = ('prompt_embeds', 'pooled_prompt_embeds', 'latents', 'timesteps', 'sigmas')
TRANSPORT = ('tetriserve/light_ray/latent_manager.py',
             'tetriserve/light_ray/latent_future.py', 'tetriserve/distributed/globals.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def validate_packet(future, request_record, expected_request, descriptors):
    if (future.sub_req_id != 1 or future.src_rank != 0 or tuple(future.step_range or ()) != (0, 2)
            or request_record != expected_request or tuple(future.tensors_info) != NAMES
            or tuple(descriptors) != NAMES):
        raise ValueError('native future identity, cursor, request or five-tensor order differs')
    for name in NAMES:
        info, row = future.tensors_info[name], descriptors[name]
        if list(info.shape) != row['shape'] or str(info.dtype) != row['dtype']:
            raise ValueError('native tensor metadata differs: ' + name)


def wire_digests(values):
    import torch
    if tuple(values) != NAMES:
        raise ValueError('exactly the original five named tensors are required')
    result = {}
    for name, tensor in values.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not torch.isfinite(tensor).all().item():
            raise ValueError('native payload must contain finite actual CUDA tensors')
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        result[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return result


def verify_derived(root, expected):
    root = Path(root).resolve(strict=True)
    manifest_path = root / 'PORT_MANIFEST.json'
    if sha(manifest_path) != expected:
        raise ValueError('derived source manifest differs')
    manifest = port.upstream.read_json(manifest_path.read_bytes())
    if manifest.get('schema') != port.PORT_SCHEMA or manifest.get('upstream_commit') != port.upstream.COMMIT:
        raise ValueError('derived source identity differs')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('linked derived source')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'PORT_MANIFEST.json'}:
        raise ValueError('derived source inventory differs')
    for name, expected_sha in manifest['files'].items():
        path = root / name
        if not path.is_relative_to(root) or path.resolve() != path or sha(path) != expected_sha:
            raise ValueError('derived source bytes or path differ')
    for name in TRANSPORT:
        if name not in manifest['preserved_native_mechanisms'] or manifest['files'][name] != manifest['preserved_native_mechanisms'][name]:
            raise ValueError('native transport must remain byte-identical: ' + name)
    return manifest


def upstream_engine_config(model):
    # Actual official dataclasses, not a stand-in runner or configuration module.
    import torch
    from tetriserve.xfuser.config.config import (
        EngineConfig, ModelConfig, RuntimeConfig, ParallelConfig, DataParallelConfig,
        SequenceParallelConfig, PipeFusionParallelConfig, TensorParallelConfig, FastAttnConfig)
    return EngineConfig(ModelConfig(model), RuntimeConfig(dtype=torch.bfloat16),
        ParallelConfig(DataParallelConfig(), SequenceParallelConfig(),
                       PipeFusionParallelConfig(), TensorParallelConfig()), FastAttnConfig())


def worker(rank, args, derived, config_path, config_sha, request_record,
           request_queues, packet_queue, report_queue):
    # Nothing in this module imports round/xfuser at module import time. Spawned
    # workers first bind the exact derived adapters and official transport.
    result = dict(rank=rank, status='failed', stage='source_binding',
                  sender_thread_stopped=False, distributed_closed=False)
    manager, pipeline, owner = None, None, None
    error = None
    try:
        verify_derived(derived['root'], derived['manifest_sha256'])
        sys.path[:0] = [derived['root'], str(Path(__file__).resolve().parent), str(args.shared_python_root)]
        os.environ['DITSERVE_TETRI_PORT_CONFIG'] = config_path
        os.environ['DITSERVE_TETRI_PORT_CONFIG_SHA256'] = config_sha
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        import torch
        import torch.distributed as dist
        import external_sd35_round as round_adapter
        from external_tetriserve_bridge import TetriSharedSD35Pipeline
        from external_sd35_round_check import compare_tensors, tensor_summary
        from tetriserve.utils import set_device
        from tetriserve.distributed.globals import PROCESS_GROUP
        from tetriserve.light_ray.latent_manager import LatentManager
        from tetriserve.light_ray.latent_future import LatentFuture
        for name, module in list(sys.modules.items()):
            if name == 'tetriserve' or name.startswith('tetriserve.'):
                origin = getattr(module, '__file__', None)
                if origin and not Path(origin).resolve().is_relative_to(Path(derived['root'])):
                    raise ValueError('a different official transport tree is imported')
        if torch.cuda.device_count() != 2:
            raise ValueError('diagnostic requires exactly two visible GPUs')
        torch.cuda.set_device(rank)
        set_device(gpu=True)
        result['gpu'] = dict(name=torch.cuda.get_device_name(rank),
                            uuid=str(torch.cuda.get_device_properties(rank).uuid), device=rank)
        result['stage'] = 'original_process_groups'
        dist.init_process_group('nccl', init_method='file://' + str(args.output_dir / 'rendezvous'),
                                rank=rank, world_size=2, timeout=timedelta(seconds=180))
        owner = PROCESS_GROUP
        owner.generate_all_process_groups(2, skip_latent_transfer=False)
        if (set(owner.all_process_groups_map) != {'[0]', '[1]', '[0, 1]'}
                or set(owner.latent_transfer_groups) != {'[0, 1]', '[1, 0]'}
                or owner.latent_transfer_groups['[0, 1]'] is owner.latent_transfer_groups['[1, 0]']):
            raise ValueError('official general/directed communicator inventory differs')
        for group in owner.latent_transfer_groups.values():
            if dist.get_backend(group) != 'nccl' or dist.get_world_size(group) != 2:
                raise ValueError('native directed transfer must use real two-rank NCCL groups')
        control_group = owner.get_process_group([0, 1], device='cpu')
        manager = LatentManager(rank, request_queues, auto_start=True)
        result['stage'] = 'load_verified_shared_model'
        pipeline = TetriSharedSD35Pipeline.from_pretrained(round_adapter.SD35_MODEL,
            upstream_engine_config(round_adapter.SD35_MODEL), torch.bfloat16)
        request = round_adapter.ImageRequest(**request_record)
        if pipeline.runtime.pool.current_ranks != (rank,):
            raise ValueError('each real model must execute in its official singleton group')
        result['model_provenance'] = pipeline.provenance
        result['backend'] = pipeline.runtime.info()
        dist.barrier(group=control_group)
        reference, reference_decoded = None, None
        if rank == 1:
            result['stage'] = 'independent_normal_shared_28_step_reference'
            with torch.inference_mode():
                reference = pipeline.pipe(prompt=request.prompt, negative_prompt=request.negative_prompt,
                    height=request.height, width=request.width, num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale, max_sequence_length=request.max_sequence_length,
                    generator=torch.Generator(device='cpu').manual_seed(request.seed), output_type='latent').images
                reference_decoded = pipeline.raw.vae.decode(reference / pipeline.raw.vae.config.scaling_factor
                    + pipeline.raw.vae.config.shift_factor, return_dict=False)[0]
                torch.cuda.synchronize()
            result['reference_steps'] = 28
        # Do not let the official 300-second latent TTL run during reference generation.
        dist.barrier(group=control_group)
        if rank == 0:
            result['stage'] = 'actual_shared_round_0_2'
            state = pipeline.advance(pipeline.prepare(request), 2)
            values = dict(zip(NAMES, state.transferred_tensors()))
            descriptors = wire_digests(values)
            future = manager.save_latent(1, values, save_tensors=True, step_range=(0, 2))
            # Native save clones on this producer stream; make those clones ready
            # before the unchanged sender thread receives the queue notification.
            # This diagnostic does not qualify asynchronous producer-stream use.
            torch.cuda.synchronize()
            result['producer_synchronized_before_notification'] = True
            if not isinstance(future, LatentFuture) or not future.ready():
                raise ValueError('original manager did not save a ready native future')
            validate_packet(future, request_record, request_record, descriptors)
            packet_queue.put((future, request_record, descriptors))
            result.update(wire_sent=descriptors, executed_interval=[0, 2], stage='original_sender_thread_transfer')
        else:
            result['stage'] = 'original_native_future_receive'
            future, transferred_request, descriptors = packet_queue.get(timeout=180)
            if not isinstance(future, LatentFuture):
                raise ValueError('metadata queue did not carry the actual official LatentFuture')
            validate_packet(future, transferred_request, request_record, descriptors)
            # This is strictly the remote branch: upstream forwards timeout to
            # Torch Work.wait, whose 2.10 API accepts timedelta (not float).
            values = future.get(timeout=timedelta(seconds=120))
            received = wire_digests(values)
            if received != descriptors:
                raise ValueError('native five-tensor migration changed payload bytes')
            result.update(wire_received=received, wire_bytes_exact=True, received_cursor=2)
        dist.barrier(group=control_group)
        if rank == 0:
            deadline = time.monotonic() + 5
            while manager.has_local_tensor(1) and time.monotonic() < deadline:
                time.sleep(.01)
            if manager.has_local_tensor(1):
                raise ValueError('native sender did not release its transferred stored payload')
            result['source_payload_released'] = True
        else:
            result['stage'] = 'actual_shared_round_2_28_on_new_rank'
            restored = round_adapter.RoundState(request, *(values[name] for name in NAMES), cursor=2)
            resumed = pipeline.advance(restored, 28).latents
            with torch.inference_mode():
                decoded = pipeline.raw.vae.decode(resumed / pipeline.raw.vae.config.scaling_factor
                    + pipeline.raw.vae.config.shift_factor, return_dict=False)[0]
                torch.cuda.synchronize()
            result.update(executed_interval=[2, 28], resumed_steps=26,
                comparisons=dict(latent=compare_tensors(reference, resumed),
                                 decoded=compare_tensors(reference_decoded, decoded)),
                output=dict(reference_latent=tensor_summary(reference), resumed_latent=tensor_summary(resumed),
                            reference_decoded=tensor_summary(reference_decoded), resumed_decoded=tensor_summary(decoded)))
            result['numerical_check_passed'] = all(row['passed'] for row in result['comparisons'].values())
            if not result['numerical_check_passed']:
                raise ValueError('cross-rank continuation differs from the normal 28-step reference')
        dist.barrier(group=control_group)
        result.update(status='passed', stage='completed', peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(rank))
    except Exception as exc:
        error = exc
        result.update(error=repr(exc), traceback=traceback.format_exc()[-14000:])
    finally:
        try:
            if manager is not None:
                manager.stop_sender_thread()
                manager.sender_thread.join(timeout=10)
                if manager.sender_thread.is_alive():
                    raise RuntimeError('original sender thread failed to stop')
                result['sender_thread_stopped'] = True
            if pipeline is not None:
                pipeline.runtime.close()  # borrowed groups remain owned by TetriServe
            if owner is not None:
                owner.destroy()
            import torch.distributed as dist
            if dist.is_initialized():
                # The original owner leaves general Gloo groups to default teardown.
                dist.destroy_process_group()
            result['distributed_closed'] = not dist.is_initialized()
        except Exception as exc:
            result.update(status='failed', cleanup_error=repr(exc))
            error = error or exc
        report_queue.put(result)
    if error is not None:
        raise RuntimeError('native migration worker failed; see rank diagnostic') from error


def numerical_receipt(row):
    """Recheck small numerical evidence independently of its aggregate flag."""
    comparisons, output = row.get('comparisons'), row.get('output')
    if (not isinstance(comparisons, dict) or set(comparisons) != {'latent', 'decoded'}
            or not isinstance(output, dict) or set(output) != {
                'reference_latent', 'resumed_latent', 'reference_decoded', 'resumed_decoded'}):
        return False
    for kind, shape in (('latent', [1, 16, 64, 64]), ('decoded', [1, 3, 512, 512])):
        for prefix in ('reference_', 'resumed_'):
            value = output[prefix + kind]
            if (not isinstance(value, dict) or value.get('shape') != shape
                    or value.get('dtype') != 'torch.bfloat16' or value.get('finite') is not True
                    or value.get('retained_array') is not False
                    or not isinstance(value.get('sha256'), str) or len(value['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in value['sha256'])):
                return False
            if any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k])
                   for k in ('minimum', 'maximum')) or value['minimum'] > value['maximum']:
                return False
        comparison = comparisons[kind]
        if (not isinstance(comparison, dict) or comparison.get('passed') is not True
                or type(comparison.get('bitwise_equal')) is not bool
                or comparison.get('elements') != math.prod(shape)):
            return False
        for key in ('max_abs_error', 'rmse', 'relative_l2_error', 'cosine_similarity', 'atol', 'rtol'):
            if type(comparison.get(key)) not in (int, float) or not math.isfinite(comparison[key]):
                return False
        if comparison['atol'] != 1e-5 or comparison['rtol'] != 1e-5:
            return False
        reference = output['reference_' + kind]
        largest_allowed_error = 1e-5 + 1e-5 * max(abs(reference['minimum']), abs(reference['maximum']))
        if (not 0 <= comparison['rmse'] <= comparison['max_abs_error'] <= largest_allowed_error
                or comparison['relative_l2_error'] < 0
                or not -1.000000000001 <= comparison['cosine_similarity'] <= 1.000000000001):
            return False
        if comparison['bitwise_equal'] and (
                comparison['max_abs_error'] != 0 or comparison['relative_l2_error'] != 0
                or reference['sha256'] != output['resumed_' + kind]['sha256']):
            return False
    return True


def assess(records):
    if len(records) != 2 or {row.get('rank') for row in records} != {0, 1}:
        return False
    rows = {row['rank']:row for row in records}
    return (all(row.get('status') == 'passed' and row.get('sender_thread_stopped') is True
                and row.get('distributed_closed') is True for row in rows.values())
            and rows[0].get('gpu', {}).get('uuid') != rows[1].get('gpu', {}).get('uuid')
            and all(row.get('gpu', {}).get('uuid') for row in rows.values())
            and rows[0].get('executed_interval') == [0, 2] and rows[1].get('executed_interval') == [2, 28]
            and rows[0].get('source_payload_released') is True and rows[1].get('wire_bytes_exact') is True
            and rows[0].get('wire_sent') == rows[1].get('wire_received')
            and set(rows[0].get('wire_sent', {})) == set(NAMES)
            and rows[1].get('received_cursor') == 2
            and rows[1].get('reference_steps') == 28 and rows[1].get('resumed_steps') == 26
            and rows[1].get('numerical_check_passed') is True and numerical_receipt(rows[1]))


def run_workers(args, derived, config_path, config_sha, request_record):
    import torch.multiprocessing as mp
    from external_sd35_round_check import check_storage
    context = mp.get_context('spawn')
    requests = [context.Queue() for _ in range(2)]
    packets, reports = context.Queue(), context.Queue()
    records, failure = [], None
    children = None
    start, last_storage = time.monotonic(), 0.
    try:
        children = mp.spawn(worker, args=(args, derived, str(config_path), config_sha, request_record,
                                         requests, packets, reports), nprocs=2, join=False)
        while True:
            if time.monotonic() - last_storage >= 5:
                check_storage()
                last_storage = time.monotonic()
            if time.monotonic() - start > args.timeout_seconds:
                raise TimeoutError('two-rank diagnostic deadline exceeded')
            done = children.join(timeout=1)
            while True:
                try:
                    records.append(reports.get_nowait())
                except queue.Empty:
                    break
            if done:
                break
    except Exception as exc:
        failure = repr(exc)
    finally:
        if children is not None:
            for child in children.processes:
                if child.is_alive():
                    child.terminate()
            for child in children.processes:
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
        while True:
            try:
                records.append(reports.get_nowait())
            except queue.Empty:
                break
        for channel in [*requests, packets, reports]:
            channel.close()
            channel.join_thread()
    return records, failure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    for name in ('source-root','source-manifest','shared-python-root','model-root','model-manifest','protocol','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('source-manifest-sha256','model-manifest-sha256','protocol-sha256','prompt'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not 120 <= args.timeout_seconds <= 3600:
        parser.error('timeout must be between 120 and 3600 seconds')
    args.output_dir = args.output_dir.resolve()
    result = dict(schema='ditserve.tetriserve-native-migration-check.v1', status='failed', stage='binding',
        native_transport_attempted=False, native_transport_executed=False, model_loaded=False, numerical_check_passed=False,
        cross_rank_sp1_verified=False, sp2_verified=False, serving_executed=False,
        http_zmq_executed=False, scheduler_executed=False, performance_measurements=False,
        qualification_eligible=False, allows=[], ranks=[], intervals=[[0,2],[2,28]], reference_steps=28,
        scope='Original native five-tensor transport plus shared SP1 continuation on another GPU; not full TetriServe')
    created = False
    try:
        from external_sd35_round_check import request_plan, check_storage
        import external_sd35_round as round_adapter
        if sha(args.protocol) != args.protocol_sha256:
            raise ValueError('formal protocol differs')
        request = request_plan(port.upstream.read_json(args.protocol.read_bytes()), args.prompt, args.seed)
        if request.steps != 28 or request.height != 512 or request.width != 512:
            raise ValueError('initial migration check requires the frozen 512x512/28-step image')
        shared = round_adapter.verify_shared_sources(args.shared_python_root)
        args.shared_python_root = shared
        raw_manifest = args.model_manifest.read_bytes()
        manifest = port.upstream.read_json(raw_manifest)
        if (hashlib.sha256(raw_manifest).hexdigest() != args.model_manifest_sha256
                or manifest.get('model') != round_adapter.SD35_MODEL
                or manifest.get('revision') != round_adapter.SD35_REVISION):
            raise ValueError('official model manifest identity differs')
        source_plan, _, _ = port.plan(args.source_root, args.source_manifest, args.source_manifest_sha256)
        result.update(request=asdict(request), source_manifest_sha256=args.source_manifest_sha256,
            shared_worker_sha256=round_adapter.FROZEN_WORKER_SHA256,
            model_manifest_sha256=args.model_manifest_sha256, protocol_sha256=args.protocol_sha256,
            native_transport_sources={name:source_plan['files'][name] for name in TRANSPORT})
        if args.command == 'plan':
            result.update(status='planned', model_weights_verified=False)
            print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
            return 0
        if not args.output_dir.is_relative_to(Path('/shared/output/DITSERVE')):
            raise ValueError('GPU diagnostic output must be under /shared/output/DITSERVE')
        for path in (args.source_root, args.shared_python_root, args.model_root, Path(__file__).resolve().parent):
            if args.output_dir.is_relative_to(path.resolve()) or path.resolve().is_relative_to(args.output_dir):
                raise ValueError('output must be disjoint from all bound input trees')
        check_storage()
        import torch
        if torch.cuda.device_count() != 2:
            raise ValueError('actual allocation must expose exactly two GPUs')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created = True
        derived = port.apply(args.source_root, args.source_manifest, args.source_manifest_sha256, args.output_dir/'derived')
        config = dict(schema='ditserve.tetriserve-sd35-port.v1', model=round_adapter.SD35_MODEL,
            revision=round_adapter.SD35_REVISION, shared_worker_sha256=round_adapter.FROZEN_WORKER_SHA256,
            shared_python_root=str(shared), model_path=str(args.model_root.resolve()),
            model_manifest=str(args.model_manifest.resolve()), model_manifest_sha256=args.model_manifest_sha256,
            guidance_scale=request.guidance_scale, max_sequence_length=request.max_sequence_length,
            attention='math', resolutions=[[512,512]], derived_manifest=str(Path(derived['root'])/'PORT_MANIFEST.json'),
            derived_manifest_sha256=derived['manifest_sha256'], diagnostic_only=True, server_launch_allowed=False)
        config_path = args.output_dir/'port-config.json'
        write_json(config_path, config)
        result.update(stage='two_rank_native_transport_and_model', derived_manifest_sha256=derived['manifest_sha256'],
                      port_config_sha256=sha(config_path))
        records, failure = run_workers(args, derived, config_path, sha(config_path), asdict(request))
        result['ranks'] = records
        result['native_transport_attempted'] = any('wire_sent' in row or 'wire_received' in row for row in records)
        result['native_transport_executed'] = any(row.get('wire_bytes_exact') is True and 'wire_received' in row for row in records)
        result['model_loaded'] = any('model_provenance' in row for row in records)
        passed = failure is None and assess(records)
        result.update(status='numerical_passed' if passed else 'numerical_failed', stage='completed',
                      numerical_check_passed=passed, cross_rank_sp1_verified=passed)
        if failure:
            result['worker_error'] = failure
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-16000:])
    if created:
        from external_sd35_round_check import check_storage
        check_storage()
        write_json(args.output_dir/'summary.json', result)
        print(json.dumps(dict(status=result['status'], summary=str(args.output_dir/'summary.json'),
                              sha256=sha(args.output_dir/'summary.json'), qualification_eligible=False)), flush=True)
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result['numerical_check_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
