"""One-GPU numerical check of the actual shared SD3.5 round adapter.

Compares the shared pipeline's normal 28-step call, uninterrupted round execution,
and a two-step/five-tensor restore/twenty-six-step execution at 512x512. Only small
JSON summaries and tensor hashes are retained. This does not run TetriServe's
scheduler/worker/transport and cannot qualify a serving comparison.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import external_sd35_round as adapter

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / 'configs/benchmarks/formal-mixed-v1.json'
DEFAULT_MODEL_MANIFEST_SHA256 = 'ea082cea29ff607d58dca53913dae779834664d4832c10bf938bdb7410c6d424'
ATOL = 1e-5
RTOL = 1e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_storage():
    try:
        proc = subprocess.run(['df', '-BG', '/shared'], capture_output=True, text=True, timeout=20)
        fields = proc.stdout.splitlines()[-1].split()
        used, percent = int(fields[2].rstrip('G')), int(fields[4].rstrip('%'))
        if proc.returncode or used > 790 or percent >= 80:
            raise ValueError('shared storage stop line')
    except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
        raise SystemExit(75)
    return {'used_gib': used, 'percent': percent}


def storage_watchdog(stop):
    while not stop.wait(30):
        try:
            check_storage()
        except BaseException:
            # Only this diagnostic process exits. No other job is signalled.
            os._exit(75)


def request_plan(protocol, prompt, seed, shape_id='sd35-512'):
    if protocol.get('schema') != 'ditserve.formal-protocol.v1' or protocol.get('id') != 'formal-mixed-v1':
        raise ValueError('formal protocol identity differs')
    sizes = {'sd35-512': 512, 'sd35-1024': 1024}
    if shape_id not in sizes:
        raise ValueError('unsupported frozen SD3.5 shape')
    size = sizes[shape_id]
    shape = protocol['shapes'][shape_id]
    expected = {'model': 'sd35', 'kind': 'image', 'height': size, 'width': size,
                'frames': 1, 'steps': 28, 'guidance_scale': 4.5, 'scope': 'formal'}
    if any(shape.get(k) != v for k, v in expected.items()):
        raise ValueError('frozen SD3.5 ' + str(size) + ' shape differs')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('a nonempty explicit prompt is required')
    # Matches frozen formal_benchmark.shape_spec for image requests.
    return adapter.ImageRequest(prompt, negative_prompt='blurry, low quality',
                                height=size, width=size, steps=28, seed=seed,
                                guidance_scale=4.5, max_sequence_length=256)


def tensor_summary(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    if not torch.isfinite(value).all().item():
        raise ValueError('nonfinite tensor')
    description = {'shape': list(value.shape), 'dtype': str(value.dtype)}
    h = hashlib.sha256(json.dumps(description, sort_keys=True).encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return {**description, 'sha256': h.hexdigest(), 'finite': True,
            'minimum': float(value.min()), 'maximum': float(value.max()), 'retained_array': False}


def compare_tensors(reference, actual, *, atol=ATOL, rtol=RTOL):
    import torch
    if tuple(reference.shape) != tuple(actual.shape) or reference.dtype != actual.dtype:
        raise ValueError('numerical comparison requires identical shape and dtype')
    if (not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0):
        raise ValueError('invalid comparison tolerance')
    a, b = reference.detach().double().cpu(), actual.detach().double().cpu()
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise ValueError('nonfinite comparison input')
    difference = b - a
    scale = float(torch.linalg.vector_norm(a))
    other_scale = float(torch.linalg.vector_norm(b))
    error_norm = float(torch.linalg.vector_norm(difference))
    denominator = scale * other_scale
    cosine = (float((a * b).sum()) / denominator if denominator > 0
              else 1.0 if error_norm == 0 else 0.0)
    return {'passed': bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            'bitwise_equal': bool(torch.equal(reference, actual)),
            'max_abs_error': float(difference.abs().max()),
            'rmse': float((difference.square().mean()).sqrt()),
            'relative_l2_error': error_norm / max(scale, 1e-12),
            'cosine_similarity': cosine, 'atol': atol, 'rtol': rtol,
            'elements': a.numel()}


def execute(request, args, result):
    import torch
    import diffusers
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('numerical check requires exactly one GPU and one rank')
    if diffusers.__version__ != '0.36.0':
        raise ValueError('numerical check requires the fixed-image diffusers 0.36.0 semantics')
    result['runtime'] = {'torch': torch.__version__, 'diffusers': diffusers.__version__,
                         'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                         'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid)}
    rounder = None
    try:
        result['stage'] = 'load_verified_model'
        rounder = adapter.SD35RoundPipeline.from_pretrained(
            args.model_root, shared_python_root=args.shared_python_root,
            model_manifest_path=args.model_manifest,
            model_manifest_sha256=args.model_manifest_sha256,
            device='cuda', attention='math', guidance_scale=request.guidance_scale,
            max_sequence_length=request.max_sequence_length)
        result['model_weights_loaded'] = True
        result['provenance'] = rounder.provenance
        result['backend'] = rounder.runtime.info()
        with torch.inference_mode():
            result['stage'] = 'normal_shared_pipeline_28_steps'
            reference = rounder.pipe(prompt=request.prompt, negative_prompt=request.negative_prompt,
                height=request.height, width=request.width, num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale, max_sequence_length=request.max_sequence_length,
                generator=torch.Generator(device='cpu').manual_seed(request.seed), output_type='latent').images
            torch.cuda.synchronize()
            result['completed_trajectories'] = 1
            result['stage'] = 'uninterrupted_round_28_steps'
            initial = rounder.prepare(request)
            complete = rounder.advance(initial, 28)
            uninterrupted = complete.latents
            torch.cuda.synchronize()
            result['completed_trajectories'] = 2
            result['stage'] = 'segmented_round_first_2_steps'
            initial = rounder.prepare(request)
            first = rounder.advance(initial, 2)
            # Retain only the five wire tensors and explicit request/cursor; no
            # scheduler object or hidden prologue state is carried across restore.
            packet = tuple(t.detach().clone() for t in first.transferred_tensors())
            restored = adapter.RoundState(request, *packet, cursor=2)
            resumed = adapter.SD35RoundPipeline(rounder.runtime,
                guidance_scale=request.guidance_scale, max_sequence_length=request.max_sequence_length,
                provenance=rounder.provenance)
            result['stage'] = 'segmented_round_restored_26_steps'
            segmented = resumed.advance(restored, 28).latents
            torch.cuda.synchronize()
            result['completed_trajectories'] = 3
            result['stage'] = 'latent_comparison'
            outputs = {'reference': reference, 'round28': uninterrupted, 'round2_26': segmented}
            result['latents'] = {name: tensor_summary(value) for name, value in outputs.items()}
            result['latent_comparisons'] = {
                'reference_vs_round28': compare_tensors(reference, uninterrupted),
                'reference_vs_round2_26': compare_tensors(reference, segmented),
                'round28_vs_round2_26': compare_tensors(uninterrupted, segmented),
            }
            result['stage'] = 'decode_and_compare'
            decoded = {}
            for name, value in outputs.items():
                decoded[name] = rounder.raw.vae.decode(
                    value / rounder.raw.vae.config.scaling_factor + rounder.raw.vae.config.shift_factor,
                    return_dict=False)[0]
            result['decoded'] = {name: tensor_summary(value) for name, value in decoded.items()}
            result['decoded_comparisons'] = {
                'reference_vs_round28': compare_tensors(decoded['reference'], decoded['round28']),
                'reference_vs_round2_26': compare_tensors(decoded['reference'], decoded['round2_26']),
                'round28_vs_round2_26': compare_tensors(decoded['round28'], decoded['round2_26']),
            }
            result['numerical_check_passed'] = all(
                item['passed'] for key in ('latent_comparisons', 'decoded_comparisons')
                for item in result[key].values())
            result['stage'] = 'completed'
    finally:
        if rounder is not None:
            rounder.runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('--shared-python-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--model-manifest-sha256', default=DEFAULT_MODEL_MANIFEST_SHA256)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--shape-id', choices=('sd35-512', 'sd35-1024'), default='sd35-512')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = {'schema': 'ditserve.external-sd35-round-check.v1', 'status': 'not_started',
              'stage': 'binding', 'planned_trajectories': 3, 'completed_trajectories': 0,
              'model_weights_loaded': False, 'numerical_check_passed': False,
              'qualification_eligible': False, 'allows': [], 'serving_executed': False,
              'tetriserve_scheduler_executed': False, 'cross_gpu_transfer_tested': False,
              'performance_measurements': False,
              'scope': 'One-GPU shared-backend numerical check, not a full TetriServe reproduction'}
    output, created = args.output_dir.resolve(), False
    stop = threading.Event()
    if args.command == 'run':
        check_storage()
        threading.Thread(target=storage_watchdog, args=(stop,), daemon=True).start()
    try:
        if digest(args.protocol) != args.protocol_sha256:
            raise ValueError('protocol SHA256 differs')
        protocol = json.loads(args.protocol.read_bytes())
        request = request_plan(protocol, args.prompt, args.seed, args.shape_id)
        adapter.verify_shared_sources(args.shared_python_root)
        if digest(args.model_manifest) != args.model_manifest_sha256:
            raise ValueError('model manifest SHA256 differs')
        result.update(status='planned', shape_id=args.shape_id, request=dict(vars(request)), intervals=[[0, 2], [2, 28]],
            protocol_sha256=args.protocol_sha256, model_manifest_sha256=args.model_manifest_sha256,
            adapter_sha256=digest(adapter.__file__), checker_sha256=digest(__file__),
            shared_worker_sha256=adapter.FROZEN_WORKER_SHA256,
            tolerance={'atol': ATOL, 'rtol': RTOL, 'scope': 'predeclared diagnostic; not a paper quality criterion'},
            generator_device='cpu', attention='sdpa_math', arrays_retained=False)
        if args.command == 'run':
            if not output.is_relative_to(Path('/shared/output/DITSERVE')):
                raise ValueError('diagnostic output must be under /shared/output/DITSERVE')
            check_storage()
            output.mkdir(parents=True, exist_ok=False)
            created = True
            with (output / 'started.json').open('x') as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            sys.dont_write_bytecode = True
            with contextlib.redirect_stdout(sys.stderr):
                execute(request, args, result)
            adapter.verify_shared_sources(args.shared_python_root)
            result['status'] = 'numerical_check_passed' if result['numerical_check_passed'] else 'numerical_check_failed'
    except Exception as exc:
        result.update(status='numerical_check_failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-8000:])
    finally:
        if args.command == 'run':
            check_storage()
        if created:
            with (output / 'summary.json').open('x') as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stop.set()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result['status'] in ('planned', 'numerical_check_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
