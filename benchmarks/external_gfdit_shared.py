"""Source-bound GF-DiT control plane with an actual shared SD3.5 SP1 executor.

This first port checks the original EDF/DAG against real xfuser inference. It
does not implement cross-rank migration, native vLLM workers, serving replay or
performance qualification. No fake vLLM module is installed. Only imports and
the logging provider change in the derived control-plane sources; their ASTs
are checked independently of those import changes.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import external_gfdit as upstream
import external_sd35_round as round_adapter

NAMESPACE = 'gfdit_shared_runtime'
PREFIX = 'vllm_omni.diffusion.runtime_v2'
SOURCE_PREFIX = PREFIX.replace('.', '/')
PORT_SCHEMA = 'ditserve.gfdit-shared-source.v1'
CONTROL_FILES = (
    '_env.py', 'protocol.py', 'codec.py', 'interfaces.py', 'topology.py',
    'cost_model.py', 'scheduler.py', 'adapters/common.py',
    'policies/__init__.py', 'policies/common.py', 'policies/disaggregate.py',
    'policies/edf_greedy.py', 'policies/edf_best_fit.py', 'policies/fcfs.py',
    'policies/sjf.py', 'policies/srtf.py', 'policies/wave_stress.py',
)
MIGRATION_FILES = ('migration_engine.py', 'data_plane.py', 'multiproc_worker.py')
STAGES = ('text_encode', 'dit_prepare', 'timestep_prepare', 'dit_step_chunk',
          'vae_decode', 'finalize')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


class ImportNormalizer(ast.NodeTransformer):
    """Normalize exactly the allowed import substitutions, not executable code."""
    def visit_ImportFrom(self, node):
        if node.module == 'vllm.logger':
            if [(n.name, n.asname) for n in node.names] != [('init_logger', None)]:
                raise ValueError('unexpected upstream logging import')
            node.module = 'logging'
            node.names = [ast.alias(name='getLogger', asname='init_logger')]
        if node.module and (node.module == PREFIX or node.module.startswith(PREFIX + '.')):
            node.module = NAMESPACE + node.module[len(PREFIX):]
        if node.module == NAMESPACE + '.multiproc_worker':
            if [(n.name, n.asname) for n in node.names] != [('WorkerReadyMessage', None)]:
                raise ValueError('real worker dependency cannot be relocated as a message type')
            node.module = NAMESPACE + '.transport_protocol'
        return node


def relocate_source(raw):
    before = raw.decode('utf-8')
    after = re.sub(r'(?m)^(\s*from )vllm_omni\.diffusion\.runtime_v2(?=\.| import)',
                   r'\1' + NAMESPACE, before)
    after = after.replace('from vllm.logger import init_logger',
                          'from logging import getLogger as init_logger')
    after = after.replace('from ' + NAMESPACE + '.multiproc_worker import WorkerReadyMessage',
                          'from ' + NAMESPACE + '.transport_protocol import WorkerReadyMessage')
    expected = ImportNormalizer().visit(ast.parse(before))
    actual = ast.parse(after)
    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
        raise ValueError('source relocation changed more than permitted imports')
    for node in ast.walk(actual):
        modules = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                   [item.name for item in node.names] if isinstance(node, ast.Import) else [])
        if any(name == 'vllm' or name.startswith(('vllm.', 'vllm_omni.')) for name in modules):
            raise ValueError('real vLLM execution dependency in selected control-plane file')
    return after.encode(), sha(ast.dump(expected, include_attributes=False).encode())


def source_plan(source_root, source_manifest, source_manifest_sha256):
    manifest = upstream.verify_source(source_root, source_manifest, source_manifest_sha256)
    root = Path(source_root).resolve(strict=True)
    output, changes = {}, {}
    for name in CONTROL_FILES:
        origin = SOURCE_PREFIX + '/' + name
        raw = (root / origin).read_bytes()
        content, algorithm_sha = relocate_source(raw)
        target = NAMESPACE + '/' + name
        output[target] = content
        changes[target] = dict(upstream_path=origin, upstream_sha256=sha(raw),
            derived_sha256=sha(content), algorithm_ast_sha256=algorithm_sha,
            change='import namespace and logging provider only')
    # Scheduler's lazy import needs this actual transport message, not the
    # vLLM worker implementation. Relocate the unchanged official dataclass.
    worker_path = SOURCE_PREFIX + '/multiproc_worker.py'
    worker_text = (root / worker_path).read_text()
    nodes = [node for node in ast.parse(worker_text).body
             if isinstance(node, ast.ClassDef) and node.name == 'WorkerReadyMessage']
    if len(nodes) != 1:
        raise ValueError('official worker-ready message identity differs')
    node = nodes[0]
    first_line = min([node.lineno] + [item.lineno for item in node.decorator_list])
    message = '\n'.join(worker_text.splitlines()[first_line-1:node.end_lineno]) + '\n'
    message_raw = ('# SPDX-License-Identifier: Apache-2.0\nfrom dataclasses import dataclass\n\n' + message).encode()
    actual_node = next(item for item in ast.parse(message_raw).body if isinstance(item, ast.ClassDef))
    if ast.dump(node, include_attributes=False) != ast.dump(actual_node, include_attributes=False):
        raise ValueError('transport message changed during extraction')
    output[NAMESPACE + '/transport_protocol.py'] = message_raw
    for name in ('__init__.py', 'adapters/__init__.py'):
        output[NAMESPACE + '/' + name] = b'# Derived control-plane package; no vLLM runtime replacement.\n'
    output['UPSTREAM_LICENSE'] = (root / 'LICENSE').read_bytes()
    record = dict(schema=PORT_SCHEMA, upstream_commit=upstream.COMMIT,
        upstream_manifest_sha256=source_manifest_sha256,
        imports_relocated=changes, files={name: sha(raw) for name, raw in output.items()},
        extracted_message=dict(upstream_path=worker_path, upstream_sha256=manifest['files'][worker_path],
            class_name='WorkerReadyMessage', class_ast_sha256=sha(ast.dump(node, include_attributes=False).encode()),
            destination=NAMESPACE + '/transport_protocol.py'),
        native_execution_not_ported={name: manifest['files'][SOURCE_PREFIX + '/' + name]
                                     for name in MIGRATION_FILES},
        migration_implemented=False, native_vllm_worker_implemented=False,
        shared_worker_sha256=round_adapter.FROZEN_WORKER_SHA256,
        port_sha256=sha(Path(__file__).read_bytes()),
        round_sha256=sha(Path(round_adapter.__file__).read_bytes()),
        qualification_eligible=False, allows=[])
    return record, output


def apply(source_root, source_manifest, source_manifest_sha256, destination):
    source = Path(source_root).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('derived output must be disjoint from official source')
    if destination.exists():
        raise FileExistsError('previous derived output must be preserved')
    record, files = source_plan(source, source_manifest, source_manifest_sha256)
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    raw = json_bytes(record)
    with (destination / 'PORT_MANIFEST.json').open('xb') as stream:
        stream.write(raw)
    return dict(root=str(destination), manifest_sha256=sha(raw), manifest=record)


def load_control(root, manifest_sha256):
    root = Path(root).resolve(strict=True)
    raw = (root / 'PORT_MANIFEST.json').read_bytes()
    if sha(raw) != manifest_sha256:
        raise ValueError('derived source manifest SHA256 differs')
    manifest = upstream.read_json(raw)
    if manifest.get('schema') != PORT_SCHEMA or manifest.get('upstream_commit') != upstream.COMMIT:
        raise ValueError('derived source identity differs')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('linked derived source')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(manifest['files']) | {'PORT_MANIFEST.json'}:
        raise ValueError('derived source inventory differs')
    for name, expected in manifest['files'].items():
        posix = PurePosixPath(name)
        if posix.is_absolute() or '..' in posix.parts or str(posix) != name or '\\' in name:
            raise ValueError('unsafe derived source path')
        if sha((root / name).read_bytes()) != expected:
            raise ValueError('derived source file differs: ' + name)
    for name, module in list(sys.modules.items()):
        if name == NAMESPACE or name.startswith(NAMESPACE + '.'):
            origin = getattr(module, '__file__', None)
            if not origin or not Path(origin).resolve().is_relative_to(root):
                raise ValueError('another derived GF-DiT control plane is already imported')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    modules = {name: importlib.import_module(NAMESPACE + '.' + name) for name in
               ('protocol', 'topology', 'cost_model', 'scheduler', 'adapters.common')}
    # A cached module must still point to the exact source inventory checked above.
    for name, module in list(sys.modules.items()):
        if name == NAMESPACE or name.startswith(NAMESPACE + '.'):
            origin = Path(module.__file__).resolve()
            if not origin.is_relative_to(root) or origin.relative_to(root).as_posix() not in manifest['files']:
                raise ValueError('control-plane module escaped bound source')
    return SimpleNamespace(protocol=modules['protocol'], topology=modules['topology'],
        cost=modules['cost_model'], scheduler=modules['scheduler'], common=modules['adapters.common'])


def make_request(request_id, image, *, chunk_size=2, deadline_ms=60000.):
    if not isinstance(request_id, str) or not re.fullmatch('[a-zA-Z0-9_-]+', request_id):
        raise ValueError('explicit safe request identity required')
    round_adapter.positive_int('chunk size', chunk_size)
    if chunk_size > image.steps:
        raise ValueError('chunk size exceeds request steps')
    if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, (int, float)) or not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError('finite positive deadline required')
    return dict(request_id=request_id, image=asdict(image), chunk_size=chunk_size,
                runtime_v2_arrival_ms=0., runtime_v2_deadline_ms=float(deadline_ms))


@dataclass(frozen=True)
class RuntimeRequest:
    """Native policy reads request attributes (it does not read mapping keys)."""
    request_id: str
    image: dict
    chunk_size: int
    runtime_v2_arrival_ms: float
    runtime_v2_deadline_ms: float


class SD35Compiler:
    """Model-specific adapter using the original generic chunked-DiT compiler."""
    def __init__(self, control):
        self.control = control

    def compile_request(self, value):
        image = round_adapter.ImageRequest(**value['image'])
        checked = make_request(value['request_id'], image, chunk_size=value['chunk_size'],
                               deadline_ms=value['runtime_v2_deadline_ms'])
        if value != checked:
            raise ValueError('request differs from explicit SP1 diagnostic schema')
        # Frozen SD3.5 medium: VAE downsample 8, transformer patch size 2.
        latent_seq_len = (image.height // 16) * (image.width // 16)
        return self.control.common.build_chunked_dit_plan(
            request_id=value['request_id'], request_value=RuntimeRequest(**value), request_type='shared_sd35',
            num_steps=image.steps, chunk_size=value['chunk_size'], priority=0, group_id=None,
            text_seq_len=image.max_sequence_length + 77, latent_seq_len=latent_seq_len,
            state_codec_id='shared_sd35_sp1_local_only', decoded_codec_id=None,
            dit_step_do_true_cfg=(True,) * image.steps,
            metadata=dict(latent_seq_len=latent_seq_len, voxels=image.height * image.width,
                text_seq_len=image.max_sequence_length + 77, num_steps=image.steps,
                denoise_chunk_size=value['chunk_size']))


def diagnostic_cost_store(control):
    """Explicit synthetic fixture: exercises native cost math, never calibration."""
    key = control.cost.CostModelKey(tp=1, sp=1, ulysses=1, ring=1, cfg=1)
    model = control.cost.StageCostModel(Path('<synthetic-sp1-logic-fixture>'), key,
        {stage: dict(coeffs=dict(a=0., b=0., c=1.)) for stage in STAGES})
    store = control.cost.CostModelStore({key: model})
    def estimate(spec, seq_len, stage):
        result = store.for_parallel_spec(spec).estimate_ms(str(stage), float(seq_len))
        if not math.isfinite(result) or result <= 0:
            raise ValueError('nonfinite/nonpositive diagnostic cost')
        return result
    return store, estimate


class SingleGpuWorkerPool:
    """Real synchronous executor transport for one GPU; no migration fallback.

    Dispatch and completion use GF-DiT's original events and worker-local artifact
    references. The transport is deliberately not the native multiprocess pool
    and cannot establish its overlap, migration or serving performance.
    """
    def __init__(self, control, topology, executor):
        self.p, self.topology, self.executor = control.protocol, topology, executor
        self.pending, self.values, self.handles = deque(), {}, {}
        self.events, self.trace, self.cleaned = [], [], []
        self.started = False

    def start(self):
        if self.started:
            raise ValueError('worker already started')
        self.started = True

    def shutdown(self):
        self.pending.clear()
        self.values.clear()
        self.handles.clear()
        self.started = False

    def pump_migrations(self):
        # There can be none: begin_migration unconditionally rejects below.
        return None

    def has_pending_migrations(self):
        return False

    def begin_migration(self, *args, **kwargs):
        raise NotImplementedError('cross-rank GF-DiT migration is not ported; no host/NCCL fallback')

    def dispatch(self, *, task, inline_inputs, release_after_exec_artifact_ids):
        group = self.topology.get_group(task.group_id)
        if (not self.started or tuple(group.ranks) != (0,)
                or group.parallel_spec != self.p.ParallelSpec()
                or group.ulysses_degree != 1 or group.ring_degree != 1):
            raise ValueError('SP1 worker cannot execute a different rank/parallel form')
        if self.pending:
            raise ValueError('native policy dispatched overlapping work onto the single worker')
        for item in inline_inputs:
            key = (item.handle.request_id, item.handle.artifact_id)
            self.values[key], self.handles[key] = item.value, item.handle
        self.pending.append((task, tuple(release_after_exec_artifact_ids)))

    def _event(self, task, kind, **kw):
        now = time.monotonic_ns()
        return self.p.WorkerEvent(str(now), task.task_id, task.request_id, task.group_id,
                                  0, kind, now, **kw)

    def poll(self, timeout_s=0.):
        if not self.pending:
            return []
        task, release = self.pending.popleft()
        p = self.p
        events = [self._event(task, p.WorkerEventKind.TASK_LAUNCH_BEGIN),
                  self._event(task, p.WorkerEventKind.TASK_EXEC_BEGIN)]
        row = dict(task_id=task.task_id, request_id=task.request_id, kind=task.kind.value,
                   group_id=task.group_id, ranks=[0], parallel_spec=asdict(task.parallel_spec),
                   step_range=asdict(task.step_range) if task.step_range else None,
                   status='failed')
        self.trace.append(row)
        try:
            inputs = {handle.artifact_id: self.values[(handle.request_id, handle.artifact_id)]
                      for handle in task.inputs}
            outputs = tuple(self.executor.execute(task, inputs))
            if tuple(item.handle for item in outputs) != task.outputs:
                raise ValueError('executor did not produce the exact declared artifacts')
            refs = []
            for item in outputs:
                key = (item.handle.request_id, item.handle.artifact_id)
                self.values[key], self.handles[key] = item.value, item.handle
                refs.append(p.WorkerLocalArtifactRef(item.handle, task.group_id, 0))
            for artifact_id in release:
                key = (task.request_id, artifact_id)
                self.values.pop(key, None)
                self.handles.pop(key, None)
            events.extend([self._event(task, p.WorkerEventKind.TASK_LAUNCH_END,
                                      metadata=dict(published_outputs=tuple(refs))),
                           self._event(task, p.WorkerEventKind.TASK_EXEC_END)])
            row['status'] = 'completed'
        except Exception as exc:
            row['error'] = repr(exc)
            events.append(self._event(task, p.WorkerEventKind.TASK_FAILED, message=repr(exc)))
        self.events.extend(dict(task_id=event.task_id, kind=event.kind.value) for event in events)
        return events

    def fetch_artifacts(self, *, request_id, group_id, artifact_ids):
        if self.topology.get_group(group_id).ranks != (0,):
            raise ValueError('cannot fetch a remote artifact')
        values = tuple(self.p.ArtifactValue(self.handles[(request_id, name)], self.values[(request_id, name)])
                       for name in artifact_ids)
        return SimpleNamespace(error=None, artifacts=values)

    def evict_request(self, request_id):
        for key in list(self.values):
            if key[0] == request_id:
                self.values.pop(key)
                self.handles.pop(key)
        self.cleaned.append(request_id)


def build_runtime(control, executor):
    p = control.protocol
    topology = control.topology.RuntimeTopology(
        (control.topology.WorkerSpec(0, 0, 'shared-sd35-gpu0'),),
        (p.ExecutionGroupSpec('sp1-rank0', (0,), p.ParallelSpec(), control.common.STANDARD_TASK_KINDS),))
    compiler = SD35Compiler(control)
    store, estimator = diagnostic_cost_store(control)
    policy = control.scheduler.EdfBestFitSchedulerPolicy(topology, estimator, allowed_sp_sizes=(1,))
    worker = SingleGpuWorkerPool(control, topology, executor)
    scheduler = control.scheduler.GlobalScheduler(topology, worker, compiler,
                                                  control.scheduler.InMemoryArtifactStore(), policy)
    return scheduler, worker, store


@dataclass(frozen=True)
class ModelState:
    request: object
    prompt_embeds: object
    pooled_prompt_embeds: object
    latents: object = None


class SD35Executor:
    """Actual per-stage shared backend calls, with explicit CPU RNG and cursor."""
    def __init__(self, control, pipeline):
        self.p, self.pipeline = control.protocol, pipeline
        self.last_latents = None
        if pipeline.raw.vae_scale_factor != 8 or pipeline.pipe.transformer.config.patch_size != 2:
            raise ValueError('SD3.5 sequence geometry differs from the bound compiler')

    def execute(self, task, inputs):
        pipe, p = self.pipeline, self.p
        torch = pipe.torch
        if len(inputs) != 1 or len(task.outputs) != 1:
            raise ValueError('each supported SD3.5 task requires exactly one input and output')
        value = next(iter(inputs.values()))
        kind = task.kind
        if kind == p.TaskKind.DIT_STEP_CHUNK:
            if not isinstance(value, round_adapter.RoundState) or task.step_range.start != value.cursor:
                raise ValueError('native task replayed/skipped an absolute denoising interval')
            result = pipe.advance(value, task.step_range.end)
            self.last_latents = result.latents if result.cursor == result.request.steps else None
        else:
            with pipe._execution():
                device = pipe.raw._execution_device
                if kind == p.TaskKind.TEXT_ENCODE:
                    if not isinstance(value, RuntimeRequest):
                        raise ValueError('text task requires the explicit native request protocol')
                    request = round_adapter.ImageRequest(**value.image)
                    pipe._profile(request.guidance_scale, request.max_sequence_length)
                    pe, ne, pp, np = pipe.raw.encode_prompt(
                        prompt=[request.prompt], prompt_2=None, prompt_3=None,
                        negative_prompt=[request.negative_prompt], negative_prompt_2=None, negative_prompt_3=None,
                        do_classifier_free_guidance=True, device=device, num_images_per_prompt=1,
                        max_sequence_length=request.max_sequence_length)
                    result = ModelState(request, torch.stack([ne, pe], dim=1).flatten(0, 1),
                                        torch.stack([np, pp], dim=1).flatten(0, 1))
                elif kind == p.TaskKind.DIT_PREPARE:
                    if not isinstance(value, ModelState) or value.latents is not None:
                        raise ValueError('DiT preparation received wrong stage state')
                    r = value.request
                    generator = torch.Generator(device='cpu').manual_seed(r.seed)
                    latents = pipe.raw.prepare_latents(1, pipe.pipe.transformer.config.in_channels,
                        r.height, r.width, value.prompt_embeds.dtype, device, generator, None)
                    result = replace(value, latents=latents)
                elif kind == p.TaskKind.TIMESTEP_PREPARE:
                    if not isinstance(value, ModelState) or value.latents is None:
                        raise ValueError('timestep preparation requires actual initial latents')
                    scheduler = pipe.scheduler_class.from_config(pipe.scheduler_config)
                    kwargs = {}
                    if pipe.scheduler_config.get('use_dynamic_shifting', False):
                        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
                        patch = pipe.pipe.transformer.config.patch_size
                        tokens = (value.latents.shape[-2] // patch) * (value.latents.shape[-1] // patch)
                        kwargs['mu'] = calculate_shift(tokens,
                            pipe.scheduler_config.get('base_image_seq_len', 256),
                            pipe.scheduler_config.get('max_image_seq_len', 4096),
                            pipe.scheduler_config.get('base_shift', .5),
                            pipe.scheduler_config.get('max_shift', 1.16))
                    scheduler.set_timesteps(value.request.steps, device=device, **kwargs)
                    ts, sigmas = scheduler.timesteps.to(device), scheduler.sigmas.to(device)
                    round_adapter.validate_schedule(ts.detach().cpu().tolist(), sigmas.detach().cpu().tolist())
                    result = round_adapter.RoundState(value.request, value.prompt_embeds,
                        value.pooled_prompt_embeds, value.latents, ts, sigmas)
                elif kind == p.TaskKind.VAE_DECODE:
                    if not isinstance(value, round_adapter.RoundState) or value.cursor != value.request.steps:
                        raise ValueError('VAE decode requires all original denoising steps')
                    result = pipe.raw.vae.decode(value.latents / pipe.raw.vae.config.scaling_factor
                                                + pipe.raw.vae.config.shift_factor, return_dict=False)[0]
                    if not torch.isfinite(result).all().item():
                        raise ValueError('nonfinite VAE output')
                elif kind == p.TaskKind.FINALIZE:
                    # Real image postprocessing; retain a tensor only for numerical
                    # comparison, never emit raw arrays into evidence files.
                    result = pipe.raw.image_processor.postprocess(value, output_type='pt')
                else:
                    raise NotImplementedError('task kind has no shared SD3.5 executor: ' + str(kind))
        torch.cuda.synchronize()
        return (p.ArtifactValue(task.outputs[0], result),)


def graph_summary(control, request):
    plan = SD35Compiler(control).compile_request(request)
    tasks = list(plan.tasks.values())
    intervals = sorted((task.step_range.start, task.step_range.end) for task in tasks if task.step_range)
    total = request['image']['steps']
    if not intervals or intervals[0][0] != 0 or intervals[-1][1] != total:
        raise ValueError('native graph did not cover complete denoising')
    if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('native graph has repeated or missing denoising steps')
    return dict(task_count=len(tasks), intervals=intervals,
        tasks=[dict(task_id=task.task_id, kind=task.kind.value, dependencies=list(task.dependencies)) for task in tasks],
        terminal_task_ids=list(plan.terminal_task_ids), original_steps=total)


def execute_gpu(control, request, args, result):
    import torch
    import diffusers
    from external_sd35_round_check import compare_tensors, tensor_summary
    if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('first shared GF-DiT port requires exactly one GPU and rank')
    if diffusers.__version__ != '0.36.0':
        raise ValueError('fixed diffusers 0.36.0 is required')
    result['runtime'] = dict(torch=torch.__version__, diffusers=diffusers.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    pipeline, scheduler = None, None
    try:
        result['stage'] = 'load_verified_shared_model'
        image = round_adapter.ImageRequest(**request['image'])
        pipeline = round_adapter.SD35RoundPipeline.from_pretrained(args.model_root,
            shared_python_root=args.shared_python_root, model_manifest_path=args.model_manifest,
            model_manifest_sha256=args.model_manifest_sha256, attention='math', device='cuda',
            guidance_scale=image.guidance_scale, max_sequence_length=image.max_sequence_length)
        result['model_weights_loaded'] = True
        result['model_provenance'] = pipeline.provenance
        result['backend'] = pipeline.runtime.info()
        executor = SD35Executor(control, pipeline)
        scheduler, worker, _ = build_runtime(control, executor)
        scheduler.start()
        result['stage'] = 'original_gfdit_scheduler_shared_execution'
        result['native_scheduler_executed'] = True
        actual = scheduler.execute(request, timeout_s=args.timeout_seconds)
        result['task_trace'] = worker.trace
        result['event_counts'] = {kind.value: sum(event['kind'] == kind.value for event in worker.events)
                                  for kind in control.protocol.WorkerEventKind}
        result['worker_requests_evicted'] = worker.cleaned
        if len(worker.trace) != result['graph']['task_count'] or any(row['status'] != 'completed' for row in worker.trace):
            raise ValueError('not every declared native DAG task completed')
        actual_latents = executor.last_latents
        if actual_latents is None:
            raise ValueError('complete native graph latent is missing')
        result['stage'] = 'normal_shared_pipeline_reference'
        with torch.inference_mode():
            reference_latents = pipeline.pipe(prompt=image.prompt, negative_prompt=image.negative_prompt,
                height=image.height, width=image.width, num_inference_steps=image.steps,
                guidance_scale=image.guidance_scale, max_sequence_length=image.max_sequence_length,
                generator=torch.Generator(device='cpu').manual_seed(image.seed), output_type='latent').images
            decoded = pipeline.raw.vae.decode(reference_latents / pipeline.raw.vae.config.scaling_factor
                                               + pipeline.raw.vae.config.shift_factor, return_dict=False)[0]
            reference = pipeline.raw.image_processor.postprocess(decoded, output_type='pt')
            torch.cuda.synchronize()
        result['completed_trajectories'] = 2
        result['comparisons'] = dict(latents=compare_tensors(reference_latents, actual_latents),
                                     images=compare_tensors(reference, actual))
        result['outputs'] = dict(reference=tensor_summary(reference), actual=tensor_summary(actual),
                                 reference_latents=tensor_summary(reference_latents), actual_latents=tensor_summary(actual_latents))
        result['numerical_check_passed'] = all(item['passed'] for item in result['comparisons'].values())
        result['stage'] = 'completed'
    finally:
        if scheduler is not None:
            result.setdefault('task_trace', scheduler.worker_pool.trace)
            scheduler.shutdown()
        if pipeline is not None:
            pipeline.runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shared-python-root', type=Path)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--model-manifest', type=Path)
    parser.add_argument('--model-manifest-sha256')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--chunk-size', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if args.command == 'run' and any(getattr(args, name) is None for name in
            ('shared_python_root', 'model_root', 'model_manifest', 'model_manifest_sha256', 'protocol', 'protocol_sha256')):
        parser.error('run requires all shared model, manifest and protocol arguments')
    if not 30 <= args.timeout_seconds <= 3600:
        parser.error('timeout must be within 30..3600 seconds')
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('previous output must be preserved')
    for path in (args.source_root, args.shared_python_root, args.model_root):
        if path is not None and (output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)):
            raise ValueError('output must be disjoint from every bound input')
    result = dict(schema='ditserve.gfdit-shared-sp1-check.v1', status='failed', stage='source_binding',
        upstream_commit=upstream.COMMIT, native_scheduler_executed=False,
        model_weights_loaded=False, completed_trajectories=0, numerical_check_passed=False,
        costs_are_measurements=False, native_vllm_worker_executed=False,
        migration_executed=False, online_regroup_verified=False, serving_executed=False,
        performance_measurements=False, qualification_eligible=False, allows=[],
        scope='SP1 shared-backend numerical check using original GF-DiT EDF/DAG; not full GF-DiT')
    stop = threading.Event()
    if args.command == 'run':
        from external_sd35_round_check import check_storage, storage_watchdog
        if not output.is_relative_to(Path('/shared/output/DITSERVE')):
            raise ValueError('GPU evidence must be under /shared/output/DITSERVE')
        check_storage()
        threading.Thread(target=storage_watchdog, args=(stop,), daemon=True).start()
    else:
        check_storage = lambda: None
    created = False
    try:
        image = round_adapter.ImageRequest(args.prompt, negative_prompt='blurry, low quality', seed=args.seed)
        if args.command == 'run':
            from external_sd35_round_check import request_plan
            if sha(args.protocol.read_bytes()) != args.protocol_sha256:
                raise ValueError('formal protocol SHA256 differs')
            image = request_plan(upstream.read_json(args.protocol.read_bytes()), args.prompt, args.seed)
            round_adapter.verify_shared_sources(args.shared_python_root)
        request = make_request('gfdit-shared-sd35', image, chunk_size=args.chunk_size)
        check_storage()
        output.mkdir(parents=True, exist_ok=False)
        created = True
        derived = apply(args.source_root, args.source_manifest, args.source_manifest_sha256, output / 'control')
        control = load_control(derived['root'], derived['manifest_sha256'])
        result.update(derived_manifest_sha256=derived['manifest_sha256'], source_manifest_sha256=args.source_manifest_sha256,
            port_sha256=sha(Path(__file__).read_bytes()), round_sha256=sha(Path(round_adapter.__file__).read_bytes()),
            shared_worker_sha256=round_adapter.FROZEN_WORKER_SHA256, request=request,
            graph=graph_summary(control, request), protocol_sha256=args.protocol_sha256,
            model_manifest_sha256=args.model_manifest_sha256,
            generator_device='cpu', attention='sdpa_math', costs='explicit synthetic positive SP1 logic fixture')
        if args.command == 'run':
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            execute_gpu(control, request, args, result)
            result['status'] = 'numerical_passed' if result['numerical_check_passed'] else 'numerical_failed'
        else:
            result.update(status='source_and_graph_planned', stage='planned')
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-16000:])
    finally:
        stop.set()
    if created:
        check_storage()
        raw = json_bytes(result)
        with (output / 'summary.json').open('xb') as stream:
            stream.write(raw)
        print(json.dumps(dict(status=result['status'], summary=str(output / 'summary.json'), sha256=sha(raw),
                              qualification_eligible=False)), flush=True)
    else:
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result['status'] in ('source_and_graph_planned', 'numerical_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
