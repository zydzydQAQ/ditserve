"""Real GF-DiT RuntimeV2Runner/MultiprocWorkerPool with a shared SD3.5 adapter.

No upstream source is patched and no vLLM module is replaced. The documented
worker-extension hook changes model loading only. Native EDF, DAG dispatch,
spawn/Pipe workers, MigrationEngine, DataPlane and GFC remain in use. The first
two-request run is a functional/numerical diagnostic with explicitly synthetic
costs; it is neither a calibrated serving result nor a complete model matrix.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

sys.dont_write_bytecode = True

import external_gfdit as upstream
import external_gfdit_shared as staged
import external_gfc_smoke as gfc_source
import external_sd35_round as rounds
from external_sd35_round_check import request_plan, tensor_summary, compare_tensors

MODEL_CLASS = 'SharedXfuserSD35RuntimeV2'
STATE_CODEC = 'shared-sd35-service-state-v1'
DECODED_CODEC = 'shared-sd35-service-decoded-v1'
STATE_SCHEMA = 'ditserve.gfdit-shared-sd35-state.v1'
SOURCE_PREFIX = 'vllm_omni.diffusion.runtime_v2'
TENSORS = ('prompt_embeds', 'pooled_prompt_embeds', 'latents', 'timesteps', 'sigmas')
MODULES = ('protocol', 'codec', 'interfaces', 'adapters.common', 'registry',
           'runner', 'scheduler', 'multiproc_worker', 'migration_engine', 'data_plane', 'cost_model')
REQUIRED_ABI = {
    'vllm.config': ('VllmConfig', 'CompilationConfig', 'set_current_vllm_config'),
    'vllm.v1.worker.workspace': ('init_workspace_manager',),
    'vllm.distributed.device_communicators.shm_broadcast': ('MessageQueue',),
    'vllm.utils.import_utils': ('resolve_obj_by_qualname',),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())



def normalized_uuid(value):
    if not isinstance(value, str):
        raise ValueError('GPU UUID must be a string')
    raw = value[4:] if value.startswith('GPU-') else value
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError):
        raise ValueError('invalid actual GPU UUID') from None
    if str(parsed) != raw.lower():
        raise ValueError('GPU UUID must use canonical hyphenated form')
    return 'GPU-' + str(parsed)


def check_storage():
    result = subprocess.run(['df', '-BG', '/shared'], capture_output=True, text=True, timeout=20)
    fields = result.stdout.splitlines()[-1].split() if result.stdout else []
    if result.returncode or len(fields) < 6:
        raise RuntimeError('cannot verify shared storage')
    used, percent = int(fields[2].rstrip('G')), int(fields[4].rstrip('%'))
    if used > 800 or percent >= 80:
        raise RuntimeError('shared storage stop line: >800G or >=80%')
    return dict(used_gib=used, percent=percent)


def watchdog(stop):
    while not stop.wait(5):
        try:
            check_storage()
        except BaseException:
            os._exit(75)


def source_binding(config):
    """Check real upstream, frozen shared model code and all adapter dependencies."""
    source = Path(config['source_root']).resolve(strict=True)
    manifest = upstream.verify_source(source, config['source_manifest'], config['source_manifest_sha256'])
    shared = rounds.verify_shared_sources(config['shared_python_root'])
    gf = gfc_source.verify_sources(config['gfc_source_root'], config['gfc_source_manifest'],
                                   config['gfc_source_manifest_sha256'], shared)
    for name, expected in config['adapter_files'].items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or sha(path.read_bytes()) != expected:
            raise ValueError('adapter source changed: ' + name)
    if str(Path(__file__).resolve()) not in config['adapter_files']:
        raise ValueError('executed service adapter is not source-bound')
    protocol_raw = Path(config['protocol']).read_bytes()
    if sha(protocol_raw) != config['protocol_sha256']:
        raise ValueError('formal protocol SHA256 differs')
    model_raw = Path(config['model_manifest']).read_bytes()
    if sha(model_raw) != config['model_manifest_sha256']:
        raise ValueError('model manifest SHA256 differs')
    model = upstream.read_json(model_raw)
    if model.get('model') != rounds.SD35_MODEL or model.get('revision') != rounds.SD35_REVISION:
        raise ValueError('only the exact official SD3.5 medium weights are supported')
    for name, module in list(sys.modules.items()):
        base = source if name == 'vllm_omni' or name.startswith('vllm_omni.') else shared if name in ('xfuser', 'ditserve', 'gfc') or name.startswith(('xfuser.', 'ditserve.', 'gfc.')) else None
        origin = getattr(module, '__file__', None)
        if base is not None and origin and not Path(origin).resolve().is_relative_to(base):
            raise ValueError('another source package is already imported: ' + name)
    sys.path[:0] = [str(shared), str(source), str(Path(__file__).resolve().parent)]
    return dict(upstream_commit=upstream.COMMIT, upstream_files=len(manifest['files']),
                upstream_manifest_sha256=config['source_manifest_sha256'], gfc=gf,
                shared_worker_sha256=rounds.FROZEN_WORKER_SHA256, adapter_files=config['adapter_files'])


def control():
    modules = {name: importlib.import_module(SOURCE_PREFIX + '.' + name) for name in MODULES}
    return SimpleNamespace(**{name.replace('.', '_'): module for name, module in modules.items()},
        common=modules['adapters.common'])


def abi_receipt(config):
    result = {}
    for name, symbols in REQUIRED_ABI.items():
        module = importlib.import_module(name)
        for symbol in symbols:
            if not callable(getattr(module, symbol, None)):
                raise ValueError('missing real vLLM ABI: ' + name + '.' + symbol)
        result[name] = dict(path=str(Path(module.__file__).resolve()), symbols=list(symbols))
    import torch
    import vllm
    import diffusers
    if diffusers.__version__ != '0.36.0':
        raise ValueError('shared numerical protocol requires diffusers 0.36.0')
    official = Path(config['source_root']).resolve()
    c = control()
    for name in MODULES:
        module = getattr(c, name.replace('.', '_'))
        expected = official / (SOURCE_PREFIX.replace('.', '/') + '/' + name.replace('.', '/') + '.py')
        if Path(module.__file__).resolve() != expected:
            raise ValueError('official GF-DiT module was replaced: ' + name)
    result['versions'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                              vllm=vllm.__version__, diffusers=diffusers.__version__)
    return result


@dataclass(frozen=True)
class ServiceRequest:
    request_id: str
    image: dict
    chunk_size: int
    runtime_v2_arrival_ms: float
    runtime_v2_deadline_ms: float

    @property
    def denoise_chunk_size(self):
        return self.chunk_size

    @property
    def diffusion_request(self):
        # RuntimeV2Runner logs the official request type's sampling parameters.
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        image = rounds.ImageRequest(**self.image)
        sampling = OmniDiffusionSamplingParams(height=image.height, width=image.width,
            num_frames=1, num_inference_steps=image.steps, guidance_scale=image.guidance_scale,
            seed=image.seed)
        return OmniDiffusionRequest(prompts=[dict(prompt=image.prompt, negative_prompt=image.negative_prompt)],
                                    sampling_params=sampling, request_ids=[self.request_id])


def validate_request(value):
    if not isinstance(value, ServiceRequest) or not re.fullmatch('[A-Za-z0-9_-]+', value.request_id):
        raise ValueError('an explicit shared-service request identity is required')
    image = rounds.ImageRequest(**value.image)
    if asdict(image) != value.image or image.height != image.width or image.height not in (512, 1024):
        raise ValueError('request must bind every immutable field of a formal SD3.5 shape')
    if (image.steps, image.guidance_scale, image.max_sequence_length, image.negative_prompt) != (28, 4.5, 256, 'blurry, low quality'):
        raise ValueError('shared service only admits the frozen SD3.5 profile')
    if type(value.chunk_size) is not int or not 1 <= value.chunk_size <= image.steps:
        raise ValueError('invalid denoising chunk size')
    for name in ('runtime_v2_arrival_ms', 'runtime_v2_deadline_ms'):
        number = getattr(value, name)
        if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
            raise ValueError('invalid arrival/deadline')
    if value.runtime_v2_deadline_ms <= value.runtime_v2_arrival_ms:
        raise ValueError('deadline must follow arrival')
    return image


def request_metadata(value):
    image = validate_request(value)
    return dict(request_id=value.request_id, image=asdict(image), request_sha256=canonical(asdict(image)),
                latent_seq_len=(image.height // 16) * (image.width // 16),
                text_seq_len=image.max_sequence_length + 77, num_steps=image.steps,
                voxels=image.height * image.width, denoise_chunk_size=value.chunk_size)


class Compiler:
    def __init__(self, native):
        self.native = native

    def compile_request(self, value):
        image = validate_request(value)
        metadata = request_metadata(value)
        return self.native.common.build_chunked_dit_plan(request_id=value.request_id,
            request_value=staged.RuntimeRequest(**asdict(value)), request_type='shared_sd35_service',
            num_steps=image.steps, chunk_size=value.chunk_size, priority=0, group_id=None,
            text_seq_len=metadata['text_seq_len'], latent_seq_len=metadata['latent_seq_len'],
            state_codec_id=STATE_CODEC, decoded_codec_id=DECODED_CODEC,
            dit_step_do_true_cfg=(True,) * image.steps, metadata=metadata)


def artifact_fields(artifact_id, image):
    """Full replicated continuation; the shared round call shards only transiently."""
    shapes = {'prompt_embeds': (2, image.max_sequence_length + 77, 4096),
              'pooled_prompt_embeds': (2, 2048), 'latents': (1, 16, image.height // 8, image.width // 8),
              'timesteps': (image.steps,), 'sigmas': (image.steps + 1,),
              'image_tensor': (1, 3, image.height, image.width)}
    if artifact_id == 'decoded':
        names = ('image_tensor',)
    elif artifact_id == 'state_text':
        names = TENSORS[:2]
    elif artifact_id == 'state_latent':
        names = TENSORS[:3]
    elif artifact_id.startswith('state_'):
        names = TENSORS
    else:
        raise ValueError('unknown migratable SD3.5 artifact: ' + artifact_id)
    return {name: dict(shape=shapes[name], dtype='torch.float32' if name in ('timesteps', 'sigmas') else 'torch.bfloat16') for name in names}


def validate_state_metadata(value, artifact):
    if not isinstance(value, dict) or value.get('schema') != STATE_SCHEMA or value.get('request_id') != artifact.request_id:
        raise ValueError('continuation request identity differs')
    image = rounds.ImageRequest(**value.get('image', {}))
    # Reuse the same immutable request validation without manufacturing timing.
    validate_request(ServiceRequest(value['request_id'], value['image'], 2, 0., 1.))
    if value.get('request_sha256') != canonical(value['image']):
        raise ValueError('continuation request contents changed')
    cursor = value.get('cursor')
    if type(cursor) is not int or not 0 <= cursor <= image.steps:
        raise ValueError('continuation cursor is invalid')
    if artifact.artifact_id in ('state_text', 'state_latent', 'state_timestep') and cursor != 0:
        raise ValueError('prologue artifact has a nonzero cursor')
    if artifact.artifact_id in ('decoded', 'state_denoised') and cursor != image.steps:
        raise ValueError('final artifact lacks all original steps')
    expected = set(artifact_fields(artifact.artifact_id, image))
    if set(value) != expected | {'schema', 'request_id', 'image', 'request_sha256', 'cursor'}:
        raise ValueError('undeclared continuation field')
    return image



def validate_model_tensors(value, artifact, device=None):
    import torch
    image = validate_state_metadata(value, artifact)
    for name, spec in artifact_fields(artifact.artifact_id, image).items():
        tensor = value[name]
        if (not isinstance(tensor, torch.Tensor) or not tensor.is_cuda
                or tuple(tensor.shape) != spec['shape'] or str(tensor.dtype) != spec['dtype']
                or device is not None and tensor.device != torch.device(device)
                or not torch.isfinite(tensor).all().item()):
            raise ValueError('actual continuation tensor differs: ' + name)
    return image


def codecs(native):
    class Codec(native.codec.ArtifactLayoutCodec):
        def __init__(self, name):
            self._id = name
        @property
        def codec_id(self):
            return self._id
        def describe_output(self, *, group, group_rank, request_metadata, artifact):
            if (group.parallel_spec.tp, group.parallel_spec.cfg) != (1, 1) or group.parallel_spec.sp != len(group.ranks):
                raise ValueError('only original GFC SP1/SP2 sessions are admitted')
            if len(group.ranks) not in (1, 2) or not 0 <= group_rank < len(group.ranks):
                raise ValueError('unsupported execution group')
            image = rounds.ImageRequest(**request_metadata['image'])
            if request_metadata['request_id'] != artifact.request_id or request_metadata['request_sha256'] != canonical(asdict(image)):
                raise ValueError('native migration metadata differs from the request')
            fields = artifact_fields(artifact.artifact_id, image)
            return native.protocol.OutputArtifactLayout(artifact, tuple(native.common.replicated_tensor_field(
                name, item['shape'], item['dtype']) for name, item in fields.items()))
        def pack_metadata(self, *, value, layout):
            image = validate_model_tensors(value, layout.handle)
            if {tuple(item.field_path) for item in layout.tensors} != {(name,) for name in artifact_fields(layout.handle.artifact_id, image)}:
                raise ValueError('migration layout omitted a continuation tensor')
            raw = super().pack_metadata(value=value, layout=layout)
            skeleton = pickle.loads(raw)
            if any(skeleton[name] is not None for name in artifact_fields(layout.handle.artifact_id, image)) or len(raw) > 65536:
                raise ValueError('tensor payload escaped the original data plane')
            return raw
        def assemble(self, *, metadata_bytes, tensors, layout, device=None):
            value = super().assemble(metadata_bytes=metadata_bytes, tensors=tensors, layout=layout, device=device)
            validate_model_tensors(value, layout.handle, device=device)
            return value
    return {name: Codec(name) for name in (STATE_CODEC, DECODED_CODEC)}


def borrowed_gfc_pool(device):
    """Bridge original logical groups into xfuser without creating any groups/runtime."""
    import torch
    from ditserve.models.xdit.groups import ProcessGroupPool
    from ditserve.models.xdit import collectives as common_collectives
    from ditserve.models.xdit.process_group import PROCESS_GROUP
    from vllm_omni.diffusion.distributed import parallel_state as native_groups
    from vllm_omni.diffusion.distributed import collective_runtime as native_collectives
    if not native_collectives.runtime_v2_uses_gfc() or common_collectives.backend() != 'nccl':
        raise ValueError('requires original GF-DiT GFC and no second common GFC runtime')
    native_collectives.get_gfc_runtime()
    native_collectives.get_gfc_migrate_runtime()
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if world != 2:
        raise ValueError('the first full-process service diagnostic requires two ranks')
    class Pool(ProcessGroupPool):
        def _device_group(self, ranks):
            ranks = tuple(ranks)
            active = native_groups.get_sp_group()
            if ranks == tuple(active.ranks):
                return active.device_group
            # Local descriptor registration on the already initialized native runtime.
            return native_collectives.make_logical_group(ranks, group_id='shared-view-' + '-'.join(map(str, ranks)))
        def _cpu_group(self, ranks):
            if tuple(ranks) == tuple(range(world)):
                return native_groups.get_world_group().cpu_group
            # Never let an unintended CPU subgroup call fall back to the world.
            return ('unsupported-logical-cpu-group', tuple(ranks))
        def _coordinator(self, ranks, **kwargs):
            result = super()._coordinator(ranks, **kwargs)
            def gather(output, tensor):
                native_collectives.all_gather_into_tensor(output, tensor, group=result.device_group)
                return True
            result.all_gather_override = gather
            return result
        def activate(self, sp_ranks, runtime_state=None, **kwargs):
            result = super().activate(sp_ranks, runtime_state, **kwargs)
            def all_to_all(flat, group):
                output = torch.empty_like(flat)
                native_collectives.all_to_all_single(output, flat, group=group)
                return output
            PROCESS_GROUP.ALL_TO_ALL = all_to_all
            return result
        def close(self):
            # Official worker owns all process groups and both GFC runtimes.
            self.groups = None
            super().close()
    pool = Pool(rank, world, device, 'gfc')
    pool.install()
    return pool


def activate_pipeline(pipeline, task):
    from vllm_omni.diffusion.distributed import parallel_state as native_groups
    active = native_groups.get_sp_group()
    if (task.parallel_spec.tp, task.parallel_spec.cfg) != (1, 1) or len(active.ranks) != task.parallel_spec.sp:
        raise ValueError('shared model and original native session disagree')
    if task.parallel_spec.sp not in (1, 2):
        raise ValueError('only SP1/SP2 are implemented in the first service check')
    if getattr(active, 'ulysses_world_size', len(active.ranks)) != len(active.ranks):
        raise ValueError('shared model requires pure Ulysses SP')
    pipeline.runtime.pool.activate(tuple(active.ranks), pipeline.runtime.state,
                                  ulysses_degree=len(active.ranks), ring_degree=1)
    pipeline.runtime.bind_global()


class Executor:
    def __init__(self, native, pipeline, state_codecs):
        self.native, self.pipeline = native, pipeline
        self.base = staged.SD35Executor(native, pipeline)
        self._codecs = state_codecs
    @property
    def output_codecs(self):
        return self._codecs
    def execute(self, task, inputs):
        if len(inputs) != 1 or len(task.outputs) != 1:
            raise ValueError('shared stages require one input and one output')
        activate_pipeline(self.pipeline, task)
        value = next(iter(inputs.values()))
        p = self.native.protocol
        metadata = task.payload['request_metadata']
        if task.kind == p.TaskKind.TEXT_ENCODE:
            if not isinstance(value, staged.RuntimeRequest) or value.image != metadata['image'] or value.request_id != task.request_id:
                raise ValueError('initial request differs from the original native plan')
            converted = value
        else:
            image = validate_state_metadata(value, task.inputs[0])
            if value['image'] != metadata['image'] or value['request_id'] != task.request_id:
                raise ValueError('request state leaked across concurrent requests')
            if task.kind == p.TaskKind.FINALIZE:
                converted = value['image_tensor']
            elif 'timesteps' in value:
                converted = rounds.RoundState(image, **{name: value[name] for name in TENSORS}, cursor=value['cursor'])
            else:
                converted = staged.ModelState(image, value['prompt_embeds'], value['pooled_prompt_embeds'], value.get('latents'))
        result = self.base.execute(task, {next(iter(inputs)): converted})[0].value
        if task.kind == p.TaskKind.FINALIZE:
            from vllm_omni.diffusion.data import DiffusionOutput
            result = DiffusionOutput(output=result.detach().cpu(), custom_output=dict(
                request_id=task.request_id, image=metadata['image'], request_sha256=metadata['request_sha256'],
                cursor=28, model=rounds.SD35_MODEL, shared_worker_sha256=rounds.FROZEN_WORKER_SHA256))
        else:
            result_value = dict(schema=STATE_SCHEMA, request_id=task.request_id, image=metadata['image'],
                                request_sha256=metadata['request_sha256'], cursor=0)
            if task.kind == p.TaskKind.VAE_DECODE:
                result_value.update(cursor=28, image_tensor=result)
            elif isinstance(result, rounds.RoundState):
                result_value.update(cursor=result.cursor, **{name: getattr(result, name) for name in TENSORS})
            elif isinstance(result, staged.ModelState):
                result_value.update(prompt_embeds=result.prompt_embeds, pooled_prompt_embeds=result.pooled_prompt_embeds)
                if result.latents is not None:
                    result_value['latents'] = result.latents
            else:
                raise ValueError('unexpected shared model stage output')
            validate_model_tensors(result_value, task.outputs[0], device=self.pipeline.raw._execution_device)
            result = result_value
        return (p.ArtifactValue(task.outputs[0], result),)


def register_adapter(native=None):
    native = native or control()
    class Adapter(native.interfaces.RuntimeV2Adapter):
        model_class_name = MODEL_CLASS
        @property
        def supported_task_kinds(self):
            return native.common.STANDARD_TASK_KINDS
        def normalize_request(self, request, denoise_chunk_size):
            validate_request(request)
            if request.chunk_size != denoise_chunk_size:
                raise ValueError('runner changed the requested denoising chunk size')
            return request
        def build_task_compiler(self, default_denoise_chunk_size, *, od_config=None, pipeline=None):
            return Compiler(native)
        def build_executors(self, pipeline):
            executor = Executor(native, pipeline, codecs(native))
            return {kind: executor for kind in self.supported_task_kinds}
        def validate_pipeline(self, pipeline, od_config):
            if not isinstance(pipeline, rounds.SD35RoundPipeline) or getattr(od_config, 'model_class_name', None) != MODEL_CLASS:
                raise ValueError('native worker did not load the actual shared SD3.5 model')
    registered = native.registry.get_registered_runtime_v2_adapters()
    if MODEL_CLASS in registered:
        if getattr(registered[MODEL_CLASS], '_service_sha256', None) != sha(Path(__file__).read_bytes()):
            raise ValueError('another adapter already owns the model registry key')
        return registered[MODEL_CLASS]
    adapter = Adapter()
    adapter._service_sha256 = sha(Path(__file__).read_bytes())
    native.registry.register_runtime_v2_adapter(adapter)
    return adapter


class SharedWorkerExtension:
    """Official WorkerWrapperBase extension hook; native device/vLLM init stays intact."""
    def re_init_pipeline(self, config):
        source_binding(config)
        abi_receipt(config)
        if self.model_runner.pipeline is not None:
            raise ValueError('extension requires the official skip-initial-load path')
        import torch
        from diffusers import StableDiffusion3Pipeline
        from ditserve.models.xdit.sd35 import pipeline_class
        from ditserve.models.xdit.runtime import XDitRuntime, engine_config
        from ditserve.models.xdit import XDIT_REVISION
        from ditserve.runtime.model_integrity import verify_model_manifest
        from xfuser.core.distributed import get_runtime_state
        integrity = verify_model_manifest(config['model_root'], config['model_manifest'], config['model_manifest_sha256'])
        raw = StableDiffusion3Pipeline.from_pretrained(config['model_root'], torch_dtype=torch.bfloat16,
                                                       local_files_only=True).to(self.device)
        if (raw.vae_scale_factor, raw.transformer.config.patch_size, raw.transformer.config.in_channels,
                raw.transformer.config.joint_attention_dim, raw.transformer.config.pooled_projection_dim) != (8, 2, 16, 4096, 2048):
            raise ValueError('actual shared model geometry differs from the source-bound codec')
        runtime = XDitRuntime.__new__(XDitRuntime)
        runtime.pool = borrowed_gfc_pool(self.device)
        runtime.config = engine_config(rounds.SD35_MODEL, 2, 'math', torch.bfloat16)
        runtime.pipe = pipeline_class()(raw, runtime.config)
        runtime.state = get_runtime_state()
        runtime.pool.activate((self.rank,), runtime.state)
        runtime.revision = XDIT_REVISION
        pipeline = rounds.SD35RoundPipeline(runtime, provenance=dict(model_integrity=integrity,
            group_ownership='official GF-DiT', shared_worker_sha256=rounds.FROZEN_WORKER_SHA256))
        pipeline.od_config = self.od_config
        pipeline.transformer = pipeline.pipe.transformer
        self.model_runner.pipeline = pipeline
        register_adapter()


def synthetic_estimator(spec, seq_len, stage):
    if spec.tp != 1 or spec.cfg != 1 or spec.sp not in (1, 2) or stage not in staged.STAGES:
        raise ValueError('functional fixture received an unsupported stage or parallel form')
    # Explicit logic fixture. It is not loaded or labelled as measured calibration.
    return 1000. / spec.sp if stage == 'dit_step_chunk' else 100.


def two_requests(protocol, *, deadline_ms=60000.):
    prompts = ('A red fox standing beside a mountain lake.', 'A white rabbit in a field of blue flowers.')
    return [ServiceRequest('shared-sd35-' + str(size), asdict(request_plan(protocol, prompt, 42 + index,
                'sd35-' + str(size))), 2, 0., deadline_ms)
            for index, (size, prompt) in enumerate(zip((512, 1024), prompts))]


def record_events(pool, result):
    """Read-only event observer delegates every event to the original aggregator."""
    original = pool._consume_event
    def observe(event):
        row = dict(type=type(event).__name__)
        for name in ('request_id', 'task_id', 'group_id', 'worker_rank', 'message'):
            if hasattr(event, name):
                row[name] = getattr(event, name)
        if hasattr(event, 'kind'):
            row['kind'] = event.kind.value
        row['observed_monotonic_ns'] = time.monotonic_ns()
        result.append(row)
        return original(event)
    pool._consume_event = observe




def task_record(task):
    return dict(task_id=task.task_id, request_id=task.request_id, kind=task.kind.value,
                step_range=asdict(task.step_range) if task.step_range else None)


def record_dispatches(pool, rows):
    original = pool.dispatch
    def observe(task, inline_inputs, release_after_exec_artifact_ids=()):
        group = pool.topology.get_group(task.group_id)
        rows.append(dict(**task_record(task), group_id=task.group_id, ranks=list(group.ranks),
                         parallel_spec=asdict(task.parallel_spec)))
        return original(task, inline_inputs, release_after_exec_artifact_ids)
    pool.dispatch = observe


def record_migrations(pool, rows):
    """Observe the original engine callbacks; never add or change a migration."""
    original = pool.begin_migration
    def observe(plan, *, on_done, on_error, timeout_s=None):
        row = dict(request_id=plan.request_id, src_group_id=plan.src_group_id,
                   dst_group_id=plan.dst_group_id, artifacts=[a.handle.artifact_id for a in plan.artifacts],
                   status='started')
        rows.append(row)
        def done(schedule):
            row['status'] = 'completed'
            row['edges'] = len(schedule.edges) if schedule is not None else 0
            row['remote_edges'] = sum(not edge.local_copy for edge in schedule.edges) if schedule else 0
            return on_done(schedule)
        def failed(error):
            row.update(status='failed', error=str(error))
            return on_error(error)
        return original(plan, on_done=done, on_error=failed, timeout_s=timeout_s)
    pool.begin_migration = observe


def assess(result):
    """Independent all-offered gate; zero migration never becomes migration coverage."""
    if result.get('expected_requests') != 2 or result.get('completed_requests') != 2:
        raise ValueError('both offered requests must complete')
    if result.get('qualification_eligible') is not False or result.get('allows') != []:
        raise ValueError('a functional service check cannot grant paper qualification')
    if result.get('costs_are_measurements') is not False or result.get('upstream_patches') != []:
        raise ValueError('cost/source scope changed')
    requests = [ServiceRequest(**row) for row in result.get('requests', [])]
    for request in requests:
        validate_request(request)
    if len(requests) != 2 or len({r.request_id for r in requests}) != 2 or sorted(r.image['height'] for r in requests) != [512, 1024]:
        raise ValueError('two distinct mixed-resolution requests are required')
    if result.get('native_pool_started') is not True or result.get('native_workers_exited') is not True:
        raise ValueError('real native worker lifetime is incomplete')
    pids = result.get('native_worker_pids', {})
    if set(pids) != {'0', '1'} or any(type(pid) is not int or pid <= 0 for pid in pids.values()) or len(set(pids.values())) != 2:
        raise ValueError('two real worker processes are required')
    if result.get('native_worker_exitcodes') != [0, 0]:
        raise ValueError('native workers did not exit cleanly')
    events = result.get('events', {}).get('task_events', [])
    from collections import Counter
    planned = result.get('planned_tasks', [])
    dispatches = result.get('dispatches', [])
    ids = [row['task_id'] for row in planned]
    if len(ids) != 38 or len(set(ids)) != 38 or Counter(row['task_id'] for row in dispatches) != Counter(ids):
        raise ValueError('every original DAG task must dispatch exactly once')
    plan_by_id = {row['task_id']: row for row in planned}
    expected_events = []
    for row in dispatches:
        if {key: row[key] for key in ('task_id', 'request_id', 'kind', 'step_range')} != plan_by_id[row['task_id']]:
            raise ValueError('native dispatch changed the original DAG interval or request')
        ranks = row['ranks']
        if (not ranks or len(ranks) not in (1, 2) or len(set(ranks)) != len(ranks)
                or any(type(rank) is not int or rank not in (0, 1) for rank in ranks)
                or row['parallel_spec'].get('sp') != len(ranks)
                or row['parallel_spec'].get('tp') != 1 or row['parallel_spec'].get('cfg') != 1):
            raise ValueError('native dispatch parallel group is invalid')
        expected_events.extend((row['task_id'], row['request_id'], row['group_id'], rank) for rank in ranks)
    actual_events = [(row.get('task_id'), row.get('request_id'), row.get('group_id'), row.get('worker_rank'))
                     for row in events if row.get('kind') == 'task_exec_end']
    if Counter(actual_events) != Counter(expected_events):
        raise ValueError('every actual dispatch rank must complete exactly once')
    for request in requests:
        intervals = sorted((row['step_range']['start'], row['step_range']['end']) for row in planned
                           if row['request_id'] == request.request_id and row['step_range'])
        if intervals != [(index, index + 2) for index in range(0, 28, 2)]:
            raise ValueError('native denoising coverage differs from all original 28 steps')
    if any(row.get('kind') in ('task_failed', 'request_failed') for row in events):
        raise ValueError('native task/request failure must remain visible')
    if any(row.get('status') != 'completed' for row in result.get('migrations', [])):
        raise ValueError('original data-plane migration did not complete')
    offered = {request.request_id for request in requests}
    for key in ('completed', 'comparisons'):
        rows = result.get(key, [])
        if len(rows) != 2 or {row.get('request_id') for row in rows} != offered:
            raise ValueError('completion/reference inventory differs from all offered requests')
    for request in requests:
        completed = [row for row in result.get('completed', []) if row.get('request_id') == request.request_id]
        compared = [row for row in result.get('comparisons', []) if row.get('request_id') == request.request_id]
        if len(completed) != 1 or len(compared) != 1:
            raise ValueError('missing or duplicate completion/reference')
        expected_shape = [1, 3, request.image['height'], request.image['width']]
        for desc in (completed[0]['output'], compared[0]['reference']):
            if (desc.get('shape') != expected_shape or desc.get('dtype') != 'torch.bfloat16'
                    or desc.get('finite') is not True or desc.get('retained_array') is not False
                    or not re.fullmatch('[0-9a-f]{64}', desc.get('sha256', ''))
                    or any(type(desc.get(key)) not in (int, float) or not math.isfinite(desc[key]) for key in ('minimum', 'maximum'))
                    or desc['minimum'] > desc['maximum']):
                raise ValueError('actual numerical tensor descriptor is incomplete')
        comparison = compared[0]['comparison']
        if (comparison.get('passed') is not True or comparison.get('atol') != 1e-5
                or comparison.get('rtol') != 1e-5 or comparison.get('elements') != math.prod(expected_shape)):
            raise ValueError('predeclared numerical tolerance did not pass')
        for key in ('max_abs_error', 'rmse', 'relative_l2_error'):
            value = comparison.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid numerical error evidence')
        reference, actual = compared[0]['reference'], completed[0]['output']
        if (comparison['max_abs_error'] > 1e-5 + 1e-5 * max(abs(reference['minimum']), abs(reference['maximum']))
                or comparison['rmse'] > comparison['max_abs_error'] + 1e-15):
            raise ValueError('reported error contradicts the numerical tolerance')
        bitwise = comparison.get('bitwise_equal')
        if type(bitwise) is not bool or bitwise != (reference['sha256'] == actual['sha256']):
            raise ValueError('bitwise equality and complete tensor hashes disagree')
        if bitwise and any(comparison[key] != 0 for key in ('max_abs_error', 'rmse', 'relative_l2_error')):
            raise ValueError('bitwise equality contradicts numerical errors')
    return True


def summarize_events(events):
    from collections import Counter
    return dict(count=len(events), by_kind=dict(Counter(row.get('kind', row['type']) for row in events)),
                task_events=events)


def run_service(config, result, timeout_seconds):
    source_binding(config)
    result['abi'] = abi_receipt(config)
    import torch
    from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionParallelConfig
    if torch.cuda.device_count() != 2 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('launch one controller with two visible GPUs; official pool owns rank spawning')
    result['raw_gpu_uuids'] = [str(torch.cuda.get_device_properties(i).uuid) for i in range(2)]
    result['gpu_uuids'] = [normalized_uuid(value) for value in result['raw_gpu_uuids']]
    if len(set(result['gpu_uuids'])) != 2:
        raise ValueError('two distinct actual GPU UUIDs are required')
    os.environ['VLLM_RUNTIME_V2_MIGRATE_GFC_P2P'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    native = control(); register_adapter(native)
    requests = two_requests(upstream.read_json(Path(config['protocol']).read_bytes()))
    result['requests'] = [asdict(request) for request in requests]
    od = OmniDiffusionConfig(model=str(config['model_root']), model_class_name=MODEL_CLASS,
        dtype=torch.bfloat16, num_gpus=2, parallel_config=DiffusionParallelConfig(), enforce_eager=True,
        enable_cpu_offload=False, enable_layerwise_offload=False, enable_sleep_mode=False,
        cache_backend='none', quantization=None, enable_runtime_v2=True,
        runtime_v2_scheduler_policy='edf_best_fit', runtime_v2_collective_backend='gfc',
        runtime_v2_edf_greedy_sp_sizes=[1, 2], runtime_v2_denoise_chunk_size=2,
        worker_extension_cls='external_gfdit_service.SharedWorkerExtension', custom_pipeline_args=dict(config))
    events, outputs = [], {}
    runner = None
    result['stage'] = 'native_worker_bootstrap'
    try:
        runner = native.runner.RuntimeV2Runner.__new__(native.runner.RuntimeV2Runner)
        runner.__init__(default_step_chunk_size=2, scheduler_policy='edf_best_fit',
            task_runtime_estimator=synthetic_estimator, omni_diffusion_config=od)
        if type(runner.worker_pool) is not native.multiproc_worker.MultiprocWorkerPool:
            raise ValueError('original multiprocess worker pool was replaced')
        result['native_worker_pids'] = {str(rank): handle.process.pid for rank, handle in runner.worker_pool.worker_handles.items()}
        result['native_pool_started'] = True
        record_events(runner.worker_pool, events)
        record_migrations(runner.worker_pool, result.setdefault('migrations', []))
        record_dispatches(runner.worker_pool, result.setdefault('dispatches', []))
        result['planned_tasks'] = [task_record(task) for request in requests
            for task in Compiler(native).compile_request(request).tasks.values()]
        result['stage'] = 'two_requests_in_original_service'
        started = time.monotonic(); end = started + timeout_seconds
        for request in requests:
            if runner.submit(request, denoise_chunk_size=2) != request.request_id:
                raise ValueError('official runner changed a request identity')
        while len(outputs) < len(requests):
            if time.monotonic() >= end:
                raise TimeoutError('all offered requests remain in denominator after timeout')
            runner.worker_pool.check_health()
            runner.poll_once(timeout_s=.01)
            for request in requests:
                if request.request_id in outputs:
                    continue
                status, payload = runner.get_request_status(request.request_id)
                if status == 'failed':
                    raise RuntimeError('official service request failed: ' + str(payload))
                if status == 'finished':
                    if (payload.error or payload.custom_output.get('request_id') != request.request_id
                            or payload.custom_output.get('image') != request.image
                            or payload.custom_output.get('request_sha256') != canonical(request.image)
                            or payload.custom_output.get('cursor') != 28):
                        raise ValueError('native completion lost request/continuation identity')
                    output = payload.output.detach().cpu()
                    if tuple(output.shape) != (1, 3, request.image['height'], request.image['width']):
                        raise ValueError('native completion has wrong decoded image geometry')
                    outputs[request.request_id] = output
                    result.setdefault('completed', []).append(dict(request_id=request.request_id,
                        observed_client_ms=(time.monotonic() - started) * 1000, output=tensor_summary(output)))
                    result['completed_requests'] = len(outputs)
                    runner.release_request(request.request_id)
        result['completed_requests'] = len(outputs)
    finally:
        if runner is not None and hasattr(runner, 'worker_pool'):
            processes = [handle.process for handle in runner.worker_pool.worker_handles.values()]
            if hasattr(runner, 'scheduler'):
                runner.close()
            else:
                runner.worker_pool.shutdown()
            result['native_workers_exited'] = all(not process.is_alive() for process in processes)
            result['native_worker_exitcodes'] = [process.exitcode for process in processes]
        result['events'] = summarize_events(events)
    if not result.get('native_workers_exited'):
        raise RuntimeError('native workers did not exit before reference model load')
    result['stage'] = 'independent_normal_shared_reference'
    pipeline = rounds.SD35RoundPipeline.from_pretrained(config['model_root'],
        shared_python_root=config['shared_python_root'], model_manifest_path=config['model_manifest'],
        model_manifest_sha256=config['model_manifest_sha256'], attention='math', device='cuda:0')
    try:
        comparisons = []
        for request in requests:
            image = rounds.ImageRequest(**request.image)
            with torch.inference_mode():
                reference = pipeline.pipe(prompt=image.prompt, negative_prompt=image.negative_prompt,
                    height=image.height, width=image.width, num_inference_steps=image.steps,
                    guidance_scale=image.guidance_scale, max_sequence_length=image.max_sequence_length,
                    generator=torch.Generator(device='cpu').manual_seed(image.seed), output_type='pt').images
                torch.cuda.synchronize()
            comparison = compare_tensors(reference.cpu(), outputs[request.request_id])
            comparisons.append(dict(request_id=request.request_id, comparison=comparison,
                                    reference=tensor_summary(reference)))
        result['comparisons'] = comparisons
        result['numerical_check_passed'] = all(row['comparison']['passed'] for row in comparisons)
        if not result['numerical_check_passed']:
            raise ValueError('full service output differs from the independent shared reference')
    finally:
        pipeline.runtime.close()
    result['native_migrations_observed'] = sum(row['status'] == 'completed' for row in result['migrations'])
    result['remote_migration_edges'] = sum(row.get('remote_edges', 0) for row in result['migrations'])
    result['migration_coverage'] = ('remote_transfer_observed' if result['remote_migration_edges'] else
                                    'only_local_copy' if result['native_migrations_observed'] else 'not_observed')
    assess(result)
    result['stage'] = 'completed'
    result['status'] = 'functional_service_passed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit', 'run'])
    for name in ('source-root', 'source-manifest', 'gfc-source-root', 'gfc-source-manifest',
                 'shared-python-root', 'model-root', 'model-manifest', 'protocol'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('source-manifest-sha256', 'gfc-source-manifest-sha256', 'model-manifest-sha256', 'protocol-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 3600:
        parser.error('timeout must be 1..3600 seconds')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ('mode', 'output_dir', 'timeout_seconds')}
    dependencies = (sys.modules[__name__], upstream, staged, rounds, gfc_source,
                    sys.modules['external_sd35_round_check'])
    config['adapter_files'] = {str(Path(module.__file__).resolve()): sha(Path(module.__file__).read_bytes()) for module in dependencies}
    if args.mode == 'audit':
        binding = source_binding(config)
        print(json.dumps(dict(source=binding, upstream_patches=[], supported_model='sd35',
            request_shapes=[512, 1024], native_abi_required=REQUIRED_ABI,
            qualification_eligible=False, allows=[], gpu_validation='pending'), allow_nan=False))
        return 0
    if args.output_dir is None:
        parser.error('run requires --output-dir')
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path('/shared/output/DITSERVE')) or output.exists():
        parser.error('run requires a new directory under /shared/output/DITSERVE')
    check_storage()
    output.mkdir(parents=True, exist_ok=False)
    stop = threading.Event(); watcher = threading.Thread(target=watchdog, args=(stop,), daemon=True); watcher.start()
    result = dict(schema='ditserve.gfdit-service-check.v1', status='failed', stage='source_binding',
        expected_requests=2, completed_requests=0, native_pool_started=False,
        qualification_eligible=False, allows=[], costs_are_measurements=False,
        cost_scope='explicit synthetic functional fixture, never formal serving performance',
        upstream_patches=[], complete_model_matrix=False, http_frontend_executed=False,
        numerical_check_passed=False, scope='official full-process runtime_v2 service with shared SD3.5 model adapter')
    code = 1
    try:
        result['source'] = source_binding(config)
        run_service(config, result, args.timeout_seconds)
        source_binding(config)
        code = 0
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), failed_stage=result.get('stage'))
    finally:
        stop.set(); watcher.join(timeout=6)
        result['storage'] = check_storage()
        with (output / 'summary.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], completed_requests=result['completed_requests'],
                         qualification_eligible=False, summary=str(output / 'summary.json'))))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
