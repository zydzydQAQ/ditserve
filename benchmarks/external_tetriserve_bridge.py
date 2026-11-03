"""Borrow TetriServe's actual groups for the frozen shared SD3.5 backend.

No group creation, scheduling policy, request queue or latent transport is
implemented here. TetriServe owns NCCL/Gloo groups and LatentManager. The adapter
only binds existing groups and validates request/continuation identity. Multi-GPU
numerical correctness is unverified until the corresponding GPU checks pass.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys

from external_sd35_round import SD35RoundPipeline, ImageRequest, verify_shared_sources, SD35_MODEL, SD35_REVISION, FROZEN_WORKER_SHA256


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks_tuple(ranks, world):
    ranks = tuple(ranks)
    if (type(world) is not int or world not in (1, 2) or not ranks
            or any(type(r) is not int or not 0 <= r < world for r in ranks)
            or ranks != tuple(sorted(set(ranks)))):
        raise ValueError('this initial port requires sorted unique ranks in a one/two-GPU world')
    return ranks


def existing_groups(official, world):
    if getattr(official, 'generated', False) is not True:
        raise ValueError('official process groups have not been generated')
    keys = {}
    for key in official.all_process_groups_map:
        ranks = ranks_tuple(ast.literal_eval(key), world)
        if key != str(list(ranks)) or key not in official.all_cpu_groups_map:
            raise ValueError('official CUDA/Gloo group inventories differ')
        keys[ranks] = key
    required = {tuple(range(world))} | {(r,) for r in range(world)}
    if not required.issubset(keys):
        raise ValueError('official world or singleton process group missing')
    return keys


def borrowed_pool(official, rank, world, device):
    """Return common xfuser coordinators referencing existing official groups."""
    import torch
    from ditserve.models.xdit.groups import ProcessGroupPool
    from ditserve.models.xdit import collectives
    if collectives.backend() != "nccl":
        raise ValueError("this port requires the original NCCL collective backend")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() != rank:
        raise ValueError('official distributed rank is not initialized')
    if torch.distributed.get_world_size() != world:
        raise ValueError('official distributed world differs')
    keys = existing_groups(official, world)

    class OfficialGroups:
        backend = torch.distributed.get_backend()
        timeout_seconds = 90
        def __init__(self):
            self.rank = rank
            self.groups = tuple(keys)
        def group(self, selected):
            selected = ranks_tuple(selected, world)
            return official.get_process_group(list(selected), device='cuda')

    class BorrowedPool(ProcessGroupPool):
        def close(self):
            # Common Pool.close normally owns/destroys metadata Gloo groups.
            # Here TetriServe owns them, including their lifetime.
            owned_groups = self.groups
            self.groups = None
            try:
                super().close()
                self.metadata.clear()
            finally:
                self.groups = owned_groups

    view = OfficialGroups()
    pool = BorrowedPool(rank, world, device, view.backend, view)
    pool.metadata = {r: official.get_process_group(list(r), device='cpu') for r in keys}
    for selected in keys:
        if rank in selected:
            for group in (view.group(selected), pool.metadata[selected]):
                if tuple(torch.distributed.get_process_group_ranks(group)) != selected:
                    raise ValueError('borrowed process-group rank order differs')
    pool.install()
    return pool


def load_port_config():
    path = Path(os.environ['DITSERVE_TETRI_PORT_CONFIG'])
    if sha(path) != os.environ['DITSERVE_TETRI_PORT_CONFIG_SHA256']:
        raise ValueError('TetriServe port config SHA256 differs')
    config = json.loads(path.read_bytes())
    if (config.get('schema') != 'ditserve.tetriserve-sd35-port.v1'
            or config.get('model') != SD35_MODEL or config.get('revision') != SD35_REVISION
            or config.get('shared_worker_sha256') != FROZEN_WORKER_SHA256
            or config.get('guidance_scale') != 4.5 or config.get('max_sequence_length') != 256
            or config.get('attention') != 'math'):
        raise ValueError('unsupported immutable SD3.5 continuation profile')
    for h, w in config.get('resolutions', []):
        ImageRequest('', height=h, width=w)
    if not config.get('resolutions'):
        raise ValueError('port config must declare allowed resolutions')
    manifest_path = Path(config['derived_manifest'])
    if sha(manifest_path) != config['derived_manifest_sha256']:
        raise ValueError('derived source manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    root = manifest_path.parent
    if manifest.get('schema') != 'ditserve.tetriserve-derived-source.v1':
        raise ValueError('unsupported derived source manifest')
    if manifest.get('upstream_commit') != '414ec2956c09466ccd7842fda9851511462bd42a':
        raise ValueError('derived source upstream identity differs')
    for relative, expected in manifest['files'].items():
        rel = PurePosixPath(relative)
        if rel.is_absolute() or '..' in rel.parts or str(rel) != relative:
            raise ValueError('invalid derived source path')
        p = root / relative
        if p.resolve() != p or not p.is_file() or sha(p) != expected:
            raise ValueError('derived source file differs: ' + relative)
    observed = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != manifest_path}
    if observed != set(manifest['files']) or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('derived source inventory differs')
    if Path(__file__).resolve() != root / 'external_tetriserve_bridge.py':
        raise ValueError('bridge is not executing from its bound derived tree')
    round_module = sys.modules[SD35RoundPipeline.__module__]
    round_path = Path(round_module.__file__).resolve()
    if (round_path != root / 'external_sd35_round.py'
            or sha(round_path) != manifest['files']['external_sd35_round.py']):
        raise ValueError('round implementation is not executing from its bound derived tree')
    return config


def validate_server_args(args):
    config = load_port_config()
    if (args.engine_config.model_config.model != SD35_MODEL or args.nnodes != 1
            or args.nproc_per_node not in (1, 2) or args.fake_engine or args.skip_latent_transfer):
        raise ValueError('SD3.5 port requires one node, real workers/transfer and one or two GPUs')
    if args.schedule_logic != 'dyn_slo_schedule_global_dp':
        raise ValueError('this official TetriServe port requires its native global DP scheduler')
    from tetriserve.utils import get_gpu_name
    profile = Path(args.perf_model_dir) / 'sd35' / get_gpu_name() / 'e2e_scaling_efficiency.json'
    if sha(profile) != config['cost_profile_sha256']:
        raise ValueError('native scheduler cost profile differs')
    rows = json.loads(profile.read_bytes())
    observed = set()
    for row in rows:
        if row['ring_degree'] != 1:
            continue
        for field in ('avg_text_encoding_time', 'avg_denoising_time', 'steps'):
            v = row[field]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError('native measured cost must be finite and positive')
        observed.add((row['height'], row['width'], row['bs'], row['ulysses_degree']))
    required = {(h, w, 1, sp) for h, w in config['resolutions']
                for sp in (1, 2) if sp <= args.nproc_per_node}
    if not required.issubset(observed):
        raise ValueError('native scheduler profile is missing a supported shape/SP cell')


def validate_request(reqs, futures=None):
    config = reqs.input_config
    batch = config.batch_size
    if type(batch) is not int or batch < 1:
        raise ValueError('invalid scheduled batch size')
    if (len(reqs.sub_req_ids) != batch or len(set(reqs.sub_req_ids)) != batch
            or len(reqs.is_last_steps) != batch or len(reqs.origin_steps) != batch
            or not isinstance(config.prompt, list) or len(config.prompt) != batch):
        raise ValueError('scheduled batch metadata is not aligned per request')
    if not isinstance(config.negative_prompt, str):
        raise ValueError('initial port requires an explicit scalar negative prompt for each batch')
    if config.guidance_scale != 4.5 or config.max_sequence_length != 256:
        raise ValueError('scheduled request differs from the immutable CFG/text profile')
    if (type(reqs.start_step) is not int or type(reqs.end_step) is not int
            or not 0 <= reqs.start_step < reqs.end_step
            or type(config.num_inference_steps) is not int
            or config.num_inference_steps != reqs.end_step - reqs.start_step):
        raise ValueError('scheduled round step interval differs from its executed step count')
    for index, sub_id in enumerate(reqs.sub_req_ids):
        total = reqs.origin_steps[index]
        ImageRequest(config.prompt[index], config.negative_prompt, config.height, config.width,
                     total, config.seed, config.guidance_scale, config.max_sequence_length)
        if (type(reqs.is_last_steps[index]) is not bool or reqs.end_step > total
                or reqs.is_last_steps[index] != (reqs.end_step == total)):
            raise ValueError('final-step marker or total steps differs')
        future = (futures or {}).get(sub_id)
        if reqs.start_step == 0:
            if future is not None:
                raise ValueError('attempt to replay from step zero with an existing continuation')
        elif (future is None or future.sub_req_id != sub_id or future.step_range is None
              or len(future.step_range) != 2 or future.step_range[1] != reqs.start_step):
            raise ValueError('continuation cursor is missing, repeated or skipped')


def validate_transferred_batch(reqs, prompt_list, pooled_list, latent_list, times_list, sigma_list):
    """Validate each complete five-tensor request before a shared batch call."""
    import torch
    batch = reqs.input_config.batch_size
    arrays = (prompt_list, pooled_list, latent_list, times_list, sigma_list)
    if any(len(items) != batch for items in arrays):
        raise ValueError('incomplete five-tensor batch')
    first_t, first_s = times_list[0], sigma_list[0]
    for i, (pe, pp, latent, times, sigmas) in enumerate(zip(*arrays)):
        total = reqs.origin_steps[i]
        if (pe.ndim != 3 or pp.ndim != 2 or pe.shape[0] != 2 or pp.shape[0] != 2
                or latent.ndim != 4 or latent.shape[0] != 1
                or times.ndim != 1 or sigmas.ndim != 1 or len(times) != total or len(sigmas) != total + 1):
            raise ValueError('each transferred request must retain [unconditional, conditional] and its full schedule')
        if not torch.equal(times, first_t) or not torch.equal(sigmas, first_s):
            raise ValueError('cannot batch requests with different timestep/sigma schedules')
        for value in (pe, pp, latent, times, sigmas):
            if value.device != latent.device or not torch.isfinite(value).all().item():
                raise ValueError('invalid transferred tensor device or finite-value check')


def activate_request(pipeline, reqs, futures=None):
    import torch
    validate_request(reqs, futures)
    ranks = ranks_tuple(reqs.parallel_ranks, torch.distributed.get_world_size())
    if reqs.attn_ring_degree != 1 or reqs.attn_ulysses_degree != len(ranks) or reqs.sp_degree != len(ranks):
        raise ValueError('this port supports only native SP1/SP2 Ulysses forms')
    if [reqs.input_config.height, reqs.input_config.width] not in pipeline.port_config['resolutions']:
        raise ValueError('scheduled resolution is outside the calibrated profile')
    from tetriserve.distributed.globals import set_seq_parallel_pg
    set_seq_parallel_pg(len(ranks), 1, torch.distributed.get_rank(), list(ranks), len(ranks))
    pipeline.runtime.pool.activate(ranks, pipeline.runtime.state, ulysses_degree=len(ranks), ring_degree=1)
    pipeline.runtime.bind_global()


class TetriSharedSD35Pipeline(SD35RoundPipeline):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, engine_config, torch_dtype, **kwargs):
        config = load_port_config()
        if pretrained_model_name_or_path != SD35_MODEL or engine_config.model_config.model != SD35_MODEL or kwargs:
            raise ValueError('explicit SD3.5 registry identity required; checkpoint aliases are not accepted')
        root = verify_shared_sources(config['shared_python_root'])
        sys.path.insert(0, str(root))
        for name, module in list(sys.modules.items()):
            if name == 'xfuser' or name.startswith('xfuser.') or name == 'ditserve' or name.startswith('ditserve.'):
                origin = getattr(module, '__file__', None)
                if origin and not Path(origin).resolve().is_relative_to(root):
                    raise ValueError('a different shared backend is already imported')
        import torch
        from diffusers import StableDiffusion3Pipeline
        from ditserve.models.xdit.sd35 import pipeline_class
        from ditserve.models.xdit.runtime import XDitRuntime, engine_config as common_engine_config
        from ditserve.runtime.model_integrity import verify_model_manifest
        from xfuser.core.distributed.runtime_state import get_runtime_state
        from tetriserve.distributed.globals import PROCESS_GROUP
        if torch_dtype is not torch.bfloat16:
            raise ValueError('frozen model requires bfloat16')
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        ranks_tuple(range(world), world)
        model_manifest = json.loads(Path(config['model_manifest']).read_bytes())
        if model_manifest.get('model') != SD35_MODEL or model_manifest.get('revision') != SD35_REVISION:
            raise ValueError('model manifest identity differs')
        integrity = verify_model_manifest(config['model_path'], config['model_manifest'], config['model_manifest_sha256'])
        device = f'cuda:{rank}'
        raw = StableDiffusion3Pipeline.from_pretrained(config['model_path'], torch_dtype=torch.bfloat16,
                                                       local_files_only=True).to(device)
        # Use XDitRuntime's existing activation/info methods, with borrowed groups
        # instead of its owning constructor (which would allocate new Gloo groups).
        runtime = XDitRuntime.__new__(XDitRuntime)
        runtime.pool = borrowed_pool(PROCESS_GROUP, rank, world, device)
        runtime.config = common_engine_config(SD35_MODEL, world, config['attention'], torch.bfloat16)
        runtime.pipe = pipeline_class()(raw, runtime.config)
        runtime.state = get_runtime_state()
        runtime.pool.activate((rank,), runtime.state)
        from ditserve.models.xdit import XDIT_REVISION
        runtime.revision = XDIT_REVISION
        result = cls(runtime, guidance_scale=config['guidance_scale'], max_sequence_length=config['max_sequence_length'],
            provenance={'shared_worker_sha256': FROZEN_WORKER_SHA256, 'model_integrity': integrity,
                        'group_ownership': 'official TetriServe', 'new_groups_created_by_bridge': 0})
        result.port_config = config
        result.engine_config = engine_config
        return result

    def to(self, device):
        if str(device) != str(self.runtime.pool.device):
            raise ValueError('cannot move an initialized distributed worker to another device')
        return self
