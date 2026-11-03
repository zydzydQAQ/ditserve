"""Reproduce the explicit SD3.5/shared-xfuser port in a new TetriServe tree.

The frozen official source is never edited. Every replacement requires an exact
upstream anchor and records before/after SHA256. Scheduler, request tracker,
LatentManager, ZMQ and distributed group creation remain byte-identical. The
result is a port awaiting GPU validation, not a qualified system reproduction.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

import external_tetriserve as upstream

PORT_SCHEMA = 'ditserve.tetriserve-derived-source.v1'
ADAPTER_FILES = ('external_tetriserve_bridge.py', 'external_sd35_round.py')
PRESERVED = (
    'tetriserve/scheduler/dyn_step_scheduler.py',
    'tetriserve/scheduler/scheduler.py', 'tetriserve/engine/request_tracker.py',
    'tetriserve/light_ray/latent_manager.py', 'tetriserve/light_ray/latent_future.py',
    'tetriserve/zmq_utils/comm.py', 'tetriserve/distributed/globals.py',
    'tetriserve/distributed/distributed.py', 'tetriserve/engine/proc_launcher.py',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError('upstream patch anchor is not unique: ' + label)
    return text.replace(old, new, 1)


def patched_sources(root):
    root = Path(root)
    changed = {}
    path = 'tetriserve/scheduler/perf_model.py'
    changed[path] = replace_once((root/path).read_text(), 'import pandas as pd\n',
        '# Unused pandas import removed by the explicit SD3.5 port.\n', path + ':unused-dependency')
    path = 'tetriserve/__init__.py'
    changed[path] = "SUPPORTED_MODEL_DICT = {\n    'sd35': 'stabilityai/stable-diffusion-3.5-medium',\n}\n"
    # Model kernels come from the shared xfuser package. Preserve the official
    # config/distributed modules without eagerly importing unused legacy kernels.
    changed['tetriserve/xfuser/__init__.py'] = (
        '"""SD3.5 port: official configuration/communication, shared model backend."""\n'
        'from .config import xFuserArgs, EngineConfig\n'
        "__all__ = ['xFuserArgs', 'EngineConfig']\n")
    path = 'tetriserve/config/args.py'
    changed[path] = replace_once((root/path).read_text(),
        '    def __post_init__(self):\n',
        '    def __post_init__(self):\n'
        '        from external_tetriserve_bridge import validate_server_args\n'
        '        validate_server_args(self)\n', path)
    path = 'tetriserve/xfuser/config/config.py'
    text = replace_once((root/path).read_text(),
        'class InputConfig:\n', 'class InputConfig:\n    guidance_scale: float = 4.5\n', path)
    changed[path] = replace_once(text,
        '                len(self.prompt) == len(self.negative_prompt) or len(self.negative_prompt) == 0\n',
        '                isinstance(self.negative_prompt, str)\n'
        '                or len(self.prompt) == len(self.negative_prompt) or len(self.negative_prompt) == 0\n',
        path + ':scalar-negative-broadcast')
    path = 'tetriserve/server/api_server.py'
    text = (root/path).read_text()
    text = replace_once(text, '    guidance_scale: float = 0.0\n',
        '    guidance_scale: float = 4.5\n'
        '    negative_prompt: str = "blurry, low quality"\n'
        '    max_sequence_length: int = 256\n', path + ':request')
    text = replace_once(text, '            prompt=request.prompt,\n',
        '            prompt=request.prompt,\n'
        '            negative_prompt=request.negative_prompt,\n'
        '            guidance_scale=request.guidance_scale,\n'
        '            max_sequence_length=request.max_sequence_length,\n', path + ':forward')
    changed[path] = text
    path = 'tetriserve/engine/runner.py'
    text = (root/path).read_text()
    text = replace_once(text,
        'from tetriserve.xfuser import xFuserFluxPipeline, xFuserStableDiffusion3Pipeline\n',
        'from external_tetriserve_bridge import (\n'
        '    TetriSharedSD35Pipeline, activate_request, validate_transferred_batch,\n'
        ')\n', path + ':imports')
    text = replace_once(text,
        '    MODEL_REGISTRY = {\n'
        '        "black-forest-labs/FLUX.1-dev": xFuserFluxPipeline,\n'
        '        "black-forest-labs/FLUX.1-schnell": xFuserFluxPipeline,\n'
        '        "stabilityai/stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,\n'
        '    }',
        '    MODEL_REGISTRY = {\n'
        '        "stabilityai/stable-diffusion-3.5-medium": TetriSharedSD35Pipeline,\n'
        '    }', path + ':registry')
    text = replace_once(text, '                self.pipeline.dummy_inference_prologue()\n',
        '                self.pipeline.dummy_inference_prologue(\n'
        '                    guidance_scale=reqs.input_config.guidance_scale,\n'
        '                    max_sequence_length=reqs.input_config.max_sequence_length,\n'
        '                )\n', path + ':restore-profile')
    text = replace_once(text, '                    num_inference_steps=reqs.origin_steps[0],\n'
        '                    generator=torch.Generator(device="cuda").manual_seed(reqs.input_config.seed),\n',
        '                    num_inference_steps=reqs.origin_steps[idx],\n'
        '                    generator=torch.Generator(device="cpu").manual_seed(reqs.input_config.seed),\n'
        '                    negative_prompt=reqs.input_config.negative_prompt,\n'
        '                    guidance_scale=reqs.input_config.guidance_scale,\n'
        '                    max_sequence_length=reqs.input_config.max_sequence_length,\n', path + ':prologue')
    text = replace_once(text, '        latents = torch.cat(latents_list, dim=0)\n',
        '        validate_transferred_batch(\n'
        '            reqs, prompt_embeds_list, pooled_prompt_embeds_list,\n'
        '            latents_list, timesteps_list, sigmas_list,\n'
        '        )\n'
        '        latents = torch.cat(latents_list, dim=0)\n', path + ':batch-schedule')
    begin = '        from tetriserve.distributed.globals import set_seq_parallel_pg\n'
    end = '        t_preprocess_start = time.perf_counter()\n'
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError('native SP activation anchors differ')
    start, stop = text.index(begin), text.index(end)
    if start >= stop:
        raise ValueError('native SP activation interval is invalid')
    text = text[:start] + '        activate_request(self.pipeline, reqs, latent_future_dict)\n\n' + text[stop:]
    changed[path] = text
    for name, content in changed.items():
        ast.parse(content, filename=name)
    return {name: content.encode() for name, content in changed.items()}


def plan(source_root, source_manifest, source_manifest_sha256, *, adapter_root=None):
    manifest = upstream.verify_source(source_root, source_manifest, source_manifest_sha256)
    root = Path(source_root).resolve()
    patches = patched_sources(root)
    adapter_root = Path(adapter_root or Path(__file__).resolve().parent)
    additions = {name: (adapter_root/name).read_bytes() for name in ADAPTER_FILES}
    for name, raw in additions.items():
        ast.parse(raw.decode(), filename=name)
    original = manifest['files']
    result = {'schema': PORT_SCHEMA, 'upstream_commit': upstream.COMMIT,
              'upstream_manifest_sha256': source_manifest_sha256,
              'upstream_archive_sha256': upstream.ARCHIVE_SHA256,
              'port_applier_sha256': sha(Path(__file__).read_bytes()),
              'patches': {name: {'before_sha256': original[name], 'after_sha256': sha(raw)}
                          for name, raw in patches.items()},
              'added_adapters': {name: sha(raw) for name, raw in additions.items()},
              'preserved_native_mechanisms': {name: original[name] for name in PRESERVED},
              'scope': 'Initial SD3.5-only port; native SP1/SP2, shared xfuser, real official transport',
              'constraints': ['sorted rank groups', 'one immutable guidance/text profile per server',
                  'aligned CFG pairs and identical complete schedules within each batch',
                  'native future step_range must match the next scheduled absolute step',
                  'calibrated cost-profile coverage required before launching the native scheduler'],
              'model_execution_verified': False, 'multi_gpu_numerical_verified': False,
              'qualification_eligible': False, 'allows': []}
    files = dict(original)
    files.update({n: sha(v) for n, v in patches.items()})
    files.update({n: sha(v) for n, v in additions.items()})
    result['files'] = files
    return result, patches, additions


def apply(source_root, source_manifest, source_manifest_sha256, output, *, adapter_root=None):
    result, patches, additions = plan(source_root, source_manifest, source_manifest_sha256,
                                      adapter_root=adapter_root)
    source, output = Path(source_root).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('derived and original trees must be disjoint')
    output.mkdir(parents=True, exist_ok=False)
    # Deliberately no deletion/overwrite/cleanup on failure: preserve evidence.
    for relative in result['files']:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in patches:
            target.write_bytes(patches[relative])
        elif relative in additions:
            target.write_bytes(additions[relative])
        else:
            shutil.copyfile(source/relative, target)
        if sha(target.read_bytes()) != result['files'][relative]:
            raise ValueError('derived source write verification failed: ' + relative)
    upstream.verify_source(source, source_manifest, source_manifest_sha256)
    target = output / 'PORT_MANIFEST.json'
    with target.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return {'schema': PORT_SCHEMA, 'status': 'derived_source_created', 'root': str(output),
            'manifest_sha256': sha(target.read_bytes()), 'files': len(result['files']),
            'patches': result['patches'], 'qualification_eligible': False, 'allows': [],
            'multi_gpu_numerical_verified': False}


def scheduler_check(derived_root, derived_manifest_sha256, output):
    """Run real native scheduling twice; synthetic costs are logic fixtures only."""
    root = Path(derived_root).resolve()
    manifest_path = root / 'PORT_MANIFEST.json'
    if sha(manifest_path.read_bytes()) != derived_manifest_sha256:
        raise ValueError('derived manifest SHA256 differs')
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('schema') != PORT_SCHEMA or manifest.get('upstream_commit') != upstream.COMMIT:
        raise ValueError('derived source identity differs')
    for name, expected in manifest['files'].items():
        path = root / name
        if path.resolve() != path or not path.is_relative_to(root) or sha(path.read_bytes()) != expected:
            raise ValueError('derived source file differs: ' + name)
    output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('diagnostics must not write inside the bound source tree')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from tetriserve.scheduler.dyn_step_scheduler import DynStepScheduler
    from tetriserve.xfuser.config import InputConfig
    from tetriserve.engine.struct import RunnerOutput
    from tetriserve.light_ray.latent_future import LatentFuture
    for name, module in list(sys.modules.items()):
        if name == 'tetriserve' or name.startswith('tetriserve.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(root):
                raise ValueError('a different TetriServe source is already imported')
    rows = []
    for batch in (1, 2):
        for sp in (1, 2):
            # Artificial positive seconds, chosen to exercise both native SP
            # decisions and native batch splitting. Never used as a calibration.
            cost = {(1, 1): .01, (1, 2): .006, (2, 1): .03, (2, 2): .02}[batch, sp]
            rows.append(dict(height=512, width=512, bs=batch, ulysses_degree=sp,
                ring_degree=1, mlp_world_size=1, steps=28,
                avg_text_encoding_time=.01, avg_denoising_time=28 * cost))
    profile = output / 'synthetic-logic-costs.json'
    with profile.open('x') as stream:
        json.dump(rows, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    def describe(blocks):
        return [dict(sub_req_ids=req.sub_req_ids, ranks=req.parallel_ranks,
            start=req.start_step, end=req.end_step, steps=req.input_config.num_inference_steps,
            batch=req.input_config.batch_size, prompts=req.input_config.prompt,
            origin_steps=req.origin_steps, final=req.is_last_steps)
            for block in blocks for req in block.requests]
    records = []
    for batch in (1, 2):
        scheduler = DynStepScheduler([0, 1], 'dyn_slo_schedule_global_dp', str(profile),
                                     use_cpp_optimizer=False)
        config = InputConfig(prompt=['fox', 'hare'][:batch], negative_prompt='blurry, low quality',
                             height=512, width=512, num_inference_steps=28)
        scheduler.add_request(100, list(range(1, batch + 1)), config, arrival_time=0., latency_threshold=1000.)
        first = scheduler.schedule()
        if not first:
            raise ValueError('native scheduler returned no first block')
        completed = {}
        for block in first:
            for req in block.requests:
                if req.end_step >= 28:
                    raise ValueError('logic fixture did not leave a continuation to replan')
                futures = [LatentFuture(sid, req.parallel_ranks[0], {},
                    step_range=(req.start_step, req.end_step)) for sid in req.sub_req_ids]
                scheduler.request_tracker.update_requests(RunnerOutput(sub_req_ids=req.sub_req_ids,
                    images=futures, run_steps=req.input_config.num_inference_steps,
                    parallel_ranks=req.parallel_ranks))
                completed.update({sid: req.end_step for sid in req.sub_req_ids})
        second = scheduler.schedule()
        if not second:
            raise ValueError('native scheduler returned no continuation block')
        views = describe(first), describe(second)
        aligned = all(len(row['prompts']) == row['batch'] == len(row['origin_steps']) == len(row['sub_req_ids'])
                      for view in views for row in view)
        contiguous = all(row['start'] == completed[sid] for row in views[1] for sid in row['sub_req_ids'])
        records.append(dict(batch=batch, first=views[0], second=views[1],
            completed_steps=completed, cursor_contiguous=contiguous, batch_metadata_aligned=aligned,
            cpp_optimizer_used=scheduler.use_cpp_optimizer))
    result = dict(schema='ditserve.tetriserve-native-scheduler-logic.v1',
        status='native_logic_observed', upstream_commit=upstream.COMMIT,
        derived_manifest_sha256=derived_manifest_sha256,
        native_scheduler_sha256=manifest['files']['tetriserve/scheduler/dyn_step_scheduler.py'],
        native_tracker_sha256=manifest['files']['tetriserve/engine/request_tracker.py'],
        cost_fixture_sha256=sha(profile.read_bytes()), costs_are_measurements=False,
        records=records, qualification_eligible=False, allows=[])
    with (output / 'summary.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'apply', 'scheduler-check'])
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--source-manifest-sha256')
    parser.add_argument('--derived-root', type=Path)
    parser.add_argument('--derived-manifest-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'scheduler-check':
        if args.derived_root is None or not args.derived_manifest_sha256 or args.output is None:
            parser.error('scheduler-check requires --derived-root, --derived-manifest-sha256 and new --output')
        result = scheduler_check(args.derived_root, args.derived_manifest_sha256, args.output)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.source_root is None or args.source_manifest is None or not args.source_manifest_sha256:
        parser.error('source commands require all three --source arguments')
    if args.command == 'apply':
        if args.output is None:
            parser.error('apply requires a new --output directory')
        result = apply(args.source_root, args.source_manifest, args.source_manifest_sha256, args.output)
    else:
        result, _, _ = plan(args.source_root, args.source_manifest, args.source_manifest_sha256)
        result = {k: v for k, v in result.items() if k != 'files'}
        result['status'] = 'source_patch_plan_only'
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
