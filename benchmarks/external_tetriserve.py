"""Probe the real, pinned TetriServe implementation without substituting a policy.

`audit` verifies the complete upstream snapshot and inspects its actual execution
interfaces. `import-probe` additionally imports its real launcher, scheduler,
runner and transfer implementation in an isolated interpreter. Neither command
starts a worker, loads weights, installs dependencies, or reports measurements.

The pinned implementation does not support the current formal models or the
shared xfuser continuation protocol. Model/backend ports must be implemented
and validated before adding a formal replay entry point.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

COMMIT = '414ec2956c09466ccd7842fda9851511462bd42a'
REPOSITORY = 'DiT-Serving/TetriServe'
ARCHIVE_SHA256 = 'a97749491c2bd6fddfe867e7a76a164bb015f3e7b439de289d99c2ca6c11b771'
MODEL_IDS = {
    'wan21': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    'wan22': 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
    'sd35': 'stabilityai/stable-diffusion-3.5-medium',
}
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / 'configs/benchmarks/formal-mixed-v1.json'
ROUND_METHODS = ('inference_prologue', 'dummy_inference_prologue', 'inference_denoising')
SD3_PIPELINE = 'model_executor/pipelines/pipeline_stable_diffusion_3.py'
GROUP_COORDINATOR = 'core/distributed/group_coordinator.py'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def verify_source(source_root, manifest_path, manifest_sha256):
    """Require an externally pinned inventory, including no extra import files."""
    root = Path(source_root).resolve(strict=True)
    raw = Path(manifest_path).read_bytes()
    if not re.fullmatch('[0-9a-f]{64}', manifest_sha256) or digest(raw) != manifest_sha256:
        raise ValueError('official source manifest SHA256 differs')
    manifest = read_json(raw)
    if (manifest.get('schema') != 'ditserve.official-source-archive.v1'
            or manifest.get('repository') != REPOSITORY
            or manifest.get('commit') != COMMIT
            or manifest.get('archive_sha256') != ARCHIVE_SHA256):
        raise ValueError('official source identity differs')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('official source inventory is empty')
    for relative, expected in files.items():
        parts = PurePosixPath(relative)
        if (parts.is_absolute() or '..' in parts.parts or str(parts) != relative
                or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('invalid official source inventory entry: ' + relative)
        path = root / relative
        if any((root.joinpath(*parts.parts[:n])).is_symlink() for n in range(1, len(parts.parts) + 1)):
            raise ValueError('symlink in official source: ' + relative)
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError('official source file SHA256 differs: ' + relative)
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in official source: ' + str(path))
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError('official source inventory differs (extra or missing files)')
    return manifest


def tree(root, relative):
    return ast.parse((Path(root) / relative).read_text(), filename=relative)


def class_node(node, name):
    return next(item for item in ast.walk(node) if isinstance(item, ast.ClassDef) and item.name == name)


def assignment(node, name):
    for item in node.body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in item.targets):
            return item.value
    raise ValueError('missing upstream assignment: ' + name)


def fields(node):
    return sorted(item.target.id for item in node.body
                  if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name))


def methods(node):
    return sorted(item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))


def shared_backend_facts(root):
    root = Path(root).resolve(strict=True)
    result = {'root': str(root), 'files': {}, 'missing_round_methods': [], 'dynamic_group_update': False}
    for relative in (SD3_PIPELINE, GROUP_COORDINATOR):
        path = root / relative
        if not path.is_file():
            raise ValueError('shared xfuser interface file missing: ' + str(path))
        result['files'][relative] = digest(path.read_bytes())
    pipeline = class_node(tree(root, SD3_PIPELINE), 'xFuserStableDiffusion3Pipeline')
    result['missing_round_methods'] = sorted(set(ROUND_METHODS) - set(methods(pipeline)))
    result['dynamic_group_update'] = any(isinstance(n, ast.FunctionDef) and n.name == 'update_sp_group'
                                         for n in ast.walk(tree(root, GROUP_COORDINATOR)))
    # Matching methods alone is not a correctness or numerical-equivalence test.
    result['drop_in_compatible'] = False
    result['compatibility_status'] = 'requires_validated_backend_port'
    return result


def audit(source_root, manifest_path, manifest_sha256, shared_xfuser_root, protocol_path=DEFAULT_PROTOCOL):
    manifest = verify_source(source_root, manifest_path, manifest_sha256)
    root = Path(source_root).resolve()
    registry_ast = assignment(class_node(tree(root, 'tetriserve/engine/runner.py'), 'FluidServeRunner'),
                              'MODEL_REGISTRY')
    registry = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(registry_ast.keys, registry_ast.values)}
    engine_models = ast.literal_eval(assignment(tree(root, 'tetriserve/__init__.py'), 'SUPPORTED_MODEL_DICT'))
    api = tree(root, 'tetriserve/server/api_server.py')
    api_fields = fields(class_node(api, 'RequestInput'))
    forwarded = sorted({kw.arg for n in ast.walk(api)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'InputConfig'
                        for kw in n.keywords if kw.arg})
    input_fields = fields(class_node(tree(root, 'tetriserve/xfuser/config/config.py'), 'InputConfig'))
    pipe = class_node(tree(root, 'tetriserve/xfuser/' + SD3_PIPELINE), 'xFuserStableDiffusion3Pipeline')
    required = {name: name in methods(pipe) for name in ROUND_METHODS}
    if not all(required.values()):
        raise ValueError('pinned upstream round interface changed')
    protocol_raw = Path(protocol_path).read_bytes()
    protocol = read_json(protocol_raw)
    if protocol.get('schema') != 'ditserve.formal-protocol.v1' or protocol.get('id') != 'formal-mixed-v1':
        raise ValueError('formal protocol identity differs')
    model_support = {}
    for model, repo in MODEL_IDS.items():
        registered = repo in registry and repo in engine_models.values()
        reasons = []
        if not registered:
            reasons.append('model absent from both the worker registry and engine model map')
        if model.startswith('wan'):
            reasons += ['HTTP API and runner do not forward num_frames',
                        'runner continuation/transfer protocol is specific to image pipelines']
        else:
            reasons += ['SD3.5 medium dual attention requires a verified transformer/backend port']
        reasons += ['shared xfuser continuation and dynamic communicator bridge is not implemented',
                    'requested guidance is not forwarded by the official API or worker']
        model_support[model] = {'repository': repo, 'registered': registered, 'runnable': False,
                                'status': 'model_backend_port_required', 'blockers': reasons}
    evidence_paths = [
        'tetriserve/server/launcher.py', 'tetriserve/server/api_server.py',
        'tetriserve/engine/runtime.py', 'tetriserve/engine/proc_launcher.py',
        'tetriserve/engine/runner.py', 'tetriserve/scheduler/dyn_step_scheduler.py',
        'tetriserve/scheduler/perf_model.py', 'tetriserve/light_ray/latent_manager.py',
        'tetriserve/distributed/globals.py', 'tetriserve/xfuser/' + SD3_PIPELINE,
    ]
    return {
        'schema': 'ditserve.external-tetriserve-probe.v1', 'status': 'blocked_model_backend_port',
        'official_source': {'repository': REPOSITORY, 'commit': COMMIT,
                            'manifest_sha256': manifest_sha256, 'archive_sha256': ARCHIVE_SHA256,
                            'verified_files': len(manifest['files']),
                            'execution_source_sha256': {p: manifest['files'][p] for p in evidence_paths}},
        'formal_protocol': {'id': protocol['id'], 'raw_sha256': digest(protocol_raw)},
        'official_entrypoint': 'python -m tetriserve.server.launcher',
        'official_schedule_logic': 'dyn_slo_schedule_global_dp',
        'worker_registry': registry, 'engine_model_map': engine_models,
        'request_api_fields': api_fields, 'api_forwarded_input_fields': forwarded,
        'input_config_fields': input_fields,
        'ignored_api_fields': sorted(set(api_fields) - set(forwarded) - {'latency_threshold'}),
        'official_round_methods': required,
        'shared_xfuser': shared_backend_facts(shared_xfuser_root),
        'current_models': model_support,
        'preserve_native_mechanisms': [
            'DynStepScheduler dynamic-programming packing, optimal shape and round DAG',
            'RequestTracker, deadlines and ZMQ worker dispatch',
            'LatentManager/LatentFuture and directed NCCL state transfer',
            'dynamic per-round SP communicator and rank assignment',
        ],
        'native_limits': {
            'model_residency': 'one globally selected model per server, replicated on every worker',
            'cost_key': ['height*width', 'batch_size', 'SP_degree'],
            'cost_ring_greater_than_one': 'excluded by official E2ECostEstimator',
            'supported_SP_choices': [1, 2, 4, 8],
            'scheduler_internal_slo_tolerance': 0.07,
            'formal_deadline_scoring': 'must use exact protocol deadline; do not inherit 7% scheduler tolerance',
            'CFG_parallelism': 'not represented in ScheduledRequests',
            'fake_engine': 'forbidden for measured comparison',
            'skip_latent_transfer': 'forbidden for measured comparison',
        },
        'qualification_eligible': False, 'allows': [], 'measurements': [],
        'serving_executed': False,
    }


# Runs actual imports. No stubs, fake torch, monkeypatches, model loads or worker launch.
_IMPORT_PROBE = r'''
import contextlib, hashlib, importlib, importlib.metadata, io, json, pathlib, sys
source, shared = (pathlib.Path(p).resolve() for p in sys.argv[1:3])
sys.path[:0] = [str(source), str(shared.parent)]
result = {'python': sys.version.split()[0], 'imports': [], 'packages': {}}
for package in ['torch','numpy','diffusers','transformers','fastapi','uvicorn','networkx','pandas','pyzmq','flash-attn','distvae']:
    try: result['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: result['packages'][package] = None
for name in ['tetriserve.server.launcher','tetriserve.scheduler.dyn_step_scheduler',
             'tetriserve.engine.runner','tetriserve.light_ray.latent_manager','xfuser']:
    entry = {'module':name, 'status':'import_failed'}
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            module = importlib.import_module(name)
        path = pathlib.Path(module.__file__).resolve()
        expected = shared if name == 'xfuser' else source / 'tetriserve'
        if not path.is_relative_to(expected): raise ValueError('module imported outside verified source: '+str(path))
        entry.update(status='imported', path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if name == 'tetriserve.engine.runner': entry['registry'] = sorted(module.FluidServeRunner.MODEL_REGISTRY)
    except Exception as exc:
        entry.update(error_type=type(exc).__name__, error=str(exc)[:2000])
    entry['diagnostic_tail'] = captured.getvalue()[-2000:]
    result['imports'].append(entry)
print('TETRISERVE_PROBE_JSON=' + json.dumps(result, sort_keys=True))
'''


def import_probe(source_root, shared_xfuser_root, *, python=sys.executable, timeout=120):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
        raise ValueError('import timeout must be in (0, 300] seconds')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1')
    command = [str(python), '-I', '-B', '-c', _IMPORT_PROBE,
               str(Path(source_root).resolve()), str(Path(shared_xfuser_root).resolve())]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {'status': 'import_probe_timeout', 'timeout_seconds': timeout, 'serving_executed': False}
    marker = 'TETRISERVE_PROBE_JSON='
    reports = [line[len(marker):] for line in proc.stdout.splitlines() if line.startswith(marker)]
    if proc.returncode != 0 or len(reports) != 1:
        return {'status': 'import_probe_process_failed', 'returncode': proc.returncode,
                'diagnostic_tail': (proc.stdout + proc.stderr)[-4000:], 'serving_executed': False}
    result = read_json(reports[0])
    result['status'] = 'imports_passed' if all(i['status'] == 'imported' for i in result['imports']) else 'imports_failed'
    result['serving_executed'] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'import-probe'])
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--shared-xfuser-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args(argv)
    try:
        result = audit(args.source_root, args.source_manifest, args.source_manifest_sha256,
                       args.shared_xfuser_root, args.protocol)
        if args.command == 'import-probe':
            result['runtime_probe'] = import_probe(args.source_root, args.shared_xfuser_root,
                                                   python=args.python, timeout=args.timeout)
            # Imports must not have modified the immutable official source tree.
            verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        # A successful audit is intentionally distinct from a runnable reproduction.
        return 2
    except (ValueError, OSError, SyntaxError, StopIteration) as exc:
        print(json.dumps({'schema': 'ditserve.external-tetriserve-probe.v1', 'status': 'invalid_evidence',
                          'error': str(exc), 'qualification_eligible': False, 'allows': [],
                          'measurements': [], 'serving_executed': False}, sort_keys=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
