"""Inspect and import the pinned GF-DiT runtime, without replacing its execution.

This is a source/ABI probe, not a serving benchmark or an execution adapter.
It never installs packages, loads model weights or starts distributed workers.
Exit 2 means inspected but not qualified; exit 1 means invalid probe evidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

REPOSITORY = 'SJTU-Liquid/GF-DiT'
COMMIT = '442531e499891d4ffa654e87c076f03326a37b98'
ARCHIVE_SHA256 = '8efb63f6e59b6fa67f631eac7c559d44157397b17bf212a20e00d83bb6c24ec0'
GFC_REFERENCE_COMMIT = '9aadcdc6d4abc30af13feac908fbb4be98313398'
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / 'configs/benchmarks/formal-mixed-v1.json'
MODEL_CLASSES = {'wan21': 'WanPipeline', 'wan22': 'WanPipeline', 'sd35': 'StableDiffusion3Pipeline'}
EVIDENCE_FILES = (
    'vllm_omni/diffusion/executor/multiproc_executor.py',
    'vllm_omni/diffusion/runtime_v2/registry.py',
    'vllm_omni/diffusion/runtime_v2/runner.py',
    'vllm_omni/diffusion/runtime_v2/adapters/wan.py',
    'vllm_omni/diffusion/runtime_v2/adapters/qwen_image.py',
    'vllm_omni/diffusion/runtime_v2/policies/edf_best_fit.py',
    'vllm_omni/diffusion/runtime_v2/policies/edf_greedy.py',
    'vllm_omni/diffusion/runtime_v2/multiproc_worker.py',
    'vllm_omni/diffusion/runtime_v2/data_plane.py',
    'vllm_omni/diffusion/runtime_v2/migration_engine.py',
    'vllm_omni/diffusion/runtime_v2/cost_model.py',
    'vllm_omni/diffusion/distributed/parallel_state.py',
    'vllm_omni/diffusion/distributed/collective_runtime.py',
    'vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py',
    'requirements/common.txt', 'requirements/cuda.txt', 'docker/Dockerfile.ci',
)
IMPORT_TARGETS = (
    'torch', 'triton', 'gfc', 'xfuser', 'vllm',
    'vllm_omni.diffusion.runtime_v2.registry',
    'vllm_omni.diffusion.runtime_v2.policies.edf_best_fit',
    'vllm_omni.diffusion.runtime_v2.runner',
    'vllm_omni.diffusion.runtime_v2.multiproc_worker',
    'vllm_omni.diffusion.distributed.collective_runtime',
)


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
    root = Path(source_root).resolve(strict=True)
    raw = Path(manifest_path).read_bytes()
    if not isinstance(manifest_sha256, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha256) or digest(raw) != manifest_sha256:
        raise ValueError('official source manifest SHA256 differs')
    manifest = read_json(raw)
    if (manifest.get('schema') != 'ditserve.official-source-archive.v1'
            or manifest.get('repository') != REPOSITORY or manifest.get('commit') != COMMIT
            or manifest.get('archive_sha256') != ARCHIVE_SHA256):
        raise ValueError('official source identity differs')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('empty official source inventory')
    for name, expected in files.items():
        parts = PurePosixPath(name)
        if (not name or parts.is_absolute() or '..' in parts.parts or '\\' in name
                or str(parts) != name or not isinstance(expected, str)
                or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('invalid source inventory entry: ' + name)
        if any(root.joinpath(*parts.parts[:n]).is_symlink() for n in range(1, len(parts.parts) + 1)):
            raise ValueError('symlink in source: ' + name)
        if digest((root / name).read_bytes()) != expected:
            raise ValueError('official source SHA256 differs: ' + name)
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in source: ' + str(path))
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError('official source inventory differs')
    return manifest


def class_constant(path, class_name, field):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    for item in cls.body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == field for t in item.targets):
            return ast.literal_eval(item.value)
    raise ValueError('missing class constant: ' + class_name + '.' + field)


def shared_inventory(shared_python_root):
    root = Path(shared_python_root).resolve(strict=True)
    inventory = {}
    for package in ('xfuser', 'gfc'):
        package_root = root / package
        if not (package_root / '__init__.py').is_file():
            raise ValueError('shared backend package missing: ' + str(package_root))
        files = {}
        for path in sorted(package_root.rglob('*')):
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError('shared backend path escapes source: ' + str(path))
            if path.is_file():
                files[path.relative_to(root).as_posix()] = digest(path.read_bytes())
        inventory[package] = files
    return inventory


def model_configs(model_paths):
    """Read local configurations only; weights and Hugging Face are never opened."""
    result = {}
    for model, path in (model_paths or {}).items():
        if model not in MODEL_CLASSES:
            raise ValueError('unknown model: ' + model)
        root = Path(path).resolve(strict=True)
        files = {}
        values = {}
        for name in ('model_index.json', 'transformer/config.json', 'vae/config.json', 'scheduler/scheduler_config.json'):
            raw = (root / name).read_bytes()
            values[name] = read_json(raw)
            files[name] = digest(raw)
        index, transformer, vae = (values[n] for n in ('model_index.json', 'transformer/config.json', 'vae/config.json'))
        if index.get('_class_name') != MODEL_CLASSES[model]:
            raise ValueError('local model class differs: ' + model)
        entry = {'root': str(root), 'files_sha256': files, 'class_name': index['_class_name'],
                 'expand_timesteps': index.get('expand_timesteps'),
                 'transformer_2': index.get('transformer_2'), 'weights_loaded': False}
        if model.startswith('wan'):
            patch = transformer.get('patch_size')
            spatial, temporal = vae.get('scale_factor_spatial'), vae.get('scale_factor_temporal')
            if not isinstance(patch, list) or len(patch) != 3 or any(type(x) is not int or x < 1 for x in patch):
                raise ValueError('invalid Wan patch geometry: ' + model)
            if spatial is None or temporal is None:
                # Older Wan configs rely on diffusers constructor defaults.
                # The official adapter indexes these config fields directly,
                # then falls back to Wan2.2 geometry on failure. Do not adopt
                # that fallback or discard the other real import observations.
                entry['geometry_status'] = 'unresolved_config_defaults_require_validation'
                entry['geometry'] = None
                entry['vae_compression_hints'] = {k: vae.get(k) for k in
                    ('dim_mult', 'temperal_downsample', 'scale_factor_spatial', 'scale_factor_temporal')}
                entry['formal_execution_blocked'] = True
            else:
                if type(spatial) is not int or spatial < 1 or type(temporal) is not int or temporal < 1:
                    raise ValueError('invalid Wan VAE geometry: ' + model)
                entry['geometry_status'] = 'explicit_config_values_only_not_model_validation'
                entry['geometry'] = {'patch_size': patch, 'spatial_scale': spatial, 'temporal_scale': temporal,
                                     'attention_heads': transformer.get('num_attention_heads')}
            entry['TI2V_expansion_path_required'] = index.get('expand_timesteps') is True
        result[model] = entry
    return result


def audit(source_root, manifest_path, manifest_sha256, shared_python_root,
          protocol_path=DEFAULT_PROTOCOL, model_paths=None):
    manifest = verify_source(source_root, manifest_path, manifest_sha256)
    root = Path(source_root).resolve()
    registered = sorted(class_constant(root / ('vllm_omni/diffusion/runtime_v2/adapters/' + file), cls, 'model_class_name')
                        for file, cls in [('wan.py', 'WanRuntimeV2Adapter'), ('qwen_image.py', 'QwenImageRuntimeV2Adapter')])
    protocol_raw = Path(protocol_path).read_bytes()
    protocol = read_json(protocol_raw)
    if protocol.get('schema') != 'ditserve.formal-protocol.v1' or protocol.get('id') != 'formal-mixed-v1':
        raise ValueError('formal protocol identity differs')
    native = {}
    for model, cls in MODEL_CLASSES.items():
        native[model] = {'model_class_name': cls, 'runtime_v2_registered': cls in registered,
                         'model_execution_verified': False,
                         'status': 'native_model_validation_required' if cls in registered else 'runtime_v2_adapter_missing'}
    return {
        'schema': 'ditserve.external-gfdit-probe.v1', 'status': 'blocked_shared_backend_port',
        'official_source': {'repository': REPOSITORY, 'commit': COMMIT,
                            'manifest_sha256': manifest_sha256, 'archive_sha256': ARCHIVE_SHA256,
                            'verified_files': len(manifest['files']),
                            'execution_source_sha256': {p: manifest['files'][p] for p in EVIDENCE_FILES}},
        'shared_python_root': str(Path(shared_python_root).resolve()),
        'shared_python_sources': shared_inventory(shared_python_root),
        'formal_protocol': {'id': protocol['id'], 'raw_sha256': digest(protocol_raw)},
        'official_entrypoint': 'vllm serve MODEL --omni --enable-runtime-v2 --runtime-v2-scheduler-policy edf_best_fit --runtime-v2-collective-backend gfc',
        'registered_runtime_v2_models': registered, 'current_models': native,
        'local_model_configurations': model_configs(model_paths),
        'native_mechanisms_to_preserve': [
            'EdfBestFitSchedulerPolicy deadline best-fit and surplus redistribution',
            'GlobalScheduler task DAG, worker dispatch, RuntimeTopology online group formation',
            'artifact layout codecs, data plane and implicit state migration',
            'worker group session activation and GFC all_gather/all2all/P2P stream ordering',
        ],
        'shared_backend_port': {
            'implemented': False,
            'required': ['Wan stage executors and state codecs bridged to shared xfuser',
                         'SD3.5 RuntimeV2Adapter/compiler/executors/codecs',
                         'GFC logical group and dynamic session bridge for xfuser attention',
                         'per-model stage cost calibration and numerical equivalence tests',
                         'multi-model residency/dispatch and common arrival/deadline replay'],
            'cannot_substitute': ['local edf-bestfit policy', 'fixed SP executor', 'torch/NCCL-only collective backend'],
        },
        'native_limits': {'dynamic_tp': 1, 'dynamic_cfg': 1, 'max_group_ranks': 16,
                          'cost_model_missing_entry': 'returns zero; must reject missing coverage in formal adapter',
                          'geometry_load_failure': 'falls back to Wan2.2 geometry; formal adapter must require actual config',
                          'model_residency': 'single model selected per server; current mixed-model benchmark requires a port',
                          'batching': 'one output per request does not prohibit native batching; tune supported mechanisms on calibration only'},
        'dependencies': {'vllm_reference': 'docker/Dockerfile.ci:v0.17.0 (not a pinned Python requirement)',
                         'requirements_files': {p: (root / p).read_text() for p in ('requirements/common.txt', 'requirements/cuda.txt')},
                         'gfc_repository': 'SJTU-Liquid/group-free-collectives',
                         'gfc_reference_commit': GFC_REFERENCE_COMMIT,
                         'gfc_reference_status': 'vendored declaration; independently verify official archive before reproduction',
                         'gfc_requirements': ['torch>=2.10', 'triton>=3.6'],
                         'native_gfc_requires': ['CUDA', 'torch.distributed._symmetric_memory', 'GPU peer memory access']},
        'qualification_eligible': False, 'allows': [], 'measurements': [], 'serving_executed': False,
    }


# Each import runs in a fresh real interpreter. No modules or APIs are stubbed.
_IMPORT_ONE = r'''
import contextlib, hashlib, importlib, importlib.metadata, inspect, io, json, pathlib, sys
source, shared = (pathlib.Path(p).resolve() for p in sys.argv[1:3])
name = sys.argv[3]
sys.path[:0] = [str(source), str(shared)]
r = {'module': name, 'status': 'import_failed', 'python': sys.version.split()[0]}
capture = io.StringIO()
try:
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        module = importlib.import_module(name)
        path = pathlib.Path(module.__file__).resolve()
        expected = source if name.startswith('vllm_omni.') else shared / name if name in ('gfc','xfuser') else None
        if expected is not None and not path.is_relative_to(expected):
            raise ValueError('module imported outside selected source: ' + str(path))
        r.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if name == 'gfc':
            r['config_signature'] = str(inspect.signature(module.SymmetricCollectiveConfig))
            api = ['register_group','all_gather','all2all','p2p_put','p2p_get','record_event','wait_for_external']
            r['runtime_api'] = {n: hasattr(module.SymmetricCollectiveRuntime,n) for n in api}
            if not all(r['runtime_api'].values()): raise ValueError('GFC runtime API incomplete')
        if name == 'torch':
            r.update(version=module.__version__, cuda_version=module.version.cuda,
                     cuda_available=module.cuda.is_available(), device_count=module.cuda.device_count())
            sm = importlib.import_module('torch.distributed._symmetric_memory')
            r['symmetric_memory_api'] = {n: hasattr(sm,n) for n in ['empty','rendezvous']}
            if not all(r['symmetric_memory_api'].values()): raise ValueError('symmetric memory API incomplete')
        if name == 'vllm_omni.diffusion.runtime_v2.registry':
            r['registered_models'] = sorted(module.get_registered_runtime_v2_adapters())
            r['model_support'] = {n: module.supports_runtime_v2_model(n) for n in ['WanPipeline','StableDiffusion3Pipeline']}
        if name == 'vllm_omni.diffusion.runtime_v2.policies.edf_best_fit':
            r['native_policy'] = module.EdfBestFitSchedulerPolicy.__qualname__
        if name == 'vllm_omni.diffusion.runtime_v2.runner':
            r['native_runner_signature'] = str(inspect.signature(module.RuntimeV2Runner))
        for package in ['torch','triton','vllm','diffusers','transformers','accelerate','cache-dit','einops']:
            try: r.setdefault('packages',{})[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: r.setdefault('packages',{})[package] = None
        r['status'] = 'imported'
except Exception as exc:
    r.update(error_type=type(exc).__name__, error=str(exc)[:2000])
r['diagnostic_tail'] = capture.getvalue()[-2000:]
print('GFDIT_IMPORT_JSON=' + json.dumps(r,sort_keys=True))
'''


def import_probe(source_root, shared_python_root, *, python=sys.executable, timeout=120):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError('import timeout must be in (0, 300] seconds')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1')
    deadline = time.monotonic() + timeout
    results = []
    for name in IMPORT_TARGETS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.append({'module': name, 'status': 'not_attempted_timeout'})
            continue
        command = [str(python), '-I', '-B', '-c', _IMPORT_ONE, str(Path(source_root).resolve()),
                   str(Path(shared_python_root).resolve()), name]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, env=env,
                                  timeout=min(30, remaining), check=False)
        except subprocess.TimeoutExpired:
            results.append({'module': name, 'status': 'import_timeout'})
            continue
        marker = 'GFDIT_IMPORT_JSON='
        reports = [s[len(marker):] for s in proc.stdout.splitlines() if s.startswith(marker)]
        if proc.returncode != 0 or len(reports) != 1:
            results.append({'module': name, 'status': 'import_process_failed', 'returncode': proc.returncode,
                            'diagnostic_tail': (proc.stdout + proc.stderr)[-4000:]})
            continue
        record = read_json(reports[0])
        if record.get('module') != name or record.get('status') not in ('imported', 'import_failed'):
            raise ValueError('import result identity differs')
        results.append(record)
    return {'status': 'imports_passed' if all(r['status'] == 'imported' for r in results) else 'imports_failed',
            'imports': results, 'serving_executed': False, 'collectives_executed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'import-probe'])
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--shared-python-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    for model in MODEL_CLASSES:
        parser.add_argument('--' + model + '-model', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args(argv)
    try:
        model_paths = {m: getattr(args, m + '_model') for m in MODEL_CLASSES if getattr(args, m + '_model') is not None}
        result = audit(args.source_root, args.source_manifest, args.source_manifest_sha256,
                       args.shared_python_root, args.protocol, model_paths)
        if args.command == 'import-probe':
            result['runtime_probe'] = import_probe(args.source_root, args.shared_python_root,
                                                   python=args.python, timeout=args.timeout)
            verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
            if shared_inventory(args.shared_python_root) != result['shared_python_sources']:
                raise ValueError('shared source changed during import probe')
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 2
    except (ValueError, OSError, SyntaxError, StopIteration, KeyError) as exc:
        print(json.dumps({'schema': 'ditserve.external-gfdit-probe.v1', 'status': 'invalid_evidence',
                          'error': str(exc), 'qualification_eligible': False, 'allows': [],
                          'measurements': [], 'serving_executed': False}, sort_keys=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
