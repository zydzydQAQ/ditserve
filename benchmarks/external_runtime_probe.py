"""Run bounded, source-bound external-system probes in the existing Kairos image.

This is a compatibility diagnostic. Successful collection does not qualify a
system, model, quality metric, or serving performance comparison.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

BUILD_COMMIT = '51740c05944b063e39bc325b64241cd611a73b41'
CORE_SHA256 = 'e358e29d1a3ca238bbf2b7f981db71e951519715aedf46d58454738aabcbd42b'
WORKER_SHA256 = 'fc493414a9ff589850ec25303c9166fb86e138af340f40dfd4caa62ed1b6995a'
GFC_SHA256 = 'e6c7f47a26d02a8091463bf9cb8ca58835c2b07c3c2cd4827d1b319c3bebf6e6'
IMAGE_DIGEST = 'sha256:2009ea26f63bcf0360ce7b4d71ed508c91c0702c97dd97e372cecfb672a818a1'
DATA = Path('/shared/data/DITSERVE')
OUTPUT = Path('/shared/output/DITSERVE')
PACKAGES = ('torch','diffusers','transformers','numpy','cache-dit','accelerate',
    'safetensors','vllm','vllm-omni','group-free-collectives','gfc','ray','fastapi',
    'uvicorn','distvae','flash-attn','triton','pyzmq','pandas','protobuf',
    'sentencepiece','einops','pybind11','huggingface-hub','imageio','grpcio')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def storage():
    result = subprocess.run(['df','-BG','/shared'], capture_output=True, text=True, timeout=20)
    try:
        row = result.stdout.splitlines()[-1].split()
        used, percent = int(row[2].rstrip('G')), int(row[4].rstrip('%'))
        if result.returncode or used > 790 or percent >= 80:
            raise ValueError('shared-storage stop threshold reached')
    except (ValueError, IndexError):
        raise SystemExit(75)
    return {'used_gib':used,'percent':percent}


def verify_inventory(root, files):
    root = Path(root)
    if not files or root.resolve() != root:
        raise ValueError('empty or linked probe package')
    for name, expected in files.items():
        rel = PurePosixPath(name)
        if (rel.is_absolute() or '..' in rel.parts or str(rel) != name
                or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('invalid probe inventory entry')
        path = root / rel
        if path.resolve() != path or not path.is_file() or sha(path) != expected:
            raise ValueError('probe package changed: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.name != 'package-manifest.json'}
    if actual != set(files) or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('unmanifested or linked probe package file')


def bind_sources():
    root = Path(os.environ['DITSERVE_EXTERNAL_PROBE_ROOT'])
    if root.parent != DATA / 'external-probes' or Path(__file__).resolve() != root / 'external_runtime_probe.py':
        raise ValueError('unexpected immutable probe root')
    mp = root / 'package-manifest.json'
    if sha(mp) != os.environ['DITSERVE_EXTERNAL_PROBE_MANIFEST_SHA256']:
        raise ValueError('probe manifest differs')
    manifest = json.loads(mp.read_bytes())
    if manifest.get('schema') != 'ditserve.external-probe-package.v1':
        raise ValueError('unexpected probe manifest schema')
    verify_inventory(root, manifest['files'])
    if os.environ['DITSERVE_BUILD'] != BUILD_COMMIT[:7]:
        raise ValueError('frozen build differs')
    build = Path('/shared/code/DITSERVE/builds') / BUILD_COMMIT[:7]
    info = json.loads((build/'BUILD_INFO.json').read_bytes())
    package = build / 'share/ditserve/python'
    inventory = {p.relative_to(package).as_posix():sha(p) for name in ('ditserve','xfuser')
        for p in sorted((package/name).rglob('*.py'))}
    if (info['commit'] != BUILD_COMMIT or sha(build/'bin/ditserve') != CORE_SHA256
            or canonical(inventory) != WORKER_SHA256):
        raise ValueError('frozen core/worker sources differ')
    spec_path = Path(os.environ['DITSERVE_JOB_SPEC_PATH'])
    if not spec_path.is_relative_to(DATA/'job-specs') or spec_path.resolve() != spec_path:
        raise ValueError('unexpected job spec path')
    spec = json.loads(spec_path.read_bytes())
    env = {x['name']:x['value'] for x in spec['env']}
    for key in ('DITSERVE_BUILD','DITSERVE_RUN_ID','DITSERVE_EXTERNAL_PROBE_ROOT',
                'DITSERVE_EXTERNAL_PROBE_MANIFEST_SHA256','DITSERVE_JOB_SPEC_PATH'):
        if env.get(key) != os.environ[key]:
            raise ValueError('submitted environment differs: '+key)
    if (spec['name'] != os.environ['DITSERVE_RUN_ID'] or spec['workers'] != 1
            or spec['gpus_per_worker'] != 1 or spec['mount_shared'] is not True
            or not spec['image'].endswith('@'+IMAGE_DIGEST)
            or spec['command'] != '/opt/venv/bin/python -B '+str(root/'external_runtime_probe.py')):
        raise ValueError('submitted diagnostic spec differs')
    gfc_inventory = {p.relative_to(package/'gfc').as_posix():sha(p)
        for p in sorted((package/'gfc').rglob('*.py'))}
    if canonical(gfc_inventory) != GFC_SHA256:
        raise ValueError('frozen GFC sources differ')
    plan = json.loads((root/'probe-plan.json').read_bytes())
    return root, package, plan, {'commit':manifest['commit'],
        'manifest_sha256':sha(mp),'job_spec_sha256':sha(spec_path),
        'build_info_sha256':sha(build/'BUILD_INFO.json'), 'core_sha256':CORE_SHA256,
        'worker_sources_sha256':WORKER_SHA256,'gfc_sources_sha256':GFC_SHA256,'image':spec['image'],
        'image_identity_scope':'digest-pinned submitted spec, not independent pod imageID',
        'probe_plan_sha256':sha(root/'probe-plan.json')}


def stop_process_group(proc):
    import signal
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_command(argv, timeout, env):
    """Keep checking the shared-storage limit while an actual import is running."""
    start = time.monotonic()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, start_new_session=True)
    timed_out = False
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=min(5,max(.1,timeout-(time.monotonic()-start))))
                break
            except subprocess.TimeoutExpired:
                storage()
                if time.monotonic()-start >= timeout:
                    import signal
                    stop_process_group(proc)
                    out, err = proc.communicate()
                    timed_out = True
                    break
    except BaseException:
        import signal
        stop_process_group(proc)
        proc.communicate()
        raise
    result = {'argv':argv,'returncode':proc.returncode,'timed_out':timed_out,
              'seconds':time.monotonic()-start,'stderr_tail':err[-12000:]}
    try:
        result['result'] = json.loads(out)
    except (ValueError, TypeError):
        result['stdout_tail'] = out[-12000:]
        result['json_output'] = False
    return result


def main():
    initial_storage = storage()
    root, package, plan, binding = bind_sources()
    run_id = os.environ['DITSERVE_RUN_ID']
    if not re.fullmatch('[a-z][a-z0-9-]{1,51}',run_id):
        raise ValueError('invalid run identity')
    run_dir = OUTPUT/'external-probes'/run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(run_dir/'cache/hf'), XDG_CACHE_HOME=str(run_dir/'cache/xdg'),
        TORCH_HOME=str(run_dir/'cache/torch'), TORCHINDUCTOR_CACHE_DIR=str(run_dir/'cache/inductor'),
        TRITON_CACHE_DIR=str(run_dir/'cache/triton'), MPLCONFIGDIR=str(run_dir/'cache/matplotlib'),
        PYTHONPATH=str(package), OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    os.environ.update(env)
    versions = {}
    for name in PACKAGES:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('single-GPU diagnostic allocation differs')
    tensor = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4,4)
    torch.cuda.synchronize()
    cuda = {'available':torch.cuda.is_available(), 'count':torch.cuda.device_count(),
        'name':torch.cuda.get_device_name(0), 'torch_cuda':torch.version.cuda,
        'arithmetic_pass':torch.equal((tensor@torch.eye(4,device='cuda')).cpu(), tensor.cpu())}
    if not cuda['arithmetic_pass']:
        raise ValueError('CUDA arithmetic probe failed')
    gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
                          '--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    cuda['nvidia_smi'] = gpu.stdout.strip()
    del tensor
    torch.cuda.empty_cache()
    results = []
    mapping = {'python':sys.executable,'package':str(root),'shared_xfuser':str(package/'xfuser')}
    for case in plan['cases']:
        storage()
        argv = [part.format(**mapping) for part in case['argv']]
        result = run_command(argv, int(case.get('timeout_seconds',180)), env)
        result['name'] = case['name']
        results.append(result)
        print(json.dumps({'probe':case['name'],'returncode':result['returncode'],
              'timed_out':result['timed_out']},ensure_ascii=False),flush=True)
    final_storage = storage()
    result = {'schema':'ditserve.external-runtime-probe.v1','run_id':run_id,
        'binding':binding,'packages':versions,'cuda':cuda,'cases':results,
        'storage_before':initial_storage,'storage_after':final_storage,
        'qualified_external_comparison':False,'performance_measurements':False,
        'qualification_eligible':False,'allows':[],
        'scope':'Actual fixed-image import/compatibility probes; no model output or performance qualification.'}
    target = run_dir/'summary.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    target.chmod(0o644)
    print(json.dumps({'summary':str(target),'sha256':sha(target),'qualified':False}),flush=True)

if __name__ == '__main__':
    main()
