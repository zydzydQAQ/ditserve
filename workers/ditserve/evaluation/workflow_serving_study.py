"""Frozen inputs and guarded experiment plan for the second 4090 study.

No CUDA imports: preparation, trace generation and reporting run on the host.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import time

from .workflow_budget import atomic_json
from .workflow_overlap import PROMPTS


# Runtime image is always resolved by the preparation launcher.
DEFAULT_ROOT = Path('/mnt/ssd0/zyd/ditserve-data/workflow')
CPU_SETS = [list(range(8)), list(range(60, 68))]
TRACE_SEEDS = [20261021, 20261022, 20261023]
CAPS = dict(calibration_quality=3600, placement=6000, online=5400,
            cpu=3600, overlap=2700, sp=3600, extension=3900)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def spec(index=0, seed=20261021, frames=17):
    return dict(request_id=f'request-{seed}-{index}', prompt=PROMPTS[(index // 2) % 4],
                negative_prompt='', seed=900000 + seed * 100 + index, height=480,
                width=832, frames=frames, steps=20, guidance_scale=5.)


def requests(seed, *, count=16, rate=None, slo_ms=None):
    """Freeze intended arrivals independently of completions (no coordinated omission)."""
    if count <= 0 or count % 2 or (rate is not None and rate <= 0):
        raise ValueError('need a positive even request count and positive optional rate')
    rng = random.Random(seed)
    rows = [dict(request_id=f'request-{seed}-{i}', spec=spec(i, seed, 17 if i % 2 == 0 else 33))
            for i in range(count)]
    rng.shuffle(rows)
    arrival = 0.
    for index, row in enumerate(rows):
        if index and rate is not None:
            arrival += rng.expovariate(rate) * 1000
        row.update(arrival_ms=arrival,
                   slo_ms=float((slo_ms or {}).get(str(row['spec']['frames']), 3600000.)))
    return rows


def base_manifest(kind, *, placement='dual_cpu', policy='shared_fifo'):
    return dict(schema='ditserve.workflow.serving.manifest.v1', kind=kind,
                placement=placement, policy=policy, model_path='/models/wan2.1-t2v-1.3b',
                controller_cpu=59, worker_cpu_sets=CPU_SETS,
                worker_devices=['cuda:0', 'cuda:1'], queue_limit=64,
                prefetch_depth=1, warmup_spec=spec(), observations=[],
                memory_margin_bytes=max(1024**3, int(24564 * 1024**2 * .05)),
                matmul_tf32=False, cudnn_tf32=True)


def freeze(repo, root, image):
    repo, root = Path(repo), Path(root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    source = root / 'sources/frozen'
    for name in ('workers', 'configs', 'docker', 'tests/python'):
        shutil.copytree(repo / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    files = [binding(path) for path in sorted(source.rglob('*')) if path.is_file()]
    for row in files:
        row['path'] = str(Path(row['path']).relative_to(source))
    source_digest = digest(files)
    models = Path('/mnt/ssd0/zyd/model/wan2.1-t2v-1.3b')
    model_files = []
    for path in sorted(models.rglob('*')):
        if path.is_file():
            stat = path.stat()
            row = dict(path=str(path.relative_to(models)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if stat.st_size < 4 * 1024**2:
                row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            model_files.append(row)
    study = dict(schema='ditserve.workflow.serving.study.v1', run_id=root.name, root=str(root),
                 source=str(source), source_digest=source_digest, image=image,
                 created_unix=time.time(), budget_seconds=28800, family_caps_seconds=CAPS,
                 trace_seeds=TRACE_SEEDS, source_files=files, model_files=model_files,
                 model_binding='small files SHA256; weight files size/mtime with read-only model mount',
                 status='prepared', blocks=[], jobs=[], decisions=[],
                 assumptions=dict(semantic_quality='unverified', experimental_slo='2x isolated baseline median',
                                  full_pipeline_bf16='capacity_rejected',
                                  bf16_capacity_evidence='/mnt/ssd0/zyd/ditserve-data/workflow/run01/run.json'))
    atomic_json(root / 'study.json', study)
    return study


def container_path(root, path):
    return '/data/workflow/' + root.name + '/' + str(Path(path).relative_to(root))


def freeze_analysis(repo, study):
    """Version controller/report fixes without changing any measurement job source."""
    root, repo = Path(study['root']), Path(repo)
    previous = json.loads((root / 'analysis-source.json').read_text()) if (root / 'analysis-source.json').exists() else None
    revision = (previous.get('revision', 0) if previous else 0) + 1
    source = root / 'sources' / f'analysis-{revision:02d}'
    for name in ('workers', 'configs', 'docker', 'tests/python'):
        shutil.copytree(repo / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    files = [binding(path) for path in sorted(source.rglob('*')) if path.is_file()]
    for row in files:
        row['path'] = str(Path(row['path']).relative_to(source))
    original = {row['path']: row['sha256'] for row in study['source_files']}
    value = dict(source=str(source), revision=revision, source_digest=digest(files), files=files,
                 measurement_source=study['source'], measurement_source_digest=study['source_digest'],
                 purpose='Controller/report revision; all experimental jobs retain original frozen measurement source.',
                 created_unix=time.time(), changes=[dict(path=row['path'], previous_sha256=original.get(row['path']),
                 sha256=row['sha256']) for row in files if original.get(row['path']) != row['sha256']])
    atomic_json(root / 'sources' / f'analysis-{revision:02d}.binding.json', value)
    atomic_json(root / 'analysis-source.json', value)
    return value


def job(study, identifier, module, config, *, family, kind=None, devices=(0, 1),
        timeout=1800, minimum=120, block=None, profiling=False):
    root = Path(study['root'])
    config_path = root / 'inputs' / (identifier + '.json')
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('refusing to replace frozen job input: ' + identifier)
    atomic_json(config_path, config)
    output = root / 'measurements' / identifier
    command = ['/opt/venv/bin/python', '-m', 'ditserve.evaluation.' + module]
    if kind:
        command += ['--kind', kind]
    command += ['--manifest' if module == 'workflow_serving' else '--config',
                container_path(root, config_path), '--out', container_path(root, output)]
    if profiling:
        (root / 'traces').mkdir(exist_ok=True)
        capture = 'ditserve.workflow/P4/stage_pair|' + json.dumps(dict(
            frames=17, mode='serial', peer_frames=17, purpose='timed', repeat=0,
            request_id='P4-timed-a17-b17-r0-serial', variant='P4'), sort_keys=True, separators=(',', ':'))
        command = ['/data/tools/nsight-workflow/extracted/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys',
                   'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                   '--capture-range=nvtx', '--capture-range-end=none', '--kill=none',
                   '--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0', '--nvtx-capture=' + capture,
                   '--gpu-metrics-devices=' + ','.join(map(str, devices)),
                   '--gpu-metrics-frequency=1000',
                   '--output', container_path(root, root / 'traces' / identifier)] + command
    value = dict(id=identifier, command=command, devices=list(devices),
                 timeout_seconds=timeout, minimum_budget_seconds=minimum,
                 family=family, block=block or identifier, configuration_sha256=digest(config),
                 output=str(output), input=str(config_path), cpu_only=not devices,
                 logical_id=identifier, attempt=1)
    if profiling:
        value.update(profiling=True, counters=True)
    existing = next((row for row in study['jobs'] if row['id'] == identifier), None)
    if existing and existing.get('superseded'):
        current = next(row for row in reversed(study['jobs'])
                       if row.get('logical_id', row['id']) == identifier and not row.get('superseded'))
        if current['configuration_sha256'] != value['configuration_sha256']:
            raise ValueError('retry configuration cannot be silently replaced')
        return current
    if existing and existing != value:
        raise ValueError('job specification changed: ' + identifier)
    if existing is None:
        study['jobs'].append(value)
    atomic_json(root / 'study.json', study)
    return value


def trace_jobs(study):
    """A small independent capture; its timings never enter performance estimates."""
    config = dict(model_path='/models/wan2.1-t2v-1.3b', device='cuda:1',
                  cpu_threads=8, cpu_masks={'cuda:1': CPU_SETS[1]},
                  height=480, width=832, steps=20, frames=[17],
                  frame_pairs=[[17, 17]], repeats=1)
    return [job(study, 'trace-overlap', 'workflow_serving_overlap', config,
                family='extension', devices=(1,), timeout=1200, profiling=True)]


def initial_jobs(study):
    for placement in ('dual_cpu', 'dedicated_fp32'):
        smoke = base_manifest('calibration', placement=placement)
        smoke['observations'] = [dict(id='smoke', role='measurement',
                                      requests=requests(20261018, count=2), estimates_ms={})]
        job(study, 'smoke-' + placement, 'workflow_serving', smoke,
            family='calibration_quality', timeout=900)
    manifest = base_manifest('calibration')
    for seed in TRACE_SEEDS:
        manifest['observations'].append(dict(id=f'saturation-{seed}', role='saturation',
            requests=requests(seed), estimates_ms={}))
    for frames in (17, 33):
        manifest['observations'].append(dict(id=f'isolated-{frames}', role='isolated',
            requests=[dict(request_id=f'isolated-{frames}-{i}', arrival_ms=0., slo_ms=3600000.,
                           spec=dict(spec(i, 20261019, frames), request_id=f'isolated-{frames}-{i}'))
                      for i in range(3)], estimates_ms={}))
    job(study, 'calibration', 'workflow_serving', manifest, family='calibration_quality', timeout=2400)
    config = dict(model_path='/models/wan2.1-t2v-1.3b', device='cuda:1',
                  cpu_mask=CPU_SETS[1], cpu_threads=8, cases=['chinese', 'truncation'],
                  seeds=[20261019, 20261020], frames=[17, 33], steps=20,
                  repeats=3, warmups=1)
    job(study, 'quality', 'workflow_serving_encoder', config, kind='quality',
        family='calibration_quality', devices=(1,), timeout=2400)
    for cores in (2, 4, 8, 16):
        config = dict(model_path='/models/wan2.1-t2v-1.3b', device='cuda:1',
                      cpu_mask=list(range(60, 60 + cores)), cpu_threads=cores,
                      prompt_cases=['english', 'truncation'], repeats=3, warmups=1)
        job(study, f'cpu-micro-{cores}', 'workflow_serving_encoder', config, kind='cpu_micro',
            family='cpu', devices=(), timeout=1200)
    common = dict(model_path='/models/wan2.1-t2v-1.3b', device='cuda:1',
                  devices=['cuda:0', 'cuda:1'], cpu_threads=8,
                  cpu_masks={'cuda:0': CPU_SETS[0], 'cuda:1': CPU_SETS[1]},
                  height=480, width=832, steps=20, frames=[17, 33], repeats=3)
    job(study, 'overlap', 'workflow_serving_overlap', common, family='overlap', devices=(1,), timeout=2700)
    job(study, 'sp-diagnostic', 'workflow_serving_sp', common, family='sp', timeout=3600)


def calibration_from_result(result):
    observations = result.get('observations', [])
    throughput, isolated, costs = [], {'17': [], '33': []}, {'17': [], '33': []}
    for obs in observations:
        role = obs.get('role') or ('saturation' if obs['id'].startswith('saturation') else 'isolated')
        completed = [row for row in obs.get('requests', []) if row.get('status') == 'completed']
        summary = obs.get('summary', {})
        if role == 'saturation':
            wall = summary.get('wall_ms')
            if wall and len(completed) == 16:
                throughput.append(len(completed) * 1000 / wall)
        else:
            for row in completed:
                frames = str(row.get('spec', {}).get('frames', row.get('frames')))
                value = row.get('service_ms', row.get('latency_ms', row.get('end_to_end_ms')))
                if value is not None:
                    isolated[frames].append(value)
                encode = row.get('encode_ms')
                gpu = row.get('gpu_ms')
                if encode is not None and gpu is not None:
                    costs[frames].append((encode, gpu))
    if len(throughput) != 3 or any(len(values) < 3 for values in isolated.values()):
        raise ValueError('calibration needs 3 complete saturation queues and 3 isolated samples per frame count')
    if any(not values for values in costs.values()):
        raise ValueError('calibration missing encoder/GPU phase durations')
    return dict(lambda0=statistics.median(throughput),
                slo_ms={key: 2 * statistics.median(values) for key, values in isolated.items()},
                estimates_ms={key: dict(encode=statistics.median(x[0] for x in values),
                                         gpu=statistics.median(x[1] for x in values))
                              for key, values in costs.items()},
                reference_throughputs=throughput, source_digest=digest(result))


def online_jobs(study, calibration):
    root = Path(study['root'])
    path = root / 'calibration.json'
    if path.exists() and json.loads(path.read_text()) != calibration:
        raise ValueError('calibration is frozen once serving traces are generated')
    atomic_json(path, calibration)
    for load in (.5, .8, 1.1):
        for repeat, seed in enumerate(TRACE_SEEDS):
            rows = requests(seed, rate=calibration['lambda0'] * load, slo_ms=calibration['slo_ms'])
            observation = dict(id=f'load{load}-seed{seed}', role='measurement', requests=rows,
                estimates_ms=calibration['estimates_ms'],
                arrival_window_ms=rows[-1]['arrival_ms'] + 1000 / (calibration['lambda0'] * load),
                observation_cutoff_ms=rows[-1]['arrival_ms'] + 2 * max(calibration['slo_ms'].values()),
                trace_sha256=digest(rows))
            candidates = [('dual_cpu', 'shared_fifo')]
            if load != 1.1:
                candidates.append(('dedicated_fp32', 'shared_fifo'))
            candidates.append(('dual_cpu', 'least_work_fifo'))
            if repeat % 2:
                candidates.reverse()
            for placement, policy in candidates:
                kind = 'online' if policy == 'least_work_fifo' or load == 1.1 else 'placement'
                family = 'online' if kind == 'online' else 'placement'
                manifest = base_manifest(kind, placement=placement, policy=policy)
                manifest.update(observations=[observation], nominal_rate=calibration['lambda0'] * load,
                                load_fraction=load, repeat=repeat, frozen_calibration=calibration)
                identifier = f'{placement}-{policy}-l{str(load).replace(".", "p")}-r{repeat}'
                job(study, identifier, 'workflow_serving', manifest, family=family,
                    timeout=1800, block=f'load-{load}-r{repeat}')


def prefetch_jobs(study, alternative):
    for cores in sorted({8, int(alternative)}):
        for depth in (0, 1, 2):
            config = dict(model_path='/models/wan2.1-t2v-1.3b', device='cuda:1',
                          cpu_mask=list(range(60, 60 + cores)), cpu_threads=cores,
                          frames=[17, 33], steps=20, repeats=3, requests=4,
                          prefetch_depth=depth, depth=depth, warmups=1, seed=20261024)
            job(study, f'prefetch-c{cores}-d{depth}', 'workflow_serving_encoder', config,
                family='cpu', kind='prefetch', devices=(1,), timeout=1800,
                block=f'prefetch-c{cores}')


def extension_jobs(study, calibration, selection):
    for repeat in range(6):
        seed = 20261101 + repeat
        load = selection['load']
        rows = requests(seed, count=32, rate=calibration['lambda0'] * load, slo_ms=calibration['slo_ms'])
        obs = dict(id=f'confirm-{seed}', role='measurement', requests=rows,
                   estimates_ms=calibration['estimates_ms'],
                   arrival_window_ms=rows[-1]['arrival_ms'] + 1000 / (calibration['lambda0'] * load),
                   observation_cutoff_ms=rows[-1]['arrival_ms'] + 2 * max(calibration['slo_ms'].values()),
                   trace_sha256=digest(rows))
        arms = [('dual_cpu', 'shared_fifo')]
        arms.append(('dedicated_fp32', 'shared_fifo') if selection['family'] == 'placement'
                    else ('dual_cpu', 'least_work_fifo'))
        if repeat % 2:
            arms.reverse()
        for placement, policy in arms:
            config = base_manifest(selection['family'], placement=placement, policy=policy)
            config.update(observations=[obs], nominal_rate=calibration['lambda0'] * load,
                          load_fraction=load, repeat=repeat, confirmation=True,
                          selection=selection, frozen_calibration=calibration)
            job(study, f'confirm-r{repeat}-{placement}-{policy}', 'workflow_serving', config,
                family='extension', timeout=2400, minimum=600, block=f'confirm-r{repeat}')


def retry_block(study, block):
    """Retry every arm together using fresh paths; historical evidence remains intact."""
    root = Path(study['root'])
    budget_path = root / 'budget.json'
    if study.get('active_job') or (budget_path.exists() and json.loads(budget_path.read_text()).get('active')):
        raise RuntimeError('settle the active attempt before creating a retry block')
    previous = [row for row in study['jobs'] if row['block'] == block and not row.get('superseded')]
    if not previous:
        raise ValueError('no current block named ' + block)
    attempt = max(row.get('attempt', 1) for row in previous) + 1
    for original in previous:
        logical = original.get('logical_id', original['id'])
        identifier = logical + '-attempt' + str(attempt)
        config = json.loads(Path(original['input']).read_text())
        config_path = root / 'inputs' / (identifier + '.json')
        if config_path.exists():
            raise FileExistsError(config_path)
        atomic_json(config_path, config)
        revised = dict(original, id=identifier, logical_id=logical, attempt=attempt,
                       input=str(config_path), output=str(root / 'measurements' / identifier),
                       block=block + '-attempt' + str(attempt))
        command = list(original['command'])
        for option, path in (('--manifest' if '--manifest' in command else '--config', config_path),
                             ('--out', Path(revised['output']))):
            command[command.index(option) + 1] = container_path(root, path)
        if revised.get('profiling') and '--output' in command:
            command[command.index('--output') + 1] = container_path(root, root / 'traces' / identifier)
        revised['command'] = command
        original['superseded'] = identifier
        study['jobs'].append(revised)
    study['decisions'].append(dict(block=block, decision='retry_whole_block', attempt=attempt))
    atomic_json(root / 'study.json', study)
