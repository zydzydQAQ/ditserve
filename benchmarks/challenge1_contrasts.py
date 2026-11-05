#!/usr/bin/env python3
"""Challenge 1 form-planning & elastic CFG2xSP contrasts (simulation + GPU config generator).

Evaluates why DitServe prunes Pure Ulysses SP (c=1, u>1) and elastically scales
within the non-dominated CFG2 x Ulysses-SP family (CFG2 <-> CFG2SP2 <-> CFG2SP3 <-> CFG2SP4)
using calibrated 8-GPU PCIe 4.0 profiles and the native C++ controller (`ditserve rpc simulate`),
and generates frozen GPU serving study configs without modifying `p2_contrasts.py`.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / 'benchmarks', ROOT / 'workers'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import formal_benchmark as fb
from native_serving import FORMS, FORM_DEGREES, IN_FRAMEWORK_MECHANISM_DISPLAY, MODELS
from pcie_collective_characterization import load_bundle_forms
from stage_gates import canonical_hash


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


DEFAULT_BINARY = ROOT / '.cache' / 'build' / 'ditserve'

IN_FRAMEWORK_MAIN_ARMS = (
    'static-cfg2sp4',
    'tetriserve',
    'sbp-esp',
    'edf-bestfit',
    'ditserve',
)

ARM_SPECS: Dict[str, Dict[str, Any]] = {
    'pure-sp-elastic': {
        'policy': 'edf-bestfit',
        'description': 'Pure Ulysses SP elastic scaling (c=1 only: SP1..SP6/SP8) without CFG2 subgrouping',
        'forms_mode': 'pure_sp',
        'options_override': {
            'prune_dominated': False,
            'min_video_degree': 1,
        },
    },
    'tetriserve': {
        'policy': 'tetriserve',
        'description': 'TetriServe round-based deadline-DP elastic scheduler over unpruned forms',
        'forms_mode': 'all',
        'options_override': {
            'prune_dominated': False,
            'min_video_degree': 1,
        },
    },
    'sbp-esp': {
        'policy': 'sbp-esp',
        'description': 'SBP-ESP step-boundary elastic SP with SLO-aware deadline knapsack and dynamic image batching',
        'forms_mode': 'all',
        'options_override': {
            'prune_dominated': False,
            'min_video_degree': 1,
        },
    },
    'edf-bestfit': {
        'policy': 'edf-bestfit',
        'description': 'EDF-BF smallest deadline-feasible SP or largest fitting group with EDF surplus redistribution',
        'forms_mode': 'all',
        'options_override': {
            'prune_dominated': False,
            'min_video_degree': 1,
        },
    },
    'abl-no-prune': {
        'policy': 'abl-no-prune',
        'description': 'Unpruned CFG x SP elastic scheduler with block preemption (allows dominated pure-SP forms)',
        'forms_mode': 'all',
        'options_override': {
            'prune_dominated': False,
            'min_video_degree': 1,
        },
    },
    'edf-cfg2': {
        'policy': 'edf-cfg2',
        'description': 'Static 2-GPU CFG2 (c=2, u=1): communication-minimal pair without elastic SP widening',
        'forms_mode': 'cfg2_only',
        'options_override': {},
    },
    'static-cfg2sp2': {
        'policy': 'static-cfg2sp2',
        'description': 'Static 4-GPU CFG2xSP2 (c=2, u=2): fixed medium degree without elasticity',
        'forms_mode': 'all',
        'options_override': {},
    },
    'static-cfg2sp3': {
        'policy': 'static-cfg2sp3',
        'description': 'Static 6-GPU CFG2xSP3 (c=2, u=3): fixed 6-GPU degree without elasticity',
        'forms_mode': 'all',
        'options_override': {},
    },
    'static-cfg2sp4': {
        'policy': 'static-cfg2sp4',
        'description': 'Static 8-GPU CFG2xSP4 (c=2, u=4): fixed widest degree without elasticity',
        'forms_mode': 'all',
        'options_override': {},
    },
    'cfg2-first': {
        'policy': 'cfg2-first',
        'description': 'Pruned elastic CFG2xSP (CFG2 <-> CFG2SP2 <-> CFG2SP3 <-> CFG2SP4) at step boundaries',
        'forms_mode': 'all',
        'options_override': {
            'prune_dominated': True,
            'min_video_degree': 2,
        },
    },
    'ditserve': {
        'policy': 'ditserve',
        'description': 'DitServe: pruned elastic CFG2xSP + sub-step block preemption + pair-fit image placement',
        'forms_mode': 'all',
        'options_override': {
            'prune_dominated': True,
            'min_video_degree': 2,
        },
    },
}
CHALLENGE1_ARMS = ARM_SPECS

DEFAULT_VIDEO_SLO_SCALES = (0.45, 0.55, 0.65, 0.80, 1.00, 1.50)
DEFAULT_LOADS = (0.6, 0.8, 1.0, 1.2)
DEFAULT_SEEDS = (301, 302, 303)


def model_forms(model: str, gpus: int, mode: str = 'all') -> List[str]:
    """Return legal parallel forms for `model` on `gpus` under `mode`."""
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    base = [form for form in FORMS if FORM_DEGREES[form] <= gpus and not (model == 'wan21' and form == 'sp8')]
    if mode == 'pure_sp':
        return [form for form in base if not form.startswith('cfg2')]
    if mode == 'cfg2_only':
        return ['cfg2']
    if mode == 'pruned_cfg2xsp':
        return [form for form in base if form.startswith('cfg2')]
    if mode == 'all':
        return base
    raise ValueError(f'Unknown forms_mode: {mode}')


def resolve_arm_config(
    arm: str,
    model: str,
    gpus: int,
    base_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve policy name and options for a Challenge-1 contrast arm."""
    if arm not in CHALLENGE1_ARMS:
        raise ValueError(f'Unknown Challenge-1 arm: {arm}')
    spec = CHALLENGE1_ARMS[arm]
    options = copy.deepcopy(base_options or {})
    options['sp_transport'] = options.get('sp_transport', 'export')
    options['forms'] = model_forms(model, gpus, spec['forms_mode'])
    options.update(copy.deepcopy(spec['options_override']))
    return {
        'arm': arm,
        'policy': spec['policy'],
        'description': spec['description'],
        'forms': options['forms'],
        'options': options,
    }


def build_base_cell(
    protocol: Dict[str, Any],
    model: str,
    bundle: Dict[str, Any],
    pools: Dict[str, Any],
    *,
    seed: int = 301,
    load: float = 0.8,
    video_fraction: float = 0.2,
    pattern: str = 'poisson',
    video_slo_scale: float = 1.5,
    image_slo_scale: Optional[float] = None,
    split: str = 'test',
) -> Dict[str, Any]:
    """Build a calibrated mixed-resolution workload and profile cell from a verified bundle."""
    p = copy.deepcopy(protocol)
    if load not in p['loads']:
        p['loads'] = sorted(set(p['loads']) | {load})
    if seed not in p['seeds'][split]:
        p['seeds'][split] = sorted(set(p['seeds'][split]) | {seed})
    p['slo'] = copy.deepcopy(p['slo'])
    p['slo']['video_scale'] = float(video_slo_scale)
    if image_slo_scale is not None:
        p['slo']['image_scale'] = float(image_slo_scale)

    # Bind protocol_sha256 on a working copy of bundle for custom SLO/load sweeps
    bundle_copy = copy.deepcopy(bundle)
    bundle_copy.pop('sha256', None)
    bundle_copy['protocol_sha256'] = canonical_hash(p)
    bundle_copy = fb.seal(bundle_copy)

    config = fb.serving(
        p,
        model,
        bundle['gpu_uuids'],
        bundle_copy,
        pools,
        bundle['model_manifests'],
        split=split,
        seed=seed,
        load=load,
        video_fraction=video_fraction,
        pattern=pattern,
        suite='main',
    )
    phase = config['phases'][0]
    return {
        'model': model,
        'gpus': len(bundle['gpu_uuids']),
        'gpu_uuids': list(bundle['gpu_uuids']),
        'seed': seed,
        'load': load,
        'video_fraction': video_fraction,
        'pattern': pattern,
        'video_slo_scale': float(video_slo_scale),
        'image_slo_scale': float(p['slo']['image_scale']),
        'profile': phase['profile'],
        'workload': phase['matrix']['/workload'][0],
        'trace_provenance': phase['matrix']['/trace_provenance'][0],
        'base_options': phase['options'],
        'study_config': config,
    }


def run_rpc_simulate(
    binary: Path,
    profile: Dict[str, Any],
    workload: Sequence[Dict[str, Any]],
    policy: str,
    gpus: int,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Invoke `ditserve rpc simulate` on a single cell and return parsed JSON."""
    payload = {
        'profile': profile,
        'workload': list(workload),
        'policy': policy,
        'gpus': int(gpus),
        'options': options,
        'dispatch_ms': 0.0,
    }
    proc = subprocess.run(
        [str(binary), 'rpc', 'simulate'],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'ditserve rpc simulate failed (rc={proc.returncode}) for policy={policy}: {proc.stderr.strip()}'
        )
    return json.loads(proc.stdout)


def summarize_forms_usage(requests: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Count parallel forms and degrees exercised by video requests in a simulation run."""
    form_counts: Dict[str, int] = {}
    degree_counts: Dict[str, int] = {}
    multi_form_videos = 0
    widened_videos = 0
    video_total = 0

    for req in requests:
        if req.get('kind') != 'video':
            continue
        video_total += 1
        forms = req.get('forms', [])
        degrees = req.get('degrees', [])
        if len(forms) > 1:
            multi_form_videos += 1
        if any(int(d) > 2 for d in degrees):
            widened_videos += 1
        for f in forms:
            form_counts[f] = form_counts.get(f, 0) + 1
        for d in degrees:
            key = str(d)
            degree_counts[key] = degree_counts.get(key, 0) + 1

    return {
        'video_requests': video_total,
        'widened_video_requests': widened_videos,
        'multi_form_video_requests': multi_form_videos,
        'forms_histogram': dict(sorted(form_counts.items())),
        'degrees_histogram': dict(sorted(degree_counts.items(), key=lambda kv: int(kv[0]))),
    }


def _goodput_rps(metrics: Dict[str, Any]) -> float:
    obs_ms = float(metrics.get('observation_ms') or 0.0)
    ontime = float(metrics.get('ontime') or 0.0)
    return (ontime * 1000.0 / obs_ms) if obs_ms > 0 else 0.0


def _summarize_arm_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(runs)
    merged_forms: Dict[str, int] = {}
    for r in runs:
        for k, cnt in r['usage']['forms_histogram'].items():
            merged_forms[k] = merged_forms.get(k, 0) + cnt
    return {
        'overall_slo_attainment': round(sum(float(r['metrics']['deadline_attainment'] or 0.0) for r in runs) / n, 4),
        'video_slo_attainment': round(sum(float(r['video']['attainment'] or 0.0) for r in runs) / n, 4),
        'image_slo_attainment': round(sum(float(r['image']['attainment'] or 0.0) for r in runs) / n, 4),
        'goodput_rps': round(sum(_goodput_rps(r['metrics']) for r in runs) / n, 4),
        'video_p50_ms': round(sum(float(r['video']['latency_p50_ms'] or 0.0) for r in runs) / n, 2),
        'video_p95_ms': round(sum(float(r['video']['latency_p95_ms'] or 0.0) for r in runs) / n, 2),
        'image_p95_ms': round(sum(float(r['image']['latency_p95_ms'] or 0.0) for r in runs) / n, 2),
        'mean_resizes': round(sum(r['resizes'] for r in runs) / n, 2),
        'mean_block_preemptions': round(sum(r['block_preemptions'] for r in runs) / n, 2),
        'widened_video_requests_total': sum(r['usage']['widened_video_requests'] for r in runs),
        'video_requests_total': sum(r['usage']['video_requests'] for r in runs),
        'forms_histogram_total': dict(sorted(merged_forms.items())),
    }


def run_challenge1_simulation_suite(
    bundle_path: Path,
    image_pool_path: Path,
    video_pool_path: Path,
    *,
    binary: Path = DEFAULT_BINARY,
    arms: Sequence[str] = tuple(CHALLENGE1_ARMS.keys()),
    video_slo_scales: Sequence[float] = DEFAULT_VIDEO_SLO_SCALES,
    loads: Sequence[float] = DEFAULT_LOADS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    video_fraction: float = 0.2,
    pattern: str = 'poisson',
) -> Dict[str, Any]:
    """Run the Challenge-1 calibrated simulation matrix across SLO scales, loads, and seeds."""
    bundle = read_json(bundle_path)
    fb.verify_seal(bundle)
    model = bundle['model']
    gpus = len(bundle['gpu_uuids'])
    pools = {
        'image': read_json(image_pool_path),
        'video': read_json(video_pool_path),
    }
    proto = fb.protocol()

    # Compute CFG2 vs SP1 step/service ratios from the bundle for clear SLO interpretation
    video_forms_by_shape, _, _ = load_bundle_forms(bundle_path)
    shape_ratios: Dict[str, Any] = {}
    for sid, entry in bundle['classes'].items():
        cost = entry['cost']
        if cost.get('kind') != 'video':
            continue
        by_form = video_forms_by_shape[sid]
        steps = int(entry['spec']['steps'])
        prep = float(cost.get('prepare_ms', 0.0))
        fin = float(cost.get('finish_ms', 0.0))
        enc = float(cost.get('encode_ms', 0.0))
        sp1_step = float(by_form['sp1'])
        cfg2_step = float(by_form['cfg2'])
        sp1_total = prep + steps * sp1_step + fin + enc
        cfg2_total = prep + steps * cfg2_step + fin + enc
        shape_ratios[sid] = {
            'sp1_service_ms': round(sp1_total, 2),
            'cfg2_service_ms': round(cfg2_total, 2),
            'sp1_to_cfg2_service_ratio': round(sp1_total / cfg2_total, 4),
            'form_step_ms': {fname: round(float(ms), 2) for fname, ms in by_form.items()},
        }

    # Pre-build unique (load, video_slo_scale, seed) cells and dispatch all (cell, arm)
    # simulations across a thread pool (longest arms first to avoid tail stragglers).
    cells_by_key: Dict[tuple, Dict[str, Any]] = {}
    for v_scale in video_slo_scales:
        for seed in seeds:
            key = (round(0.8, 6), round(float(v_scale), 6), int(seed))
            if key not in cells_by_key:
                cells_by_key[key] = build_base_cell(
                    proto,
                    model,
                    bundle,
                    pools,
                    seed=seed,
                    load=0.8,
                    video_fraction=video_fraction,
                    pattern=pattern,
                    video_slo_scale=v_scale,
                )
    for load_val in loads:
        for seed in seeds:
            key = (round(float(load_val), 6), round(0.65, 6), int(seed))
            if key not in cells_by_key:
                cells_by_key[key] = build_base_cell(
                    proto,
                    model,
                    bundle,
                    pools,
                    seed=seed,
                    load=load_val,
                    video_fraction=video_fraction,
                    pattern=pattern,
                    video_slo_scale=0.65,
                )

    def _eval_cell_arm(cell: Dict[str, Any], arm: str, seed: int) -> Dict[str, Any]:
        resolved = resolve_arm_config(arm, model, gpus, cell['base_options'])
        sim = run_rpc_simulate(
            binary,
            cell['profile'],
            cell['workload'],
            resolved['policy'],
            gpus,
            resolved['options'],
        )
        usage = summarize_forms_usage(sim['requests'])
        return {
            'seed': seed,
            'metrics': sim['metrics'],
            'video': sim['by_kind']['video'],
            'image': sim['by_kind']['image'],
            'resizes': sim['resizes'],
            'block_preemptions': sim['block_preemptions'],
            'gpu_utilization': sim['gpu_utilization'],
            'usage': usage,
        }

    priority_order = {'ditserve': 0, 'abl-no-prune': 1}
    ordered_arms = sorted(arms, key=lambda a: (priority_order.get(a, 2), a))
    max_workers = min(16, max(1, os.cpu_count() or 8))
    futures_by_task: Dict[tuple, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for arm in ordered_arms:
            for key, cell in cells_by_key.items():
                futures_by_task[(key, arm)] = pool.submit(_eval_cell_arm, cell, arm, key[2])
        results_by_task: Dict[tuple, Dict[str, Any]] = {
            task_key: fut.result() for task_key, fut in futures_by_task.items()
        }

    # 1. Video SLO Scale Sweep at fixed load=0.8, pattern=poisson
    slo_sweep_rows: List[Dict[str, Any]] = []
    for v_scale in video_slo_scales:
        by_arm: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in arms}
        for seed in seeds:
            key = (round(0.8, 6), round(float(v_scale), 6), int(seed))
            for arm in arms:
                by_arm[arm].append(results_by_task[(key, arm)])
        arm_summaries: Dict[str, Any] = {
            arm: _summarize_arm_runs(runs) for arm, runs in by_arm.items()
        }
        slo_sweep_rows.append({
            'video_slo_scale_vs_sp1': v_scale,
            'approx_video_slo_scale_vs_cfg2': round(
                v_scale * sum(x['sp1_to_cfg2_service_ratio'] for x in shape_ratios.values()) / len(shape_ratios),
                3,
            ),
            'arms': arm_summaries,
        })

    # 2. Load Sweep at video_slo_scale=0.65 (where CFG2 has ~1.27x slack and queuing triggers elastic SP)
    load_sweep_rows: List[Dict[str, Any]] = []
    for load_val in loads:
        by_arm = {arm: [] for arm in arms}
        for seed in seeds:
            key = (round(float(load_val), 6), round(0.65, 6), int(seed))
            for arm in arms:
                res = dict(results_by_task[(key, arm)])
                res.pop('gpu_utilization', None)
                by_arm[arm].append(res)
        arm_summaries = {
            arm: _summarize_arm_runs(runs) for arm, runs in by_arm.items()
        }
        load_sweep_rows.append({
            'load': load_val,
            'video_slo_scale_vs_sp1': 0.65,
            'arms': arm_summaries,
        })

    return {
        'schema': 'ditserve.challenge1-contrasts.v1',
        'model': model,
        'gpus': gpus,
        'bundle_sha256': bundle['sha256'],
        'shape_ratios': shape_ratios,
        'seeds': list(seeds),
        'video_fraction': video_fraction,
        'pattern': pattern,
        'slo_sweep': slo_sweep_rows,
        'load_sweep': load_sweep_rows,
    }


def prepare_gpu_study_configs(
    bundle_path: Path,
    image_pool_path: Path,
    video_pool_path: Path,
    output_dir: Path,
    *,
    arms: Sequence[str] = IN_FRAMEWORK_MAIN_ARMS,
    video_slo_scale: float = 0.65,
    load: float = 0.8,
    seed: int = 301,
) -> Dict[str, Any]:
    """Emit frozen single-arm GPU study configs for Challenge 1 serving verification."""
    bundle = read_json(bundle_path)
    fb.verify_seal(bundle)
    model = bundle['model']
    gpus = len(bundle['gpu_uuids'])
    pools = {
        'image': read_json(image_pool_path),
        'video': read_json(video_pool_path),
    }
    cell = build_base_cell(
        fb.protocol(),
        model,
        bundle,
        pools,
        seed=seed,
        load=load,
        video_slo_scale=video_slo_scale,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        'schema': 'ditserve.challenge1-gpu-configs.v1',
        'model': model,
        'gpus': gpus,
        'seed': seed,
        'load': load,
        'video_slo_scale': video_slo_scale,
        'configs': {},
    }
    for arm in arms:
        resolved = resolve_arm_config(arm, model, gpus, cell['base_options'])
        config = copy.deepcopy(cell['study_config'])
        phase = config['phases'][0]
        phase['policy'] = resolved['policy']
        phase['matrix']['/policy'] = [resolved['policy']]
        phase['options'] = resolved['options']
        v_tag = str(video_slo_scale).replace('.', 'p')
        l_tag = str(load).replace('.', 'p')
        config['experiment'] = f'c1-{model}-{arm}-s{seed}-l{l_tag}-slo{v_tag}'
        config['benchmark']['policies'] = [resolved['policy']]
        config['benchmark']['requested_policies'] = [resolved['policy']]
        config['benchmark']['challenge1_arm'] = {
            'arm': arm,
            'policy': resolved['policy'],
            'mechanism_display': IN_FRAMEWORK_MECHANISM_DISPLAY.get(resolved['policy'], resolved['policy']),
            'description': resolved['description'],
            'forms': resolved['forms'],
            'video_slo_scale': video_slo_scale,
        }
        path = output_dir / f'{config["experiment"]}.json'
        write_json(path, config)
        manifest['configs'][arm] = {
            'path': str(path),
            'sha256': canonical_hash(config),
            'policy': resolved['policy'],
            'forms': resolved['forms'],
        }
    write_json(output_dir / 'manifest.json', manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    sim_p = sub.add_parser('simulate-calibrated', help='Run calibrated C++ simulations across Challenge-1 arms')
    sim_p.add_argument('--bundles', nargs='+', required=True, help='Paths to verified 8-GPU bundle source.json files')
    sim_p.add_argument('--image-pool', required=True, help='Path to sealed image prompt pool JSON')
    sim_p.add_argument('--video-pool', required=True, help='Path to sealed video prompt pool JSON')
    sim_p.add_argument('--binary', default=str(DEFAULT_BINARY), help='Path to compiled ditserve binary')
    sim_p.add_argument('--output', required=True, help='Output JSON path')

    gpu_p = sub.add_parser('prepare-gpu', help='Emit frozen GPU study configs for Challenge-1 arms')
    gpu_p.add_argument('--bundle', required=True, help='Path to verified bundle source.json')
    gpu_p.add_argument('--image-pool', required=True, help='Path to sealed image prompt pool JSON')
    gpu_p.add_argument('--video-pool', required=True, help='Path to sealed video prompt pool JSON')
    gpu_p.add_argument('--output-dir', required=True, help='Directory for generated study configs')
    gpu_p.add_argument('--video-slo-scale', type=float, default=0.65)
    gpu_p.add_argument('--load', type=float, default=0.8)
    gpu_p.add_argument('--seed', type=int, default=301)
    gpu_p.add_argument('--arms', nargs='+', default=list(IN_FRAMEWORK_MAIN_ARMS))

    args = parser.parse_args(argv)
    if args.cmd == 'simulate-calibrated':
        results = {}
        for bpath in args.bundles:
            suite = run_challenge1_simulation_suite(
                Path(bpath),
                Path(args.image_pool),
                Path(args.video_pool),
                binary=Path(args.binary),
            )
            results[suite['model']] = suite
        out = {
            'schema': 'ditserve.challenge1-simulation-report.v1',
            'models': results,
        }
        write_json(Path(args.output), out)
        return 0

    if args.cmd == 'prepare-gpu':
        prepare_gpu_study_configs(
            Path(args.bundle),
            Path(args.image_pool),
            Path(args.video_pool),
            Path(args.output_dir),
            arms=tuple(args.arms),
            video_slo_scale=args.video_slo_scale,
            load=args.load,
            seed=args.seed,
        )
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
