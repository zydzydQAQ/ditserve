"""CPU-only run02 reporting; guarded normal timers, numerical checks and quality."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
import struct
import zlib

from .workflow_budget import atomic_json
from .workflow_report import preserve_dependencies
from .workflow_retention import POLICY, archived_run, finalize_workflow
from .workflow_serving_study import binding, digest


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def compact(value):
    """Telemetry remains in its hashed source; CPU micro samples remain useful."""
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            telemetry = isinstance(item, list) and (key in {'nvml_samples', 'timeline', 'phase_records'} or
                (key == 'phases' and any(isinstance(row, dict) and 'start_ns' in row for row in item)) or
                (key == 'samples' and (value.get('kind') == 'prefetch' or any(isinstance(row, dict) and
                    ('memory_total_bytes' in row or 'gpu_busy_percent' in row or 'compute_processes' in row) for row in item))))
            result[key] = dict(retained_in_source=True, count=len(item)) if telemetry else compact(item)
        return result
    return value


def observations(result):
    return result.get('observations', [])


def completed_requests(result):
    if 'observations' in result:
        return [row for observation in observations(result) for row in observation.get('requests', []) if row.get('status') == 'completed']
    return [row for row in result.get('requests', []) if row.get('outputs')] if result.get('kind') == 'prefetch' else []


def exact_outputs(left, right):
    a, b = completed_requests(left), completed_requests(right)
    lhs, rhs = {row['request_id']: row for row in a}, {row['request_id']: row for row in b}
    if len(lhs) != len(a) or len(rhs) != len(b):
        return dict(status='failed', passed=False, reason='duplicate request IDs')
    if not lhs or lhs.keys() != rhs.keys():
        return dict(status='unverified', passed=False, reason='incomplete or unequal completed request sets')
    mismatches = []
    for key in lhs:
        for name in ('latent', 'video'):
            x, y = lhs[key].get('outputs', lhs[key]).get(name, {}), rhs[key].get('outputs', rhs[key]).get(name, {})
            if (not x.get('sha256') or x.get('sha256') != y.get('sha256') or x.get('finite') is not True or
                    y.get('finite') is not True or x.get('shape') != y.get('shape') or x.get('dtype') != y.get('dtype')):
                mismatches.append(dict(request_id=key, output=name))
        if not lhs[key].get('spec') or lhs[key].get('spec') != rhs[key].get('spec') or lhs[key].get('spec_sha256') != rhs[key].get('spec_sha256'):
            mismatches.append(dict(request_id=key, output='request_spec'))
    return dict(status='failed' if mismatches else 'passed', passed=not mismatches, requests=len(lhs), mismatches=mismatches)


def performance_valid(result):
    reasons = []
    if result.get('status') != 'completed':
        reasons.append('result status: ' + str(result.get('status')))
    if result.get('kind') in ('quality', 'P5') or result.get('include_in_performance') is False:
        reasons.append('numerical evidence only')
    if result.get('performance_evidence') is False:
        reasons.append('performance_evidence=false')
    if result.get('schema') == 'ditserve.workflow.serving.v1':
        if result.get('performance_evidence') is not True or not observations(result):
            reasons.append('missing positive serving performance evidence')
        for observation in observations(result):
            if observation.get('status') != 'completed' or observation.get('performance_evidence') is not True:
                reasons.append('observation not eligible: ' + str(observation.get('id')))
            for gate in ('capacity_gate', 'memory_margin_gate'):
                if observation.get(gate, {}).get('status') != 'passed':
                    reasons.append(gate + ' not passed: ' + str(observation.get('id')))
    for gate in ('capacity_gate', 'memory_margin_gate'):
        if gate in result and result[gate].get('status') != 'passed':
            reasons.append(gate + ' not passed')
    qualification = result.get('qualification') or {}
    if any(qualification.get(key) is False for key in ('passed', 'finite_and_shape', 'repeated_exact', 'cpu_bridge_exact')):
        reasons.append('explicit numerical qualification failed')
    if result.get('kind') == 'P4':
        if result.get('performance_evidence') is not True:
            reasons.append('P4 performance not qualified')
        for pair in result.get('pairs', []):
            for mode in pair.get('modes', {}).values():
                if mode.get('performance_evidence') is not True or mode.get('memory_margin_gate', {}).get('status') != 'passed' or mode.get('correctness', {}).get('passed') is not True:
                    reasons.append('P4 arm correctness/capacity/performance gate not passed')
    return dict(eligible=not reasons, reasons=sorted(set(reasons)))


def _numeric_status(result):
    q = result.get('qualification') or {}
    if result.get('status') in ('qualification_failed', 'correctness_failed') or any(q.get(key) is False for key in ('passed', 'finite_and_shape', 'repeated_exact', 'cpu_bridge_exact')):
        return 'failed'
    if result.get('kind') == 'quality':
        return 'compared' if result.get('status') == 'completed' and q.get('cpu_bridge_exact') is True else 'unverified'
    return 'passed' if q.get('passed') is True else 'unverified'


def collect(root):
    root = Path(root)
    study = read(root / 'study.json', dict(jobs=[], status='not_prepared', run_id=root.name))
    results, phases, sources, failures = {}, [], [], []
    for job in study.get('jobs', []):
        logical = job.get('logical_id', job['id'])
        ledger_path = root / 'launches' / job['id'] / 'launch.json'
        ledger = read(ledger_path, {})
        path = Path(job.get('output', root / 'measurements' / job['id'])) / 'run.json'
        if not path.is_relative_to(root):
            path = root / 'measurements' / job['id'] / 'run.json'
        result = read(path)
        terminal = result and result.get('status') not in ('running', 'pending', None)
        guarded = ledger.get('status') == 'completed' and terminal
        superseded = bool(job.get('superseded'))
        if terminal:
            sources.append(dict(**binding(path), job=job['id'], logical_id=logical, launch_status=ledger.get('status'), eligible=bool(guarded and not superseded), superseded=job.get('superseded')))
        profiling = bool(job.get('profiling') or job.get('counters') or job.get('family') in ('trace', 'profiler'))
        if guarded and not superseded:
            if logical in results:
                raise ValueError('multiple active attempts for logical job: ' + logical)
            results[logical] = dict(result, include_in_performance=False, profiling=True) if profiling else result
        if not superseded and ledger.get('status') in {'failed', 'timeout', 'budget_exhausted', 'interrupted', 'gpu_monitor_unavailable', 'interrupted_foreign_gpu_workload'}:
            failures.append(dict(phase=job['id'], error=ledger['status'], launch=str(ledger_path)))
        if terminal and not guarded and not superseded:
            failures.append(dict(phase=job['id'], error='result lacks completed guarded launch', result_status=result['status'], source=str(path)))
        if guarded and not superseded and result.get('status') in ('failed', 'oom', 'capacity_rejected', 'invalid_memory_margin', 'qualification_failed', 'correctness_failed', 'capacity_failed'):
            failures.append(dict(phase=job['id'], error=result['status'], scientific_outcome=True, source=str(path)))
        evidence = performance_valid(result) if guarded and not superseded else dict(eligible=False, reasons=['superseded evidence' if superseded else 'no completed guarded result'])
        if profiling:
            evidence = dict(eligible=False, reasons=['profiler timing is not normal performance'])
        execution = 'superseded' if superseded else 'completed' if guarded else ledger.get('status', 'not_started')
        if guarded and not superseded and result.get('status') in ('failed', 'interrupted'):
            execution = result['status']
        phases.append(dict(id=job['id'], logical_id=logical, status=execution, result_status=result.get('status') if result else None,
            superseded=job.get('superseded'), attempt=job.get('attempt', 1 if ledger else 0), family=job.get('family', 'unknown'),
            qualification=dict(scopes=dict(numeric=dict(status=_numeric_status(result or {})), semantic=dict(status='unverified'), performance=dict(status='measured' if evidence['eligible'] else 'unverified'))),
            performance_validity=evidence, metrics=compact(result) if guarded else {}, source=str(path) if result else None))
    return study, results, phases, sources, failures


def quality_recipe(row):
    """The encoder emits flat recipe fields; accept older nested fixtures too."""
    if isinstance(row.get('recipe'), dict):
        return row['recipe']
    return {key: row[key] for key in ('id', 'case', 'seed', 'frames', 'spec', 'spec_sha256') if key in row}


def quality_evidence(results):
    records = [(key, value) for key, value in results.items() if value.get('kind') == 'quality']
    comparisons, failures = [], []
    for key, record in records:
        q = record.get('qualification') or {}
        if record.get('status') != 'completed' or q.get('cpu_bridge_exact') is not True or q.get('finite_and_shape') is not True:
            failures.append(dict(job=key, reason='CPU bridge/finite/shape qualification not passed'))
        for row in record.get('requests', []):
            if row.get('variant') not in ('T0', 'T1', 'T2'):
                continue
            for name in ('latent', 'video'):
                metric = row.get('comparison', {}).get(name, {})
                if not metric or any(metric.get(field) is not True for field in ('finite', 'reference_finite', 'shape_match')):
                    failures.append(dict(job=key, variant=row.get('variant'), output=name, reason='comparison missing or nonfinite/shape mismatch'))
                comparisons.append(dict(job=key, variant=row['variant'], recipe=quality_recipe(row), output=name, metrics=metric))
            if row['variant'] == 'T0' and row.get('baseline_bridge_exact') is not True:
                failures.append(dict(job=key, reason='baseline bridge not exact'))
    compared = any(row['variant'] == 'T1' for row in comparisons) and not failures
    return dict(status='failed' if failures else 'compared' if compared else 'unverified', compared=compared,
                sources=[key for key, value in records], comparisons=comparisons, failures=failures, semantic='unverified',
                interpretation='cross-device differences reported; no automatic perceptual threshold or bitwise equivalence claim')


def _pair(left, right, *, family, load, repeat, reference, candidate, quality, confirmation=False):
    reasons = [reason for value in (left, right) for reason in performance_valid(value)['reasons']]
    a = observations(left)[0] if len(observations(left)) == 1 else {}
    b = observations(right)[0] if len(observations(right)) == 1 else {}
    ac, bc = left.get('configuration', {}), right.get('configuration', {})
    ao, bo = (ac.get('observations') or [{}])[0], (bc.get('observations') or [{}])[0]
    if not ao.get('requests') or ao.get('requests') != bo.get('requests'):
        reasons.append('paired frozen trajectories differ')
    for field in ('arrival_window_ms', 'observation_cutoff_ms', 'estimates_ms'):
        if ao.get(field) != bo.get(field):
            reasons.append('paired observation differs: ' + field)
    for field in ('worker_cpu_sets', 'controller_cpu', 'worker_devices', 'queue_limit', 'prefetch_depth'):
        if ac.get(field) != bc.get(field):
            reasons.append('paired resource differs: ' + field)
    if family == 'online':
        numeric = exact_outputs(left, right)
        if not numeric['passed']:
            reasons.append('scheduling outputs not verified exact')
    else:
        numeric = dict(status=quality['status'], passed=None, quality_sources=quality['sources'], reason='device/placement change: numerical comparison only')
        if not quality['compared']:
            reasons.append('FP32 quality comparison missing or failed')
    rows = [completed_requests(value) for value in (left, right)]
    expected_ids = {row['request_id'] for row in ao.get('requests', [])}
    if any(len(group) != len(expected_ids) or {row['request_id'] for row in group} != expected_ids for group in rows):
        reasons.append('completed requests differ from frozen trajectory')
    if not all(rows) or not all(row.get(name, {}).get('finite') is True for group in rows for row in group for name in ('latent', 'video')):
        reasons.append('missing finite outputs')
    if not all(obs.get('summary', {}).get('unfinished') == 0 and obs.get('summary', {}).get('rejected') == 0 for obs in (a, b)):
        reasons.append('incomplete paired request sets')
    metric = 'latency_p50_ms' if family == 'placement' and load == .5 else 'slo_attainment'
    av, bv = a.get('summary', {}).get(metric), b.get('summary', {}).get(metric)
    numbers = all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (av, bv))
    improvement = ((av - bv) / av if av > 0 and bv > 0 else None) if numbers and metric == 'latency_p50_ms' else bv - av if numbers else None
    if improvement is None:
        reasons.append('primary metric unavailable')
    return dict(family=family, load=load, repeat=repeat, reference=reference, candidate=candidate, confirmation=confirmation,
        metric=metric, reference_value=av, candidate_value=bv, improvement=improvement, threshold=.05,
        improvement_basis='(baseline - candidate) / baseline latency reduction' if metric == 'latency_p50_ms' else 'candidate - baseline SLO attainment (fraction; 0.05 = five percentage points)',
        numeric=numeric, eligible=not reasons, invalid_reasons=sorted(set(reasons)),
        screening_pass=not reasons and improvement >= .05, trace_sha256=digest(ao.get('requests', [])),
        summary_reference=a.get('summary'), summary_candidate=b.get('summary'))


def compare(results):
    pairs, quality = [], quality_evidence(results)
    for load in (.5, .8, 1.1):
        label = str(load).replace('.', 'p')
        for repeat in range(3):
            reference = f'dual_cpu-shared_fifo-l{label}-r{repeat}'
            for family, candidate in [('placement', f'dedicated_fp32-shared_fifo-l{label}-r{repeat}'), ('online', f'dual_cpu-least_work_fifo-l{label}-r{repeat}')]:
                if reference in results and candidate in results:
                    pairs.append(_pair(results[reference], results[candidate], family=family, load=load, repeat=repeat, reference=reference, candidate=candidate, quality=quality))
    return pairs


def select_extension(root):
    unused, results, unused_phases, unused_sources, unused_failures = collect(root)
    pairs, candidates = compare(results), []
    for family in ('placement', 'online'):
        for load in (.5, .8, 1.1):
            group = [row for row in pairs if row['family'] == family and row['load'] == load]
            if len(group) == 3 and {row['repeat'] for row in group} == {0, 1, 2} and all(row['screening_pass'] for row in group):
                candidates.append(dict(family=family, load=load, metric=group[0]['metric'], median_improvement=statistics.median(row['improvement'] for row in group), evidence=[row['candidate'] for row in group]))
    selected = min(candidates, key=lambda row: (0 if row['family'] == 'placement' else 1, -row['median_improvement'], row['load'])) if candidates else None
    atomic_json(Path(root) / 'extension-selection.json', dict(selected=selected, candidates=candidates,
        criterion='all three paired traces each >=5% latency reduction relative to baseline or >=5 percentage point SLO gain, with all validity/numerical/capacity gates satisfied', semantics='screening selection; semantic quality remains unverified'))
    return selected


def bootstrap_paired(values, seed=20261025, draws=10000):
    if len(values) < 3:
        return None
    if type(draws) is not int or draws < 100 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError('bootstrap requires finite paired blocks and >=100 draws')
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return dict(low=means[int((draws - 1) * .025)], high=means[int((draws - 1) * .975)], confidence=.95,
        estimate=statistics.mean(values), unit='mean paired trajectory improvement', blocks=len(values), draws=draws, seed=seed,
        resampling_unit='entire paired trajectory; both arms remain together', caution='few independent blocks; exploratory interval')


def confirmation_summary(results):
    quality = quality_evidence(results)
    screening = {digest(obs.get('requests', [])) for value in results.values() if value.get('schema') == 'ditserve.workflow.serving.v1' and not value.get('configuration', {}).get('confirmation') for obs in value.get('configuration', {}).get('observations', [])}
    groups = {}
    for identifier, value in results.items():
        config = value.get('configuration', {})
        if not config.get('confirmation'):
            continue
        selection = config.get('selection', {})
        key = (selection.get('family'), config.get('load_fraction'), config.get('repeat'))
        arm = 'reference' if config.get('placement') == 'dual_cpu' and config.get('policy') == 'shared_fifo' else 'candidate'
        if arm in groups.setdefault(key, {}):
            groups[key]['duplicate'] = True
        groups[key][arm] = (identifier, value)
    pairs, seen = [], set()
    for (family, load, repeat), group in sorted(groups.items(), key=lambda item: str(item[0])):
        if 'reference' not in group or 'candidate' not in group:
            pairs.append(dict(family=family, load=load, repeat=repeat, eligible=False, invalid_reasons=['confirmation arm missing']))
            continue
        reference, left = group['reference']; candidate, right = group['candidate']
        row = _pair(left, right, family=family, load=load, repeat=repeat, reference=reference, candidate=candidate, quality=quality, confirmation=True)
        if row['trace_sha256'] in screening or row['trace_sha256'] in seen:
            row['invalid_reasons'].append('confirmation trajectory reused')
        if group.get('duplicate'):
            row['invalid_reasons'].append('duplicate arm')
        seen.add(row['trace_sha256']); row['eligible'] = not row['invalid_reasons']; pairs.append(row)
    valid = [row for row in pairs if row['eligible']]
    complete = len(pairs) == 6 and len(valid) == 6 and {row['repeat'] for row in valid} == set(range(6)) and len({(row['family'], row['load']) for row in valid}) == 1
    return dict(status='completed' if complete else 'partial' if pairs else 'not_started', expected_blocks=6, valid_blocks=len(valid), pairs=pairs,
        interval=bootstrap_paired([row['improvement'] for row in valid]) if complete else None,
        interpretation='fresh complete paired traces only; no CI before all six planned blocks pass')


def supplemental_rows(results):
    micro, prefetch, overlap, sp, groups = [], [], [], [], {}
    for identifier, record in results.items():
        if record.get('profiling'):
            continue
        kind, config = record.get('kind'), record.get('configuration', {})
        if kind == 'cpu_micro':
            for case in sorted({row.get('case') for row in record.get('samples', []) if row.get('case')}):
                rows = [row for row in record['samples'] if row.get('case') == case and not row.get('warmup', False)]
                values = [row['encode_ms'] for row in rows if isinstance(row.get('encode_ms'), (int, float))]
                cores = [row['process_average_cores'] for row in rows if isinstance(row.get('process_average_cores'), (int, float))]
                micro.append(dict(job=identifier, cpu_threads=config.get('cpu_threads'), case=case, samples=len(values), encode_ms_median=statistics.median(values) if values else None, process_average_cores_median=statistics.median(cores) if cores else None, qualification=record.get('qualification'), performance=performance_valid(record)))
        elif kind == 'prefetch':
            groups.setdefault(config.get('cpu_threads'), {})[config.get('prefetch_depth', config.get('depth'))] = (identifier, record)
        elif kind == 'P4':
            for row in record.get('pairs', []):
                modes = row.get('modes', {})
                overlap.append(dict(job=identifier, frames_a=row.get('frames_a'), frames_b=row.get('frames_b'), repeat=row.get('repeat'), speedup=row.get('speedup'), stage_pair_speedup=row.get('stage_pair_speedup'),
                    serial_wall_ms=modes.get('serial', {}).get('wall_ms'), streams_wall_ms=modes.get('streams', {}).get('wall_ms'),
                    serial_stage_pair_ms=modes.get('serial', {}).get('stage_pair_wall_ms'), streams_stage_pair_ms=modes.get('streams', {}).get('stage_pair_wall_ms'),
                    correctness={name: value.get('correctness') for name, value in modes.items()}, performance=performance_valid(record), device_overlap='unverified without matching CUDA timeline'))
            if not record.get('pairs'):
                overlap.append(dict(job=identifier, status=record.get('status'), qualification=record.get('qualification'), performance=performance_valid(record)))
        elif kind == 'P5':
            for row in record.get('variants', []):
                sp.append(dict(job=identifier, frames=row.get('frames'), mode=row.get('mode'), status=row.get('status'),
                    prefixes=[{key: compact(value.get(key)) for key in ('prefix', 'scheduler_steps', 'passed', 'first_nonzero_step', 'first_threshold_failure_step', 'video_comparison', 'localization')} for value in row.get('prefixes', [])],
                    qualification=record.get('qualification'), performance_evidence=False))
            if not record.get('variants'):
                sp.append(dict(job=identifier, status=record.get('status'), qualification=record.get('qualification'), performance_evidence=False))
    for cores, depths in sorted(groups.items()):
        if 0 not in depths:
            prefetch.append(dict(cpu_threads=cores, status='depth0_missing')); continue
        reference, left = depths[0]
        for depth, (candidate, right) in sorted(depths.items()):
            if depth == 0:
                continue
            numeric = exact_outputs(left, right)
            a, b = {row['repeat']: row for row in left.get('queues', [])}, {row['repeat']: row for row in right.get('queues', [])}
            ratios = [a[key]['wall_ms'] / b[key]['wall_ms'] for key in sorted(a.keys() & b.keys()) if a[key].get('wall_ms') and b[key].get('wall_ms')]
            prefetch.append(dict(cpu_threads=cores, depth=depth, reference=reference, candidate=candidate, paired_speedups=ratios,
                median_paired_speedup=statistics.median(ratios) if ratios else None, numeric=numeric,
                eligible=numeric['passed'] and a.keys() == b.keys() and bool(ratios) and all(performance_valid(value)['eligible'] for value in (left, right)),
                capacity_qualification=right.get('memory_margin_gate', {}).get('status', 'unverified'), scope='same cores; depth0 versus candidate; no cross-core exact claim'))
    return dict(cpu_micro=micro, prefetch=prefetch, overlap=overlap, sp=sp)


def cuda_overlap_trace(root, study):
    """Independent profiler evidence; never an end-to-end performance sample."""
    root = Path(root)
    jobs = [job for job in study.get('jobs', []) if job.get('logical_id', job['id']) == 'trace-overlap' and not job.get('superseded')]
    path = root / 'traces' / ((jobs[0]['id'] if len(jobs) == 1 else 'trace-overlap') + '.parsed.json')
    if not path.is_file():
        return dict(status='unavailable', reason='P4 parsed CUDA trace is missing', source=None, arms=[], include_in_performance=False)
    parsed = read(path)
    ledger = read(root / 'launches' / jobs[0]['id'] / 'launch.json', {}) if len(jobs) == 1 else {}
    source_matches = len(jobs) == 1 and Path(parsed.get('source') or '').name.startswith(jobs[0]['id'] + '.')
    guarded = ledger.get('status') == 'completed' and source_matches
    valid_schema = parsed.get('schema_version') == 'ditserve.workflow_trace.v1'
    kernels = parsed.get('availability', {}).get('cuda_kernels') is True
    rows = []
    for arm in parsed.get('arms', []) if valid_schema else []:
        if not arm.get('name', '').startswith('P4/'):
            continue
        for device, values in arm.get('devices', {}).items():
            available = guarded and kernels and values.get('dit_vae_kernel_overlap_available') is True
            rows.append(dict(arm=arm['name'], device=device, available=available,
                dit_vae_kernel_overlap_ms=values.get('dit_vae_kernel_overlap_ms') if available else None,
                dit_a_kernel_union_ms=values.get('dit_a_kernel_union_ms') if available else None,
                vae_b_kernel_union_ms=values.get('vae_b_kernel_union_ms') if available else None,
                basis=values.get('dit_vae_kernel_overlap_basis')))
    available = any(row['available'] for row in rows)
    return dict(status='available' if available else 'unavailable', source=binding(path), capture_source=parsed.get('source'),
        launch_status=ledger.get('status', 'unmonitored'), source_matches_current_attempt=source_matches, job=jobs[0]['id'] if len(jobs) == 1 else None,
        availability=parsed.get('availability'), warnings=parsed.get('warnings', []), arms=rows,
        reason=None if available else 'requires a completed guarded capture and both correlated kernel roles',
        include_in_performance=False, interpretation='same-device A DiT / B VAE compute kernel intersection; profiler timing is separate from normal paired speedups')


def _table(headers, rows):
    def esc(value):
        return html.escape('unavailable' if value is None else json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(round(value, 5)) if isinstance(value, float) else str(value))
    return '<div class="scroll"><table><tr>' + ''.join('<th>' + esc(value) + '</th>' for value in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + esc(value) + '</td>' for value in row) + '</tr>' for row in rows) + '</table></div>'


def render_html(run):
    m, budget = run['metrics'], run['budget']
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>run02 DiT Serving</title><style>body{font:15px system-ui;max-width:1350px;margin:32px auto;padding:0 20px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}pre{white-space:pre-wrap}h1,h2{color:#17436b}</style>'
    page += '<h1>双4090 DiT Serving 补充实验</h1><p>技术状态：' + html.escape(run['status']) + '；累计计费 ' + str(round(budget.get('charged_seconds', 0) / 60, 1)) + ' / 480 分钟。</p><p>短队列 p95 为经验分位数，语义质量未验收；尚无数据时表格留空。</p>'
    page += '<h2>任务与资格</h2>' + _table(['任务', '类别', '技术状态', '结果状态', '数值', '性能'], [[row['id'], row['family'], row['status'], row['result_status'], row['qualification']['scopes']['numeric']['status'], row['qualification']['scopes']['performance']['status']] for row in run['phases']])
    page += '<h2>P1/P2 配对筛选</h2>' + _table(['类别', '负载', '轨迹', '指标', '基线', '候选', '改善', '有效', '原因'], [[row.get(key) for key in ('family', 'load', 'repeat', 'metric', 'reference_value', 'candidate_value', 'improvement', 'eligible', 'invalid_reasons')] for row in m['paired_comparisons']])
    page += '<h2>新轨迹确认</h2><p>' + html.escape(json.dumps({key: value for key, value in m['confirmation'].items() if key != 'pairs'}, ensure_ascii=False)) + '</p>'
    page += _table(['类别', '负载', '轨迹', '改善', '有效', '原因'], [[row.get(key) for key in ('family', 'load', 'repeat', 'improvement', 'eligible', 'invalid_reasons')] for row in m['confirmation']['pairs']])
    page += '<h2>P3 CPU 编码与预取</h2>' + _table(['核数', '文本', '样本', '编码中位数 ms', 'CPU平均核数'], [[row.get(key) for key in ('cpu_threads', 'case', 'samples', 'encode_ms_median', 'process_average_cores_median')] for row in m['cpu_micro']])
    page += _table(['核数', '深度', '配对加速比', 'exact', '有效', '容量资格'], [[row.get(key) for key in ('cpu_threads', 'depth', 'paired_speedups', 'numeric', 'eligible', 'capacity_qualification')] for row in m['prefetch']])
    page += '<h2>P4 DiT/VAE</h2><p>完整两请求与局部阶段加速比分别列示；没有 CUDA trace 时不宣称实际设备重叠。</p>'
    page += _table(['A帧', 'B帧', '轮次', '串行完整 ms', '双stream完整 ms', '完整加速比', '阶段加速比', '性能资格'], [[row.get(key) for key in ('frames_a', 'frames_b', 'repeat', 'serial_wall_ms', 'streams_wall_ms', 'speedup', 'stage_pair_speedup', 'performance')] for row in m['overlap']])
    trace = m['cuda_overlap_trace']
    page += '<h2>P4 独立 CUDA 时间线</h2><p>' + html.escape(trace['status']) + '；只作设备重叠证据，不纳入正常性能。' + html.escape(trace.get('reason') or '') + '</p>'
    page += _table(['NVTX arm', 'GPU', '可用', 'A DiT kernel union ms', 'B VAE kernel union ms', '实际 kernel 交集 ms'], [[row.get(key) for key in ('arm', 'device', 'available', 'dit_a_kernel_union_ms', 'vae_b_kernel_union_ms', 'dit_vae_kernel_overlap_ms')] for row in trace['arms']])
    page += '<h2>P5 SP2 数值定位</h2>' + _table(['帧数', '方案', '结果', '前缀与首次差异', '性能'], [[row.get('frames'), row.get('mode'), row.get('status'), row.get('prefixes'), '未测量'] for row in m['sp']])
    page += '<h2>编码质量比较</h2><p>数值状态：' + html.escape(m['quality']['status']) + '；FP32不同设备只作比较，语义未验收。</p>'
    page += _table(['方案', '用例', '输出', 'max_abs', 'RMSE', 'cosine', 'exact'], [[row['variant'], row['recipe'].get('id'), row['output']] + [row['metrics'].get(key) for key in ('max_abs', 'rmse', 'cosine', 'exact')] for row in m['quality']['comparisons']])
    page += '<p>匿名帧浏览单独生成：<code>python -m ditserve.evaluation.workflow_serving_report --root ... --quality-gallery</code></p>'
    page += '<h2>限制与失败</h2><pre>' + html.escape(json.dumps(m['limitations'] + run['failures'], ensure_ascii=False, indent=2)) + '</pre></html>'
    return page


def build_report(root, *, final=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if final:
        if read(root / 'study.json', {}).get('active_job') or read(root / 'budget.json', {}).get('active'):
            raise RuntimeError('cannot finalize while experiments are active')
        saved = archived_run(root)
        if saved is not None:
            return saved
    study, results, phases, sources, failures = collect(root)
    budget = read(root / 'budget.json', {})
    if final and (study.get('active_job') or budget.get('active')):
        raise RuntimeError('cannot finalize while experiments are active')
    current = [row for row in phases if row['status'] != 'superseded']
    complete = bool(current) and all(row['status'] == 'completed' for row in current)
    request_rows = [dict(row, experiment_job=identifier, evidence_kind=result.get('kind'), include_in_performance=performance_valid(result)['eligible']) for identifier, result in results.items() for row in completed_requests(result)]
    request_rows += [dict(row, experiment_job=identifier, evidence_kind=result.get('kind'), include_in_performance=False) for identifier, result in results.items() for obs in observations(result) for row in obs.get('requests', []) if row.get('status') != 'completed']
    run = dict(schema='ditserve.run.v1', experiment='workflow', run_id=root.name, status='completed' if complete else 'partial', created_utc=datetime.now(timezone.utc).isoformat(),
        config=dict(root=str(root), study=study, analysis_source=read(root / 'analysis-source.json'), units=dict(time='ms', memory='bytes'), completion_meaning='technical completion is separate from numerical/semantic qualification'),
        metrics=dict(paired_comparisons=compare(results), confirmation=confirmation_summary(results), sources=sources, quality=quality_evidence(results), **supplemental_rows(results), cuda_overlap_trace=cuda_overlap_trace(root, study), extension_selection=read(root / 'extension-selection.json'),
            limitations=['Small-trace p95 is descriptive; requests in a trace are not independent repetitions.', 'CI resamples complete fresh paired trajectories, excluding screening.', 'GPU FP32 versus CPU FP32 is compared, not assumed bitwise equivalent.', 'Semantic quality is unverified; SLO rates are not quality-qualified goodput.', 'NVML samples can miss short peaks; raw telemetry stays in source files.', '4090 evidence does not establish A100 or eight-GPU scalability.']),
        requests=request_rows, phases=phases, failures=failures, budget=budget, retention=dict(pending_dependencies=[], policy=POLICY),
        artifacts=[dict(path=str(Path(row['path']).relative_to(root)), bytes=row['bytes'], sha256=row['sha256'], kind='control', dependencies=[]) for row in sources])
    if final:
        preserve_dependencies(root, run)
    (root / ('ANALYSIS.html' if final else 'ANALYSIS-LIVE.html')).write_text(render_html(run), encoding='utf-8')
    atomic_json(root / ('run.json' if final else 'analysis-preview.json'), run)
    return run


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _png(path, array):
    import numpy as np
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4) or not np.isfinite(array).all():
        raise ValueError('expected finite HWC frame')
    if array.dtype.kind == 'f':
        array = np.clip(array, 0, 1) * 255
    array = array.astype(np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = np.ascontiguousarray(array[:, :, :3])
    def chunk(name, payload):
        return struct.pack('>I', len(payload)) + name + payload + struct.pack('>I', zlib.crc32(name + payload) & 0xffffffff)
    height, width = array.shape[:2]
    body = b''.join(b'\0' + array[index].tobytes() for index in range(height))
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(body, 6)) + chunk(b'IEND', b''))


def generate_quality_gallery(root, seed=20261025):
    """Explicit CPU-only command: five frames per clip, labels in separate map."""
    import numpy as np
    root = Path(root).resolve()
    unused, results, unused_p, unused_s, unused_f = collect(root)
    groups = {}
    for value in results.values():
        if value.get('kind') != 'quality' or value.get('status') != 'completed':
            continue
        for row in value.get('requests', []):
            artifact = row.get('artifacts', {}).get('video', {})
            if (row.get('variant') in ('cpu_direct', 'T1', 'T2') and artifact.get('path')
                    and artifact.get('retained') is not False):
                groups.setdefault(quality_recipe(row)['id'], []).append(row)
    if not groups:
        raise ValueError('no retained guarded quality videos; run quality with save_videos=true for manual review')
    destination = root / 'quality-gallery'; destination.mkdir(exist_ok=True)
    rng, mapping, sections = random.Random(seed), [], []
    for number, (recipe, candidates) in enumerate(sorted(groups.items()), 1):
        rng.shuffle(candidates); cards = []
        for index, row in enumerate(candidates):
            label = 'group%02d-%s' % (number, chr(65 + index)); artifact = row['artifacts']['video']; source = Path(artifact['path'])
            prefix = Path('/data/workflow') / root.name
            if source.is_relative_to(prefix):
                source = root / source.relative_to(prefix)
            elif not source.is_absolute():
                source = root / source
            source = source.resolve()
            if not source.is_relative_to(root):
                raise ValueError('artifact outside study')
            if not source.is_file():
                raise ValueError('quality video is no longer retained; run quality with save_videos=true for manual review')
            source_hash = _file_hash(source)
            if artifact.get('file_sha256') and source_hash != artifact['file_sha256']:
                raise ValueError('quality artifact hash changed')
            video = np.load(source, mmap_mode='r', allow_pickle=False)
            if video.ndim == 5 and video.shape[0] == 1:
                video = video[0]
            if video.ndim != 4:
                raise ValueError('expected one video')
            if video.shape[-1] not in (1, 3, 4) and video.shape[1] in (1, 3, 4):
                video = video.transpose(0, 2, 3, 1)
            frames = sorted({round((len(video) - 1) * fraction) for fraction in (0, .25, .5, .75, 1)}); images = []
            for frame in frames:
                filename = label + '-f%d.png' % frame; _png(destination / filename, video[frame])
                images.append('<figure><img loading="lazy" src="' + filename + '"><figcaption>frame ' + str(frame) + '</figcaption></figure>')
            cards.append('<article><h3>' + label + '</h3><div>' + ''.join(images) + '</div></article>')
            mapping.append(dict(anonymous_id=label, recipe=recipe, variant=row['variant'], source=str(source), file_sha256=source_hash, frame_indices=frames))
        prompt = quality_recipe(candidates[0]).get('spec', {}).get('prompt', '')
        sections.append('<section><h2>Group %02d</h2><p>%s</p>%s</section>' % (number, html.escape(prompt[:1000]), ''.join(cards)))
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Anonymous video frames</title><style>body{font:15px system-ui;margin:24px}article>div{display:flex;flex-wrap:wrap}figure{margin:5px}img{width:220px}section{border-bottom:2px solid #ddd}</style><h1>匿名视频帧浏览</h1><p>五帧抽样不能代替运动连续性检查；本页不作质量通过判断。</p>' + ''.join(sections) + '</html>'
    (destination / 'index.html').write_text(page, encoding='utf-8')
    atomic_json(root / 'quality-gallery-map.json', dict(seed=seed, mappings=mapping, semantic='unverified'))
    return dict(page=str(destination / 'index.html'), clips=len(mapping), mapping=str(root / 'quality-gallery-map.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--final', action='store_true'); parser.add_argument('--archive', type=Path)
    parser.add_argument('--quality-gallery', action='store_true', help='CPU-only anonymous frame browser; no canonical finalization')
    parser.add_argument('--keep-raw', action='store_true', help='Archive without pruning redundant raw artifacts')
    args = parser.parse_args(argv)
    if args.quality_gallery:
        if args.final or args.archive:
            parser.error('gallery is a separate command')
        print(json.dumps(generate_quality_gallery(args.root), ensure_ascii=False)); return
    if args.archive and not args.final:
        parser.error('--archive requires --final')
    build_report(args.root, final=args.final)
    if args.final:
        from ..results import load, verify
        outcome = finalize_workflow(args.root, args.archive, keep_raw=args.keep_raw)
        if outcome.get('status') != 'completed':
            raise RuntimeError('canonical archive finalization did not complete: ' + json.dumps(outcome))
        destination = Path(outcome['workbook']).parent
        expected = read(args.root / 'run.json')
        if load(destination)['run'] != expected:
            raise RuntimeError('archive canonical round-trip mismatch')
        print(json.dumps(dict(finalization=outcome, verification=verify(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
