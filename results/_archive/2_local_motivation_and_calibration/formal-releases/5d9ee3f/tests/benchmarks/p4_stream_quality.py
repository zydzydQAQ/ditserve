#!/usr/bin/env python3
"""Offline P4 quality measurement, one pair and frame at a time.

Reads existing hash-bound runs/NPY files; never launches workers, downloads
models, deletes artifacts, or closes retention dependencies. LPIPS is opt-in,
using only the strict offline adapter after a real CPU checkpoint selftest.
VBench remains unknown. Sealed v2 batches support source-bound offline merging.

The sealed evidence manifest uses schema ditserve.p4-pair-evidence.v1 and binds
plan_sha256, runtime_source_sha256, runtime_identity_sha256, and runs containing
pair_id, arm_id, path, sha256, request_id, optional artifact_path. Paths are local
or relative to the manifest. Runs must carry benchmark.p4_quality with the same
plan_sha256/arm_id and formal_protocol.role=p4_quality. Partial manifests retain
the entire frozen plan denominator; they do not create cleanup receipts.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import scipy

import formal_benchmark as formal
from cluster_config import SD35_PATH
from ditserve.evaluation.common import canonical_hash, seal, verify_seal
from dataclasses import dataclass
from native_exact import percentile, validate_pair_identity, validate_run
from native_quality import fold_count, ssim
from p2_benefit import expanded
from p4_quality_plan import SCHEMA as PLAN_SCHEMA, quality_arms, shape_ids, validate_rule


@dataclass(frozen=True)
class _FileIdentity:
    size: int
    sha256: str


class offline_lpips:
    __file__ = __file__
    SCHEMA = 'ditserve.p4-offline-lpips.v1'
    PACKAGES = {'torch': '2.10.0+cu126', 'torchvision': '0.25.0+cu126', 'lpips': '0.1.4'}
    DEFAULT_BACKBONE = '/shared/data/DITSERVE/models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
    BACKBONE = _FileIdentity(244408911, '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02')
    HEAD = _FileIdentity(6009, 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0')
    SOURCES = {
        ('lpips', 'lpips/__init__.py'): _FileIdentity(4846, '36ee004a45e2cc2c5ab47fa3955f00c0e41f6a151ef0249fb6f362533ed58b22'),
        ('lpips', 'lpips/lpips.py'): _FileIdentity(10101, '780d09b907cb9b661e0ae28b2d163ddfba92f9e870d7feba34d4790cc6590658'),
        ('lpips', 'lpips/pretrained_networks.py'): _FileIdentity(6507, '6a27f714c51796db466e86bebba6a617c1bfc4d566f3a1756497629c1248686e'),
        ('torchvision', 'torchvision/models/alexnet.py'): _FileIdentity(4488, '76f0592d51fad133931c234d9168344c52edb89e46861ff15cdef7af7572ab9f'),
    }
    canonical_hash = staticmethod(canonical_hash)

    class OfflineLPIPS:
        def __init__(self, backbone_path=None, head_path=None):
            raise ValueError('offline LPIPS checkpoint not loaded in standalone evaluation')

SCHEMA = 'ditserve.p4-stream-quality.v2'
EVIDENCE_SCHEMA = 'ditserve.p4-pair-evidence.v1'
CHUNK_BYTES = 256 * 1024
INVALID_EVIDENCE = (ValueError, KeyError, TypeError, OSError, AttributeError, IndexError, EOFError)
METRIC_ERRORS = (ValueError, ImportError, OSError, RuntimeError, AttributeError, TypeError, KeyError)
LPIPS_TRANSFORM = 'RGB float32 [0,1] -> NCHW float32 [-1,1] by 2*x-1; no resize/crop; frame arithmetic mean'
LPIPS_PREPROCESSING = 'RGB HWC float, explicit range -> NCHW float32 [-1,1]; no resize/crop'
# Deterministic 64x64 RGB pair from the source-pinned adapter's real selftest.
LPIPS_SELFTEST_INPUT_SHA256 = '5d617b3e5ccbc641f574f73343b8140914a1d0745f1eef1badff84f772f611e5'


def lpips_receipt_identity(receipt):
    """Check the adapter's own seal and exact checkpoint/runtime contract."""
    body = {key: value for key, value in receipt.items() if key != 'sha256'}
    if receipt.get('sha256') != offline_lpips.canonical_hash(body):
        raise ValueError('offline LPIPS selftest receipt hash differs')
    if (receipt.get('schema') != offline_lpips.SCHEMA or receipt.get('status') != 'ready'
            or receipt.get('metric') != 'lpips-alex-v0.1' or receipt.get('device') != 'cpu'
            or receipt.get('preprocessing') != LPIPS_PREPROCESSING
            or receipt.get('adapter_sha256') != file_hash(Path(offline_lpips.__file__))
            or receipt.get('p4_qualification_eligible') is not False
            or receipt.get('retention_dependencies_closed') is not False):
        raise ValueError('offline LPIPS adapter is not verified and ready')
    test = receipt['selftest']
    if (test.get('status') != 'passed_real_cpu_weights'
            or type(test.get('exact_distance')) not in (int, float) or test['exact_distance'] != 0.
            or type(test.get('network_attempts')) is not int or test['network_attempts'] != 0
            or test.get('input_sha256') != LPIPS_SELFTEST_INPUT_SHA256):
        raise ValueError('real CPU LPIPS selftest is missing')
    for key in ('perturbed_distance', 'symmetric_distance', 'independent_reference_distance'):
        value = test.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('LPIPS selftest needs positive finite perturbed distances')
    if (not math.isclose(test['perturbed_distance'], test['symmetric_distance'], rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(test['perturbed_distance'], test['independent_reference_distance'], rel_tol=1e-5, abs_tol=1e-7)):
        raise ValueError('LPIPS selftest reference/symmetry differs')
    weights = {}
    for key, expected in (('backbone', offline_lpips.BACKBONE), ('calibrated_head', offline_lpips.HEAD)):
        value = receipt['weights'][key]
        if value.get('sha256') != expected.sha256 or value.get('bytes') != expected.size:
            raise ValueError('LPIPS checkpoint identity differs')
        weights[key] = dict(bytes=expected.size, sha256=expected.sha256)
    runtime = receipt['runtime']
    if any(runtime['packages'].get(key) != value for key, value in offline_lpips.PACKAGES.items()):
        raise ValueError('LPIPS package identity differs')
    sources = {}
    for (_, member), expected in offline_lpips.SOURCES.items():
        value = runtime['sources'][member]
        if value.get('sha256') != expected.sha256 or value.get('bytes') != expected.size:
            raise ValueError('LPIPS package source identity differs')
        sources[member] = dict(bytes=expected.size, sha256=expected.sha256)
    # Paths and tiny selftest rounding differences are receipt provenance, not
    # alternative metric definitions. Every batch still carries its full receipt.
    return dict(metric=receipt['metric'], adapter_sha256=receipt['adapter_sha256'], device='cpu',
        packages=runtime['packages'], python=runtime['python'], sources=sources, weights=weights,
        unit='dimensionless perceptual distance; lower is better', transform=LPIPS_TRANSFORM)


def prepare_lpips(enabled=False, *, backbone=None, head=None):
    if type(enabled) is not bool or not enabled and (backbone is not None or head is not None):
        raise ValueError('LPIPS needs an explicit Boolean opt-in before weight paths')
    state = dict(requested=enabled, status='unknown', receipt=None, metric_identity=None,
                 reason='offline LPIPS was not requested')
    if not enabled:
        return None, state
    try:
        adapter = offline_lpips.OfflineLPIPS(backbone or offline_lpips.DEFAULT_BACKBONE, head)
        receipt = adapter.selftest()
        identity = lpips_receipt_identity(receipt)
        if adapter.receipt() != receipt:
            raise ValueError('LPIPS adapter changed after selftest')
        state.update(status='ready', receipt=receipt, metric_identity=identity, reason=None)
        return adapter, state
    except METRIC_ERRORS as error:
        state['reason'] = f'{type(error).__name__}: {error}'
        return None, state


def metric_identity(lpips):
    return dict(implementation=dict(module_sha256=file_hash(Path(__file__)),
        ssim_module_sha256=file_hash(Path(ssim.__code__.co_filename)),
        matrix_expansion_module_sha256=file_hash(Path(expanded.__code__.co_filename)),
        offline_lpips_module_sha256=file_hash(Path(offline_lpips.__file__)),
        numpy=np.__version__, scipy=scipy.__version__),
        pixels=dict(range=[0., 1.], layout='[1,T,H,W,3]', dtype='float32',
                    mse='all pixels', psnr='10*log10(1/MSE); infinity encoded as null plus flag',
                    ssim='per-frame/channel Gaussian sigma1.5 truncate3 mean',
                    temporal='RMSE of consecutive reference-relative error frames'),
        lpips=dict(requested=lpips['requested'], status=lpips['status'], identity=lpips['metric_identity']))


def source_binding(plan, evidence, metrics):
    return dict(plan_sha256=plan['sha256'], protocol_sha256=plan['protocol_sha256'],
        core_sha256=next(iter(plan['calibration_sources'].values()))['core_sha256'],
        runtime_source_sha256=evidence['runtime_source_sha256'],
        runtime_identity_sha256=evidence['runtime_identity_sha256'],
        model_mapping_sha256=canonical_hash(plan['model_manifests']),
        calibration_bundle_sha256=plan['calibration_bundle_sha256'], gpu_uuids=plan['gpu_uuids'],
        metric_identity_sha256=canonical_hash(metrics))


def source_entry(entry):
    return None if entry is None else dict(run_sha256=entry['sha256'], request_id=entry['request_id'])


def output_receipt(row):
    output = row['outputs'][0]
    artifact = output['output']
    return dict(artifact_sha256=artifact['sha256'], array_sha256=artifact['array_sha256'],
        shape=artifact['shape'], dtype=artifact['dtype'], generation_sha256=canonical_hash(output['generation']),
        path_history_sha256=canonical_hash(output['path_history']))


def hex_hash(value):
    return isinstance(value, str) and formal.HEX64.fullmatch(value) is not None


def fingerprint(path):
    stat = path.stat()
    if not path.is_file():
        raise ValueError('evidence must be a regular file')
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def local_path(value, root):
    if not isinstance(value, str) or not value or '://' in value:
        raise ValueError('an explicit local evidence path is required')
    path = Path(value)
    return path if path.is_absolute() else root / path


def bound_json(path):
    before = fingerprint(path)
    if before[2] > 32 * 1024 ** 2:
        raise ValueError('JSON exceeds bounded 32 MiB input size')
    raw = path.read_bytes()
    if fingerprint(path) != before:
        raise ValueError('JSON changed during reading')
    def reject_constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, parse_constant=reject_constant), hashlib.sha256(raw).hexdigest(), before


def validate_plan(plan, protocol):
    verify_seal(plan)
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('protocol_sha256') != canonical_hash(protocol)
            or plan.get('split') != 'test' or plan.get('generation_seeds') != [301, 302, 303]):
        raise ValueError('sealed P4 plan/protocol/test seed identity differs')
    validate_rule(plan['rule'])
    ids = shape_ids(protocol, plan['model'])
    if plan['shape_ids'] != ids or plan['arms'] != quality_arms(plan['rule'], plan['form']):
        raise ValueError('P4 shapes or exact/fixed/adaptive rules changed')
    if plan['rule_sha256'] != canonical_hash(plan['rule']):
        raise ValueError('fold rule hash differs')
    count = max(32, protocol['quality']['paired_prompts_per_model_minimum'])
    prompts = plan['prompts']
    if (len(prompts) != count or plan.get('prompts_per_model') != count
            or plan.get('prompts_per_shape') != count):
        raise ValueError('P4 must retain every required prompt cluster')
    hashes, sample_ids = [], []
    for row in prompts:
        digest = canonical_hash(' '.join(row['prompt'].split()))
        if not row['prompt'].strip() or row['normalized_prompt_sha256'] != digest:
            raise ValueError('prompt cluster identity differs')
        hashes.append(digest); sample_ids.append(str(row['dataset_sample_id']))
    if len(set(hashes)) != count or len(set(sample_ids)) != count:
        raise ValueError('duplicate prompt cluster or dataset identity')
    expected = []
    for sid in ids:
        for index, prompt in enumerate(prompts):
            for seed in plan['generation_seeds']:
                expected.append(dict(id=f'{sid}-p{index:03d}-s{seed}', shape_id=sid,
                    dataset_sample_id=prompt['dataset_sample_id'], prompt_index=index,
                    generation_seed=seed, spec=dict(formal.shape_spec(protocol, sid), prompt=prompt['prompt'], seed=seed)))
    if (plan['pairs'] != expected or plan['expected']['pair_count'] != len(expected)
            or plan['expected']['generated_artifact_count'] != len(expected) * len(plan['arms'])):
        raise ValueError('complete prompt x generation-seed x shape denominator differs')
    if (plan['model_mapping_canonical_sha256'] != canonical_hash(plan['model_manifests'])
            or plan['acceptance']['numeric_thresholds'] != plan['rule']['numeric_thresholds']):
        raise ValueError('model or threshold binding differs')
    if not plan['gpu_uuids'] or len(set(plan['gpu_uuids'])) != len(plan['gpu_uuids']):
        raise ValueError('invalid allocation identity')
    form = re.fullmatch(r'cfg2(?:sp([2-4]))?', plan['form'])
    if (form is None or len(plan['gpu_uuids']) not in (2, 4, 6, 8)
            or 2 * int(form.group(1) or 1) > len(plan['gpu_uuids'])):
        raise ValueError('quality form does not fit the native allocation')
    if (set(plan['calibration_sources']) != set(ids)
            or any(not hex_hash(plan.get(key)) for key in ('calibration_bundle_sha256', 'parent_pool_sha256', 'test_pool_sha256'))):
        raise ValueError('missing exact calibration/prompt source bindings')
    for source in plan['calibration_sources'].values():
        if (source.get('gpu_uuids') != plan['gpu_uuids']
                or any(not hex_hash(source.get(key)) for key in ('core_sha256', 'run_sha256'))):
            raise ValueError('calibration source hashes or GPU placement differ')
    if len({source['core_sha256'] for source in plan['calibration_sources'].values()}) != 1:
        raise ValueError('calibration core changes between shapes')
    declared = dict(workers={'video': dict(backend='wan', config={'model_path': formal.MODELS[plan['model']]['path']}),
                             'image': dict(backend='sd35', config={'model_path': SD35_PATH})}, benchmark={})
    formal.bind_manifests(declared, plan['model'], plan['model_manifests'])
    if (plan['worker_model_bindings'] != {key: value['config'] for key, value in declared['workers'].items()}
            or plan['model'] == 'wan22' and (not plan['model_manifests']['wan22'].get('model_path')
                or declared['workers']['video']['config']['model_path'] == formal.MODELS['wan22']['path'])):
        raise ValueError('official model mapping and actual worker bindings differ')
    stats = plan['statistics']
    if (stats.get('resampling_unit') != 'prompt cluster; keep all three seeds and shapes attached'
            or stats.get('confidence') != .95 or type(stats.get('bootstrap_seed')) is not int
            or type(stats.get('bootstrap_samples')) is not int or not 100 <= stats['bootstrap_samples'] <= 100000):
        raise ValueError('freeze bounded prompt-cluster bootstrap settings')
    return plan


def index_evidence(evidence, plan):
    verify_seal(evidence)
    if evidence.get('schema') != EVIDENCE_SCHEMA or evidence.get('plan_sha256') != plan['sha256']:
        raise ValueError('evidence does not bind this sealed plan')
    for key in ('runtime_source_sha256', 'runtime_identity_sha256'):
        if not hex_hash(evidence.get(key)):
            raise ValueError('evidence requires frozen ' + key)
    planned = {(pair['id'], arm['id']) for pair in plan['pairs'] for arm in plan['arms']}
    result = {}
    for entry in evidence.get('runs', []):
        key = entry.get('pair_id'), entry.get('arm_id')
        if key not in planned or key in result:
            raise ValueError('duplicate or unplanned pair/arm evidence')
        if not hex_hash(entry.get('sha256')) or not isinstance(entry.get('request_id'), str):
            raise ValueError('run entry needs raw file SHA256 and request identity')
        result[key] = entry
    return result


def merge_patch(target, patch):
    """RFC 7396, matching nlohmann::json::merge_patch in execute_cells."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def effective_workers(record, phase_id, observed_cell, arm, plan, pair, request_id):
    """Resolve matrix -> worker_config -> video_worker_options in native order."""
    phases = [phase for phase in record['config'].get('phases', []) if phase.get('id') == phase_id]
    if len(phases) != 1:
        raise ValueError('missing unique executed phase configuration')
    cells = expanded(phases[0])
    index = observed_cell.get('cell')
    if type(index) is not int or not 0 <= index < len(cells):
        raise ValueError('observed cell index is outside expanded matrix')
    cell = cells[index]
    if (cell.get('id') != phase_id or cell.get('kind') != 'cluster' or cell.get('policy') != arm['policy']
            or 'actions' in cell or phases[0].get('paired', False)):
        raise ValueError('quality execution requires one unambiguous native cluster policy cell')
    # Native GPU cells currently do not emit a separate configuration. Reject
    # contradictory extra metadata rather than silently ignoring a claimed override.
    for key in ('config', 'worker_config', 'video_worker_options', 'options'):
        if key in observed_cell:
            expected = cell if key == 'config' else cell.get(key)
            if observed_cell[key] != expected:
                raise ValueError('observed cell configuration contradicts expanded phase: ' + key)
    if 'gpus' in observed_cell and observed_cell['gpus'] != len(plan['gpu_uuids']):
        raise ValueError('observed cell GPU count differs')
    options = cell.get('options', {})
    if (options.get('forms') != [plan['form']] or options.get('sp_transport') != 'export'
            or options.get('tiered_memory', False)):
        raise ValueError('effective form/transport differs from isolated quality controls')
    workload = [row for row in cell.get('workload', []) if row.get('request_id') == request_id]
    if (len(workload) != 1 or workload[0].get('kind') != 'video'
            or workload[0].get('dataset_sample_id') != pair['dataset_sample_id']
            or {key: value for key, value in workload[0].get('spec', {}).items() if key != 'request_id'} != pair['spec']):
        raise ValueError('expanded cell workload differs from planned prompt/seed/shape')
    definitions = copy.deepcopy(record['config'].get('workers', {}))
    if 'worker_config' in cell:
        name = cell.get('worker', 'model')
        patch = cell['worker_config']
        if name not in definitions or not isinstance(patch, dict) or set(patch) & {'device', 'gpu_uuid', 'device_id', 'backend', 'work_dir'}:
            raise ValueError('ambiguous or forbidden cell worker_config override')
        definitions[name]['config'] = merge_patch(definitions[name].get('config'), patch)
    slots = cell.get('gpus', [])
    names = [slot.get('video') for slot in slots]
    if (not names or len(set(names)) != len(names) or any(name not in definitions for name in names)
            or [definitions[name].get('gpu_uuid') for name in names] != plan['gpu_uuids']):
        raise ValueError('effective slot/rank GPU order differs')
    if 'video_worker_options' in cell:
        patch = cell['video_worker_options']
        if not isinstance(patch, dict):
            raise ValueError('video_worker_options must be an object')
        for name in names:
            config = definitions[name].get('config')
            if not isinstance(config, dict):
                config = {}
                definitions[name]['config'] = config
            config['options'] = merge_patch(config.get('options'), patch)
    workers = [definitions[name] for name in names]
    if any(worker.get('backend') != 'wan' for worker in workers):
        raise ValueError('quality slots do not use Wan workers')
    return workers


def load_row(entry, pair, arm, plan, evidence, root):
    path = local_path(entry['path'], root)
    record, digest, _ = bound_json(path)
    if digest != entry['sha256']:
        raise ValueError('raw run hash differs')
    benchmark = validate_run(record)
    source = record['source']
    if (canonical_hash(source) != evidence['runtime_source_sha256']
            or source['core_sha256'] != plan['calibration_sources'][pair['shape_id']]['core_sha256']
            or not source.get('files') or any(not hex_hash(value) for value in source['files'].values())):
        raise ValueError('runtime core/Python source identity differs or is missing')
    binding = benchmark.get('formal_protocol', {})
    if (binding.get('sha256') != plan['protocol_sha256'] or binding.get('split') != 'test'
            or binding.get('role') != 'p4_quality' or pair['shape_id'] not in binding.get('shape_ids', [])
            or benchmark.get('p4_quality') != {'plan_sha256': plan['sha256'], 'arm_id': arm['id']}
            or benchmark.get('model_manifests') != plan['model_manifests']):
        raise ValueError('executed protocol/plan/arm/model binding differs')
    sessions = record.get('budget', {}).get('sessions', [])
    if not sessions or any(session.get('status') != 'completed'
            or session.get('identity', {}).get('gpu_uuids') != plan['gpu_uuids'] for session in sessions):
        raise ValueError('completed allocation evidence differs or is missing')
    cells = [(phase.get('id', 'cluster'), cell) for phase in record.get('phases', [])
             for cell in phase.get('cells', []) if cell.get('policy') == arm['policy']]
    if len(cells) != 1 or cells[0][1].get('status') != 'completed':
        raise ValueError('missing unique completed quality-policy cell')
    phase, cell = cells[0]
    workers = effective_workers(record, phase, cell, arm, plan, pair, entry['request_id'])
    expected_binding = plan['worker_model_bindings']['video']
    for worker in workers:
        config = worker.get('config', {})
        if any(config.get(key) != value for key, value in expected_binding.items()):
            raise ValueError('effective worker model snapshot/manifest differs')
        if config.get('options', {}).get('cfg_fold') != arm['cfg_fold']:
            raise ValueError('effective CFG-fold rule differs')
    rows = [row for row in record['requests'] if row.get('request_id') == entry['request_id']
            and row.get('phase', 'cluster') == phase and row.get('cell') == cell['cell'] and row.get('kind') == 'video']
    if len(rows) != 1:
        raise ValueError('missing or duplicate planned request')
    row = rows[0]
    if (row.get('status') != 'completed' or row.get('dataset_sample_id') != pair['dataset_sample_id']
            or {key: value for key, value in row.get('spec', {}).items() if key != 'request_id'} != pair['spec']
            or row.get('forms') != [plan['form']]):
        raise ValueError('completed prompt/seed/shape/form identity differs')
    if len(row.get('outputs', [])) != 1:
        raise ValueError('quality request needs exactly one decoded output')
    generation = row['outputs'][0]['generation']
    if (generation.get('model_integrity') != 'manifest_verified'
            or generation.get('model_manifest_sha256') != plan['model_manifests'][plan['model']]['sha256']
            or generation.get('model_path') != expected_binding['model_path']
            or canonical_hash(generation.get('runtime_identity')) != evidence['runtime_identity_sha256']):
        raise ValueError('produced model/runtime/approximation identity differs')
    count = fold_count(row, arm['expected_actual_fold_count'])
    history = row['outputs'][0]['path_history']
    if (generation.get('approximate') is not bool(count)
            or any(item.get('reused_forwards', 0) != 0 for item in history)):
        raise ValueError('actual approximation includes unplanned reuse or differs from folded history')
    indices = [item['step_index'] for item in history if item.get('cfg_folded')]
    if arm['id'] != 'exact':
        if (count > plan['rule']['max_folded_steps'] or not set(indices) <= set(arm['cfg_fold']['folded_steps'])
                or row['cfg_fold'].get('saved_uncond_forwards') != count):
            raise ValueError('actual folded steps/forward savings violate frozen budget/window')
        if arm['id'].startswith('fixed-') and indices != arm['cfg_fold']['folded_steps']:
            raise ValueError('fixed-N actual schedule differs')
    return row, count


@contextmanager
def checked_array(row, pair, entry, root):
    """No full-array copies: hash file bytes and a contiguous memmap in chunks."""
    artifact = row['outputs'][0]['output']
    if not all(hex_hash(artifact.get(key)) for key in ('sha256', 'array_sha256')):
        raise ValueError('decoded artifact needs file and array SHA256')
    path = local_path(entry.get('artifact_path', artifact.get('path')), root)
    before = fingerprint(path)
    spec = pair['spec']
    shape = [1, spec['frames'], spec['height'], spec['width'], 3]
    if before[2] > math.prod(shape) * 4 + 4096:
        raise ValueError('NPY file exceeds planned float32 payload and header bound')
    if file_hash(path) != artifact['sha256'] or fingerprint(path) != before:
        raise ValueError('decoded file hash differs or changed')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if (not isinstance(array, np.memmap) or list(array.shape) != shape or artifact.get('shape') != shape
                or str(array.dtype) != 'float32' or artifact.get('dtype') != 'float32'
                or not array.flags.c_contiguous):
            raise ValueError('decoded array must match planned C-contiguous float32 shape')
        digest = hashlib.sha256()
        view = memoryview(array).cast('B')
        for start in range(0, len(view), CHUNK_BYTES):
            digest.update(view[start:start + CHUNK_BYTES])
        view.release()
        if digest.hexdigest() != artifact['array_sha256']:
            raise ValueError('decoded array hash differs')
        yield array
        if fingerprint(path) != before:
            raise ValueError('decoded artifact changed during evaluation')
    finally:
        if isinstance(array, np.memmap):
            array._mmap.close()
        elif hasattr(array, 'close'):
            array.close()


def frame_metrics(reference, candidate, *, lpips_adapter=None, lpips_reason=None):
    """Global pixel MSE/PSNR and framewise SSIM; O(one frame) working arrays."""
    if (reference.shape != candidate.shape or reference.ndim != 5 or reference.shape[0] != 1
            or reference.shape[-1] != 3 or min(reference.shape) < 1
            or not np.issubdtype(reference.dtype, np.floating) or not np.issubdtype(candidate.dtype, np.floating)):
        raise ValueError('matching nonempty floating [1,T,H,W,3] arrays are required')
    error = temporal = score = lpips_total = 0.
    lpips_failure = lpips_reason or 'verified offline LPIPS adapter unavailable'
    lpips_available = lpips_adapter is not None
    previous = None
    count = reference.shape[1]
    for index in range(count):
        ref, other = reference[0, index], candidate[0, index]
        for frame in (ref, other):
            if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
                raise ValueError('decoded values must be finite and in [0,1]; no clipping')
        delta = other.astype(np.float64) - ref.astype(np.float64)
        error += float(np.sum(delta * delta))
        if previous is not None:
            change = delta - previous
            temporal += float(np.sum(change * change))
        previous = delta
        score += ssim(ref[None, None], other[None, None])
        if lpips_available:
            try:
                distance = lpips_adapter.score_frame(ref, other, input_range='zero_one')
                if type(distance) not in (float, int) or not math.isfinite(distance) or distance < 0:
                    raise ValueError('LPIPS frame distance must be finite and nonnegative')
                lpips_total += distance
            except METRIC_ERRORS as failure:
                lpips_available = False
                lpips_failure = f'frame {index}: {type(failure).__name__}: {failure}'
    mse = error / reference.size
    return dict(mse=mse, psnr_db=None if mse == 0 else -10 * math.log10(mse),
        psnr_is_infinite=mse == 0, exact_match=mse == 0, ssim=score / count,
        temporal_delta_rmse=None if count == 1 else math.sqrt(temporal / ((count - 1) * reference[0, 0].size)),
        lpips_mean=lpips_total / count if lpips_available else None,
        lpips_status='measured' if lpips_available else 'unknown',
        lpips_frames=count if lpips_available else 0,
        lpips_reason=None if lpips_available else lpips_failure)


def cluster_bootstrap(rows, *, samples, seed):
    """Resample prompts with all their observed seeds/shapes, never individual pairs."""
    if type(samples) is not int or not 100 <= samples <= 100000 or type(seed) is not int:
        raise ValueError('invalid bounded bootstrap settings')
    clusters = defaultdict(list)
    for row in rows:
        if not math.isfinite(row['difference']):
            raise ValueError('cluster bootstrap requires finite differences; use MSE for infinite PSNR')
        clusters[row['prompt_sha256']].append(row['difference'])
    ordered = [clusters[key] for key in sorted(clusters)]
    pairs = sum(map(len, ordered))
    result = dict(method='paired prompt-cluster bootstrap, retaining all seed/shape members',
        independent_prompt_clusters=len(ordered), observed_pairs=pairs, resamples=samples,
        mean_difference=sum(map(sum, ordered)) / pairs if pairs else None, ci95=None,
        status='unknown' if len(ordered) < 2 else 'measured',
        scope='available paired observations; full planned denominator and failures are reported separately')
    if len(ordered) >= 2:
        totals = np.asarray([sum(values) for values in ordered], dtype=np.float64)
        counts = np.asarray([len(values) for values in ordered], dtype=np.int64)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(samples):
            draw = rng.integers(0, len(ordered), size=len(ordered))
            draws.append(float(totals[draw].sum() / counts[draw].sum()))
        result['ci95'] = [percentile(draws, .025), percentile(draws, .975)]
    return result


def matched_comparisons(plan, results):
    stats = plan['statistics']
    rows = []
    relations = Counter()
    missing = []
    for pair in plan['pairs']:
        adaptive = results[(pair['id'], 'adaptive')]
        if adaptive['status'] != 'measured':
            missing.append(pair['id']); continue
        count = adaptive['actual_fold_count']
        control = results[(pair['id'], 'exact' if count == 0 else 'fixed-' + str(count))]
        if control['status'] != 'measured':
            missing.append(pair['id']); continue
        a, b = adaptive['metrics'], control['metrics']
        relation = ('both_exact' if a['exact_match'] and b['exact_match'] else
                    'candidate_exact_control_finite' if a['exact_match'] else
                    'candidate_finite_control_exact' if b['exact_match'] else 'both_finite')
        relations[relation] += 1
        rows.append(dict(pair_id=pair['id'], shape_id=pair['shape_id'], actual_fold_count=count,
            prompt_sha256=plan['prompts'][pair['prompt_index']]['normalized_prompt_sha256'],
            mse_difference=a['mse'] - b['mse'], ssim_difference=a['ssim'] - b['ssim'],
            lpips_difference=(a['lpips_mean'] - b['lpips_mean'] if
                a['lpips_status'] == b['lpips_status'] == 'measured' else None),
            zero_fold_exact_self_control=count == 0 and a['exact_match'] and b['exact_match']))
    groups = {'all': rows}
    for sid in plan['shape_ids']:
        groups['shape:' + sid] = [row for row in rows if row['shape_id'] == sid]
        for count in range(plan['rule']['max_folded_steps'] + 1):
            groups[f'shape:{sid}/fold:{count}'] = [row for row in rows
                if row['shape_id'] == sid and row['actual_fold_count'] == count]
    for count in range(plan['rule']['max_folded_steps'] + 1):
        groups['fold:' + str(count)] = [row for row in rows if row['actual_fold_count'] == count]
    summaries = {}
    for name, selected in groups.items():
        summaries[name] = {metric: cluster_bootstrap(
            [dict(prompt_sha256=row['prompt_sha256'], difference=row[metric + '_difference'])
             for row in selected if row[metric + '_difference'] is not None],
            samples=stats['bootstrap_samples'], seed=stats['bootstrap_seed']) for metric in ('mse', 'ssim', 'lpips')}
    return dict(status='complete' if not missing else 'incomplete', expected_pairs=len(plan['pairs']),
        matched_pairs=len(rows), missing_pair_ids=missing, groups=summaries,
        psnr=dict(status='exact_relations_and_mse_domain', relations=dict(relations), mean_difference=None, ci95=None,
                  scope='never subtract infinity; MSE candidate-minus-control is finite and lower favors candidate'),
        zero_fold_exact_self_controls=sum(row['zero_fold_exact_self_control'] for row in rows),
        zero_fold_scope='no unconditional-forward reduction; not adaptive superiority',
        lpips=dict(status='measured' if rows and all(r['lpips_difference'] is not None for r in rows) else 'unknown',
            measured_matched_pairs=sum(r['lpips_difference'] is not None for r in rows),
            expected_pairs=len(plan['pairs']), unit='candidate-minus-control perceptual distance; lower favors candidate'),
        superiority_qualified=False)


VBENCH_SCHEMA = 'ditserve.benchmark.vbench_evaluation.v1'
VBENCH_REVISION = '1039fe66d7aa2f4b214003ca2d35f5011c48bfee'


def validate_vbench_receipt(receipt):
    if receipt is None:
        return dict(status='unknown', reason='no pinned complete official VBench evidence consumed')
    if not isinstance(receipt, dict) or not verify_seal(receipt):
        raise ValueError('unsealed or tampered VBench evaluation receipt')
    if receipt.get('schema_version') != VBENCH_SCHEMA:
        raise ValueError('unexpected VBench evaluation receipt schema')
    if receipt.get('revision') != VBENCH_REVISION:
        raise ValueError('VBench evaluation receipt revision does not match pinned revision')
    if receipt.get('status') != 'completed' or receipt.get('verified') is not True:
        raise ValueError('VBench evaluation receipt is not verified/completed')
    dimensions = receipt.get('dimensions')
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError('VBench evaluation receipt has no measured dimensions')
    for name, entry in dimensions.items():
        if not isinstance(entry, dict) or type(entry.get('score')) not in (int, float) or not math.isfinite(entry['score']):
            raise ValueError(f'invalid VBench dimension score: {name}')
    return dict(
        status='measured',
        receipt_sha256=receipt['sha256'],
        manifest_sha256=receipt.get('manifest_sha256'),
        revision=receipt['revision'],
        dimensions=copy.deepcopy(dimensions),
        reason=None,
    )


def evaluate(plan, protocol, evidence, *, root=Path('.'), lpips=False, lpips_backbone=None, lpips_head=None, vbench_receipt=None):
    validate_plan(plan, protocol)
    vbench_state = validate_vbench_receipt(vbench_receipt)
    index = index_evidence(evidence, plan)
    lpips_adapter, lpips_state = prepare_lpips(lpips, backbone=lpips_backbone, head=lpips_head)
    metrics_identity = metric_identity(lpips_state)
    arms = {arm['id']: arm for arm in plan['arms']}
    results = {}
    for pair in plan['pairs']:
        reference_key = pair['id'], 'exact'
        reference_entry = index.get(reference_key)
        for arm_id in arms:
            results[(pair['id'], arm_id)] = dict(pair_id=pair['id'], arm_id=arm_id,
                shape_id=pair['shape_id'], status='missing', reason='no submitted run for planned pair/arm',
                prompt_sha256=plan['prompts'][pair['prompt_index']]['normalized_prompt_sha256'],
                generation_seed=pair['generation_seed'], spec_sha256=canonical_hash(pair['spec']),
                submitted_source=source_entry(index.get((pair['id'], arm_id))),
                reference_source=source_entry(reference_entry), evidence_sha256=evidence['sha256'])
        if reference_entry is None:
            for arm_id in arms:
                if (pair['id'], arm_id) in index:
                    results[(pair['id'], arm_id)].update(status='blocked', reason='exact reference missing')
            continue
        try:
            reference, _ = load_row(reference_entry, pair, arms['exact'], plan, evidence, root)
            identity = validate_pair_identity(reference, reference)
            if not all(identity.values()):
                raise ValueError('reference identity/weights unverified')
            with checked_array(reference, pair, reference_entry, root) as ref:
                exact_metrics = frame_metrics(ref, ref, lpips_adapter=lpips_adapter, lpips_reason=lpips_state['reason'])
                results[reference_key].update(status='measured', actual_fold_count=0, metrics=exact_metrics,
                    run_sha256=reference_entry['sha256'], artifact_sha256=reference['outputs'][0]['output']['sha256'],
                    verified_output=output_receipt(reference), reference_output=output_receipt(reference))
                for arm_id, arm in arms.items():
                    if arm_id == 'exact' or (pair['id'], arm_id) not in index:
                        continue
                    target = results[(pair['id'], arm_id)]
                    entry = index[(pair['id'], arm_id)]
                    try:
                        candidate, count = load_row(entry, pair, arm, plan, evidence, root)
                        identity = validate_pair_identity(reference, candidate, allow_approximate=True)
                        if not all(identity.values()):
                            raise ValueError('paired runtime/weight identity unverified')
                        with checked_array(candidate, pair, entry, root) as other:
                            metrics = frame_metrics(ref, other, lpips_adapter=lpips_adapter, lpips_reason=lpips_state['reason'])
                        target.update(status='measured', actual_fold_count=count, metrics=metrics,
                            run_sha256=entry['sha256'], artifact_sha256=candidate['outputs'][0]['output']['sha256'],
                            verified_output=output_receipt(candidate), reference_output=output_receipt(reference))
                    except INVALID_EVIDENCE as error:
                        target.update(status='failed', reason=str(error))
        except INVALID_EVIDENCE as error:
            # A changed/corrupt reference invalidates even already evaluated arms.
            for arm_id in arms:
                if (pair['id'], arm_id) in index:
                    results[(pair['id'], arm_id)].update(status='failed', reason='exact reference: ' + str(error))
    for row in results.values():
        if row['status'] != 'measured':
            for key in ('metrics', 'actual_fold_count', 'verified_output', 'reference_output', 'run_sha256', 'artifact_sha256'):
                row.pop(key, None)
    result = summarize_results(plan, results, lpips_state, vbench_state=vbench_state)
    result.update(schema=SCHEMA, plan_sha256=plan['sha256'], evidence_sha256=evidence['sha256'],
        protocol_sha256=plan['protocol_sha256'], evidence=copy.deepcopy(evidence),
        metric_identity=metrics_identity, source_binding=source_binding(plan, evidence, metrics_identity),
        implementation=metrics_identity['implementation'])
    return seal(result)


def summarize_results(plan, results, lpips_state, vbench_state=None):
    """Recompute summaries from full-denominator observations, also after merging."""
    arms = {arm['id']: arm for arm in plan['arms']}
    counts = Counter(row['status'] for row in results.values())
    thresholds = plan['acceptance']['numeric_thresholds']
    by_arm = {}
    for arm_id in arms:
        rows = [results[(pair['id'], arm_id)] for pair in plan['pairs']]
        measured = [row for row in rows if row['status'] == 'measured']
        failed_numeric = sum((not row['metrics']['psnr_is_infinite'] and row['metrics']['psnr_db'] < thresholds['min_psnr'])
                             or row['metrics']['ssim'] < thresholds['min_ssim'] for row in measured)
        known_lpips = [row for row in measured if row['metrics']['lpips_status'] == 'measured']
        failed_lpips = sum(row['metrics']['lpips_mean'] > thresholds['max_lpips'] for row in known_lpips)
        numeric = ('failed' if failed_numeric or failed_lpips else
                   'passed_numeric_only' if len(known_lpips) == len(rows) else 'unknown')
        by_arm[arm_id] = dict(expected_pairs=len(rows), measured_pairs=len(measured),
            missing_or_failed_pairs=len(rows) - len(measured), psnr_ssim_threshold_failures=failed_numeric,
            lpips_measured_pairs=len(known_lpips), lpips_unknown_or_missing_pairs=len(rows) - len(known_lpips),
            lpips_threshold_failures=failed_lpips,
            lpips_mean=sum(row['metrics']['lpips_mean'] for row in known_lpips) / len(known_lpips) if known_lpips else None,
            numeric_qualification=numeric,
            unknown_reason='full planned evidence and LPIPS required; semantic/P4 acceptance remains separate',
            mse_mean=sum(row['metrics']['mse'] for row in measured) / len(measured) if measured else None,
            exact_matches=sum(row['metrics']['exact_match'] for row in measured))
    lpips_count = sum(row['status'] == 'measured' and row['metrics']['lpips_status'] == 'measured' for row in results.values())
    return dict(status='complete' if counts['measured'] == len(results) else 'incomplete',
        denominator=dict(planned_pairs=len(plan['pairs']), planned_pair_arms=len(results),
            submitted_pair_arms=sum(row['submitted_source'] is not None for row in results.values()), statuses=dict(counts),
            scope='missing, failed and blocked arms remain in the original planned denominator'),
        by_arm=by_arm, pair_results=list(results.values()), matched_fold_comparison=matched_comparisons(plan, results),
        lpips=dict(status='measured' if lpips_count == len(results) else 'partial' if lpips_count else 'unknown',
            measured_pair_arms=lpips_count, expected_pair_arms=len(results),
            unknown_or_missing_pair_arms=len(results) - lpips_count,
            unit='dimensionless perceptual distance; lower is better', transform=LPIPS_TRANSFORM,
            adapter=copy.deepcopy(lpips_state)),
        semantic_qualification='measured' if (vbench_state and vbench_state.get('status') == 'measured') else 'unknown',
        vbench=copy.deepcopy(vbench_state) if vbench_state is not None else dict(
            status='unknown', reason='no pinned complete official VBench evidence consumed'),
        qualification_eligible=False, superiority_qualified=False,
        resource_scope=dict(artifact_maps_at_once=2, file_hash_chunk_bytes=CHUNK_BYTES,
            metric_working_set='frame-sized float64/SSIM arrays and optional CPU LPIPS activations; no full-video copies',
            lpips_model_scope='one persistent CPU model; startup/selftest also loads the full AlexNet checkpoint and reference model',
            memory_limit='RSS/allocator peaks are unmeasured; memmap OS page cache is not a hard RSS cap',
            artifact_bytes_written=0, raw_artifacts_deleted=0, gpu_operations=0),
        retention=dict(dependencies_closed=False, deletion_authorized=False,
            reason='measurement/merge does not close raw consumers; semantic/replay and stage acceptance remain open'),
        limitations=['No runtime configuration generator or GPU runner is implemented here.',
            'Input runs require explicit p4_quality plan/arm binding; legacy unbound runs cannot be relabelled.',
            'Bounded evaluation memory and receipt merging do not reclaim disk space or authorize cleanup.',
            'Bootstrap intervals describe available matched pairs; incomplete evidence does not acquire qualification.',
            'Semantic quality, fission/reuse causality and full stage acceptance remain unverified.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=formal.DEFAULT_PROTOCOL)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lpips', action='store_true', help='require real offline CPU selftest before LPIPS scores')
    parser.add_argument('--lpips-backbone', type=Path)
    parser.add_argument('--lpips-head', type=Path)
    parser.add_argument('--vbench-receipt', type=Path)
    args = parser.parse_args()
    inputs = {name: bound_json(getattr(args, name)) for name in ('plan', 'evidence', 'protocol')}
    if args.vbench_receipt is not None:
        inputs['vbench_receipt'] = bound_json(args.vbench_receipt)
    if (args.lpips_backbone or args.lpips_head) and not args.lpips:
        parser.error('explicit LPIPS weight paths require --lpips')
    result = evaluate(inputs['plan'][0], inputs['protocol'][0], inputs['evidence'][0], root=args.evidence.parent,
                      lpips=args.lpips, lpips_backbone=args.lpips_backbone, lpips_head=args.lpips_head,
                      vbench_receipt=inputs['vbench_receipt'][0] if 'vbench_receipt' in inputs else None)
    if any(fingerprint(getattr(args, name)) != value[2] for name, value in inputs.items()):
        raise ValueError('input plan/evidence/protocol changed during evaluation')
    result['input_file_sha256'] = {name: value[1] for name, value in inputs.items()}
    formal.write_new(args.output, seal(result))
    print(json.dumps(dict(status=result['status'], qualification_eligible=False, output=str(args.output))))


if __name__ == '__main__':
    main()
