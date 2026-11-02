#!/usr/bin/env python3
"""Prepare a P0-gated, same-allocation two-GPU isolated P1 diagnostic.

No submission, installation, download or qualification occurs here. ``spec``
recomputes an explicit P0 receipt before writing a spec. Inside that future job,
``preflight`` binds the deployed core/Python, input files and actual allocation;
``prepare`` accepts both complete formal calibration records from that job.
Inputs are copied as small JSON snapshots so offline review needs no cluster
absolute paths. Original calibration sources remain protected dependencies.
"""

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

import formal_benchmark as formal
import p1_isolated_batch as isolated
from cluster_config import allocated_uuids
from p1_benchmark import validate_plan
from stage_gates import read_json, source_file, verify_receipt
from ditserve.evaluation.common import canonical_hash, seal, verify_seal

SCHEMA = 'ditserve.p1-isolated-job.v1'
PREFLIGHT = 'ditserve.p1-isolated-preflight.v1'
JSON_LIMIT = 128 * 1024 ** 2


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def byte_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_json(path):
    path = Path(path)
    before = path.stat()
    if not path.is_file() or before.st_size > JSON_LIMIT:
        raise ValueError('expected bounded JSON source: ' + str(path))
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('JSON source changed during read: ' + str(path))
    return read_json(raw), raw


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(256 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def env_values(spec):
    entries = spec['env']
    values = {entry['name']: entry['value'] for entry in entries}
    if len(values) != len(entries) or any(not isinstance(value, str) for value in values.values()):
        raise ValueError('duplicate/non-string job environment')
    return values


def p0_model_mapping(receipt_path, receipt, model):
    """Use the verified P0 records' mapping, never a newly relabelled mapping."""
    base = Path(receipt_path).resolve().parent
    manifest = read_json(source_file(base, receipt['source_manifest']))
    base = (base / receipt['source_manifest']['path']).resolve().parent
    mappings, roles = [], set()
    for item in manifest['runs']:
        if item['model'] != model:
            continue
        record = read_json(source_file(base, {key: item[key] for key in ('path', 'sha256')}))
        mapping = record['config']['benchmark']['model_manifests']
        if not all(key in mapping for key in (model, 'sd35')):
            raise ValueError('P0 lacks the selected video and resident image model mapping')
        mappings.append(mapping); roles.add(item['role'])
    if roles != {'preemption', 'no_preemption'} or len({canonical_hash(v) for v in mappings}) != 1:
        raise ValueError('P0 roles lack a single frozen full model mapping')
    return mappings[0]


def prepare_spec(base_spec, protocol, plan, receipt_path, *, model, date, operator_root,
                 operator_script, benchmark_root):
    """Reuse the formal generator's shape-calibration spec; never overwrite it."""
    validate_plan(plan, protocol)
    datetime.strptime(date, '%Y%m%d')
    env = env_values(base_spec)
    commit = env['DITSERVE_BUILD']
    if (model not in ('wan21', 'wan22') or not re.fullmatch('[0-9a-f]{7,40}', commit)
            or base_spec.get('workers') != 1 or base_spec.get('gpus_per_worker') != 2
            or base_spec.get('mount_shared') is not True
            or env.get('DITSERVE_FORMAL_ACTION') != 'calibration'
            or env.get('DITSERVE_STAGE_ACTION') != 'shape_calibration'
            or env.get('WAN_MODEL') != ('wan' if model == 'wan21' else 'wan2.2-ti2v-5b')
            or env.get('DITSERVE_PROTOCOL_SHA256') != canonical_hash(protocol)):
        raise ValueError('base must be the matching immutable two-GPU formal calibration spec')
    root = PurePosixPath(operator_root)
    if ('..' in root.parts or str(root) != operator_root
            or not root.is_relative_to('/shared/data/DITSERVE/scripts/releases')
            or root == PurePosixPath('/shared/data/DITSERVE/scripts/releases')
            or operator_root == env.get('DITSERVE_KAIROS_RUNTIME_ROOT')):
        raise ValueError('provide a new immutable DITSERVE operator release')
    # Missing/blocked/stale receipts raise before any output is created.
    receipt, raw = checked_json(receipt_path)
    fresh = verify_receipt(receipt_path, build_commit=commit,
        protocol_sha256=canonical_hash(protocol), required_stage='P0', action='p1_diagnostics')
    if checked_json(receipt_path)[1] != raw:
        raise ValueError('P0 receipt changed while validating')
    if (fresh['stages']['P0']['status'] != 'passed' or 'p1_diagnostics' not in fresh['stages']['P0']['allows']
            or not base_spec['image'].endswith('@' + fresh['build']['image_digest'])
            or env.get('DITSERVE_TEST_REVISION') != fresh['build']['commit']):
        raise ValueError('P0/build/image permission differs')
    # Keep the explicit formal-job namespace accepted by the storage guard.
    name = f'd{date}-formal-p1iso-{model}-{commit[:7]}-r1'
    result = copy.deepcopy(base_spec)
    result.update(name=name, command='bash ' + operator_root + '/scripts/kairos_formal_benchmark.sh')
    env.update(DITSERVE_RUN_ID=name, DITSERVE_FORMAL_ACTION='p1_isolated', DITSERVE_STAGE_ACTION='p1_diagnostics',
        DITSERVE_BENCHMARK_MODE='formal',
        DITSERVE_KAIROS_RUNTIME_ROOT=operator_root,
        DITSERVE_BENCHMARK_ROOT=str(root / 'benchmarks'),
        DITSERVE_JOB_SPEC_PATH=f'/shared/data/DITSERVE/job-specs/{date}-p1iso-{commit[:7]}/{name}.json',
        DITSERVE_P1_PLAN_SHA256=canonical_hash(plan), DITSERVE_P0_GATE_SHA256=byte_hash(raw),
        DITSERVE_MODEL_MANIFESTS_SHA256=canonical_hash(p0_model_mapping(receipt_path, receipt, model)),
        DITSERVE_EXPECTED_CORE_SHA256=fresh['build']['core_sha256'],
        DITSERVE_EXPECTED_PYTHON_SHA256=fresh['stages']['CONTRACT']['python_sources_sha256'],
        DITSERVE_P1_HELPER_SHA256=file_hash(__file__),
        DITSERVE_P1_OPERATOR_SHA256=file_hash(operator_script),
        DITSERVE_EXPECTED_BENCHMARKS_SHA256=canonical_hash(benchmark_identity(benchmark_root)['sources']))
    result['env'] = [dict(name=key, value=value) for key, value in env.items()]
    return result


def inventory(package):
    result = {}
    for name in ('ditserve', 'xfuser'):
        paths = sorted((Path(package) / name).rglob('*.py'))
        if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
            raise ValueError('missing or linked installed Python package: ' + name)
        result.update({path.relative_to(package).as_posix(): file_hash(path) for path in paths})
    return result


def benchmark_inventory(root):
    root = Path(root)
    paths = sorted(root.rglob('*.py'))
    if (not paths or any(path.is_symlink() or not path.is_file() for path in paths)
            or not (root / 'p1_isolated_batch.py').is_file()):
        raise ValueError('missing or linked deployed benchmark sources')
    return {path.relative_to(root).as_posix(): file_hash(path) for path in paths}


def benchmark_identity(root):
    """Bind disk inventory AND already imported benchmark modules to one root.

    A new CLI process must set PYTHONPATH to this explicit snapshot followed
    by the old installed worker package. Inventory alone cannot detect an old
    compiler/module left in Python's import cache.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('missing or linked explicit benchmark snapshot')
    root = root.resolve()
    sources = benchmark_inventory(root)
    imported = {}
    required = {formal.__name__, isolated.__name__, validate_plan.__module__,
                read_json.__module__, allocated_uuids.__module__}
    modules = {name.removesuffix('.py').replace('/', '.'): name for name in sources}
    for name in sorted(required | (set(modules) & set(sys.modules))):
        if name == __name__:  # This helper belongs to operator/tools, not benchmarks.
            continue
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        relative = modules.get(name)
        if not path or relative is None or Path(path).resolve() != root / relative:
            raise ValueError('benchmark import is outside the explicit snapshot: ' + name)
        imported[name] = relative
    return dict(root=str(root), sources=sources, imports=imported)


def operator_identity(script, benchmark_root):
    benchmarks = benchmark_identity(benchmark_root)
    return dict(helper_sha256=file_hash(__file__), operator_sha256=file_hash(script),
        operator_path=str(Path(script).resolve()),
        benchmark_root=benchmarks['root'], benchmark_sources=benchmarks['sources'],
        benchmark_imports=benchmarks['imports'])


def preflight(*, model, sources, package, core, operator_script, benchmark_root, gpu_uuids, environment):
    """Bind actual installed files and allocation before calibration starts."""
    documents, references = {}, {}
    for key, path in sources.items():
        documents[key], raw = checked_json(path)
        references[key] = dict(path=str(Path(path).resolve()), sha256=byte_hash(raw))
    protocol, plan, build, spec = (documents[key] for key in ('protocol', 'pair_plan', 'build', 'job_spec'))
    validate_plan(plan, protocol)
    env = env_values(spec)
    expected = ('DITSERVE_BUILD', 'DITSERVE_RUN_ID', 'DITSERVE_PROTOCOL_SHA256', 'DITSERVE_P1_PLAN_SHA256',
        'DITSERVE_FORMAL_ACTION', 'DITSERVE_STAGE_ACTION', 'DITSERVE_P0_GATE_SHA256', 'DITSERVE_BENCHMARK_MODE',
        'DITSERVE_EXPECTED_CORE_SHA256', 'DITSERVE_EXPECTED_PYTHON_SHA256',
        'DITSERVE_P1_HELPER_SHA256', 'DITSERVE_P1_OPERATOR_SHA256', 'DITSERVE_EXPECTED_BENCHMARKS_SHA256',
        'DITSERVE_KAIROS_RUNTIME_ROOT', 'DITSERVE_BENCHMARK_ROOT', 'DITSERVE_TEST_REVISION', 'DITSERVE_IMAGE_DIGEST', 'WAN_MODEL',
        'DITSERVE_MODEL_MANIFESTS', 'DITSERVE_MODEL_MANIFESTS_SHA256', 'DITSERVE_JOB_SPEC_PATH')
    if any(environment.get(key) != env.get(key) for key in expected):
        raise ValueError('execution environment differs from submitted spec')
    if (len(gpu_uuids) != 2 or len(set(gpu_uuids)) != 2 or any(not isinstance(v, str) or not v for v in gpu_uuids)
            or spec.get('workers') != 1 or spec.get('gpus_per_worker') != 2 or spec.get('mount_shared') is not True
            or spec.get('name') != env.get('DITSERVE_RUN_ID')
            or env.get('DITSERVE_BENCHMARK_MODE') != 'formal'
            or env.get('DITSERVE_FORMAL_ACTION') != 'p1_isolated' or env.get('DITSERVE_STAGE_ACTION') != 'p1_diagnostics'
            or not formal.HEX64.fullmatch(env.get('DITSERVE_P0_GATE_SHA256', ''))
            or env.get('WAN_MODEL') != ('wan' if model == 'wan21' else 'wan2.2-ti2v-5b')
            or env.get('DITSERVE_PROTOCOL_SHA256') != canonical_hash(protocol)
            or env.get('DITSERVE_P1_PLAN_SHA256') != canonical_hash(plan)
            or env.get('DITSERVE_MODEL_MANIFESTS_SHA256') != canonical_hash(documents['model_manifests'])
            or build.get('commit') != env.get('DITSERVE_TEST_REVISION')
            or not build.get('commit', '').startswith(env.get('DITSERVE_BUILD', ''))
            or not spec.get('image', '').endswith('@' + env.get('DITSERVE_IMAGE_DIGEST', 'MISSING'))):
        raise ValueError('isolated P1 requires its frozen P0-gated two-GPU job identity')
    core_digest, python_sources = file_hash(core), inventory(Path(package))
    if (core_digest != build.get('core_sha256') or core_digest != env['DITSERVE_EXPECTED_CORE_SHA256']
            or canonical_hash(python_sources) != env['DITSERVE_EXPECTED_PYTHON_SHA256']):
        raise ValueError('installed core/Python differs from verified P0 runtime')
    operator = operator_identity(operator_script, benchmark_root)
    expected_script = Path(env['DITSERVE_KAIROS_RUNTIME_ROOT']) / 'scripts/kairos_formal_benchmark.sh'
    if (operator['helper_sha256'] != env['DITSERVE_P1_HELPER_SHA256']
            or operator['operator_sha256'] != env['DITSERVE_P1_OPERATOR_SHA256']
            or canonical_hash(operator['benchmark_sources']) != env['DITSERVE_EXPECTED_BENCHMARKS_SHA256']
            or env.get('DITSERVE_BENCHMARK_ROOT') != str(Path(env['DITSERVE_KAIROS_RUNTIME_ROOT']) / 'benchmarks')
            or Path(operator['benchmark_root']) != Path(env['DITSERVE_BENCHMARK_ROOT']).resolve()
            or Path(operator['operator_path']) != expected_script.resolve()
            or spec.get('command') != 'bash ' + str(expected_script)
            or Path(sources['job_spec']).resolve() != Path(env['DITSERVE_JOB_SPEC_PATH']).resolve()
            or Path(sources['model_manifests']).resolve() != Path(env['DITSERVE_MODEL_MANIFESTS']).resolve()):
        raise ValueError('separately frozen operator/helper/benchmark sources or input paths differ')
    pairs = [pair for pair in plan['shape_pairs'] if pair['model'] == model]
    videos = [key for key, value in protocol['shapes'].items()
              if value['model'] == model and value['kind'] == 'video' and value['scope'] == 'formal']
    images = [key for key, value in protocol['shapes'].items()
              if value['model'] == 'sd35' and value['kind'] == 'image' and value['scope'] == 'formal']
    if [(pair['video_shape'], pair['image_shape']) for pair in pairs] != list(zip(videos, images)):
        raise ValueError('operator calibration order differs from the frozen shape-pair plan')
    return seal(dict(schema=PREFLIGHT, model=model, run_id=spec['name'], commit=build['commit'],
        core_sha256=core_digest, python_sources=python_sources, gpu_uuids=gpu_uuids,
        operator=operator,
        planned_cases=[dict(pair_id=pair['id'], expected_requests=96) for pair in pairs],
        expected_requests=192, qualification_eligible=False,
        limitation='planned denominator only, including cases never reached after calibration failure',
        p0_receipt_sha256=env['DITSERVE_P0_GATE_SHA256'], sources=references,
        scope='actual deployment/allocation binding; P0 permission was recomputed before spec creation and must be rechecked by the submitting queue'))


def prepare_inputs(preflight_path, *, bundle_path, runtime_facts_path, calibration_paths,
                   package, core, benchmark_root, gpu_uuids, output_dir):
    """Validate everything before writing two reviewable, relative-path bundles."""
    checked, checked_raw = checked_json(preflight_path)
    verify_seal(checked)
    if (checked.get('schema') != PREFLIGHT or checked['gpu_uuids'] != gpu_uuids or len(gpu_uuids) != 2
            or file_hash(core) != checked['core_sha256'] or inventory(Path(package)) != checked['python_sources']
            or operator_identity(checked['operator']['operator_path'], benchmark_root) != checked['operator']):
        raise ValueError('allocation or deployed runtime changed since preflight')
    docs, blobs = {}, {'preflight.json': checked_raw}
    for key, source in checked['sources'].items():
        value, raw = checked_json(source['path'])
        if byte_hash(raw) != source['sha256']:
            raise ValueError('preflight input changed: ' + key)
        docs[key], blobs[key + '.json'] = value, raw
    for key, path in (('bundle', bundle_path), ('runtime_facts', runtime_facts_path)):
        docs[key], blobs[key + '.json'] = checked_json(path)
    p, plan, model = docs['protocol'], docs['pair_plan'], checked['model']
    pairs = [pair for pair in plan['shape_pairs'] if pair['model'] == model]
    if len(pairs) != 2 or len(calibration_paths) != 2:
        raise ValueError('both formal shape-pair calibrations are required')
    calibrations, calibration_sources = [], []
    for index, path in enumerate(calibration_paths):
        record, raw = checked_json(path)
        if (record.get('run_id') != checked['run_id'] + '-cal' + str(index)
                or record.get('config', {}).get('benchmark', {}).get('formal_protocol', {}).get('shape_ids')
                != [pairs[index]['video_shape'], pairs[index]['image_shape']]):
            raise ValueError('calibration is not this job\'s complete ordered formal shape pair')
        calibrations.append(record); blobs[f'calibration-{index}.json'] = raw
        calibration_sources.append(dict(path=str(Path(path).resolve()), sha256=byte_hash(raw)))
    runtimes = {canonical_hash(row['outputs'][-1]['generation']['runtime_identity'])
                for record in calibrations for row in record.get('requests', []) if row.get('outputs')}
    if len(runtimes) != 1:
        raise ValueError('calibrations lack a single observed frozen runtime identity')
    identity = dict(schema='ditserve.p1-isolated-identity.v1', core_sha256=checked['core_sha256'],
        python_sources=checked['python_sources'], runtime_identity_sha256=next(iter(runtimes)),
        model_manifests_sha256=canonical_hash(docs['model_manifests']))
    blobs['identity.json'] = encoded(identity)
    reference = lambda name: dict(path=name, sha256=byte_hash(blobs[name]))
    outputs, cases = {}, []
    for pair in pairs:
        manifest = dict(schema=isolated.SCHEMA, pair_id=pair['id'], gpu_uuids=gpu_uuids,
            **{key: reference(key + '.json') for key in ('protocol', 'pair_plan', 'bundle', 'model_manifests', 'identity', 'runtime_facts')},
            pools={kind: reference('pool_' + kind + '.json') for kind in ('video', 'image')},
            calibrations=[reference(f'calibration-{i}.json') for i in range(2)])
        manifest_raw = encoded(manifest)
        inputs = dict(protocol=p, pair_plan=plan, pair_id=pair['id'], gpu_uuids=gpu_uuids,
            bundle=docs['bundle'], model_manifests=docs['model_manifests'], identity=identity,
            runtime_facts=docs['runtime_facts'], calibrations=calibrations,
            pools={kind: docs['pool_' + kind] for kind in ('video', 'image')},
            inputs_manifest_sha256=byte_hash(manifest_raw))
        config = isolated.compile_case(inputs)  # Includes exact raw reassembly and runtime/form/row validation.
        name = pair['id']
        outputs.update({name + '-inputs.json': manifest_raw, name + '-config.json': encoded(config),
                        name + '-planned-coverage.json': encoded(isolated.coverage(config))})
        cases.append(dict(pair_id=name, inputs=name + '-inputs.json', inputs_sha256=byte_hash(manifest_raw),
            config=name + '-config.json', config_sha256=byte_hash(outputs[name + '-config.json']),
            run_id=checked['run_id'] + '-' + name, expected_requests=96))
    result = seal(dict(schema=SCHEMA, action='p1_isolated', requires_stage='P0', stage_action='p1_diagnostics',
        run_id=checked['run_id'], model=model, gpu_uuids=gpu_uuids, preflight_sha256=byte_hash(checked_raw),
        cases=cases, expected_requests=192, calibration_sources=calibration_sources,
        diagnostic_only=True, qualification_eligible=False, dependencies_closed=False,
        limitation='coverage is not output equivalence, zero-stop validation, timing analysis, or P1 qualification'))
    outputs['execution-plan.json'] = encoded(result)
    # No partial preparations are published for a failed validation. Once writing
    # begins a filesystem failure leaves its exact partial directory for review.
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in {**blobs, **outputs}.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    spec = sub.add_parser('spec')
    for name in ('base_spec', 'protocol', 'pair_plan', 'p0_receipt', 'output', 'operator_script', 'benchmark_root'):
        spec.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    for name in ('model', 'date', 'operator_root'):
        spec.add_argument('--' + name.replace('_', '-'), required=True)
    first = sub.add_parser('preflight')
    first.add_argument('--model', choices=['wan21', 'wan22'], required=True)
    for name in ('protocol', 'pair_plan', 'model_manifests', 'build', 'job_spec', 'datasets', 'output', 'package', 'core', 'operator_script', 'benchmark_root'):
        first.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    prepare = sub.add_parser('prepare')
    for name in ('preflight', 'bundle', 'runtime_facts', 'output_dir', 'package', 'core', 'benchmark_root'):
        prepare.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    prepare.add_argument('--calibrations', nargs='+', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'spec':
        result = prepare_spec(*(checked_json(getattr(args, key))[0] for key in ('base_spec', 'protocol', 'pair_plan')),
            args.p0_receipt, model=args.model, date=args.date, operator_root=args.operator_root,
            operator_script=args.operator_script, benchmark_root=args.benchmark_root)
        formal.write_new(args.output, result)
    elif args.command == 'preflight':
        import os
        protocol = checked_json(args.protocol)[0]
        sources = {key: getattr(args, key) for key in ('protocol', 'pair_plan', 'model_manifests', 'build', 'job_spec')}
        sources.update({'pool_' + kind: args.datasets / (protocol['datasets'][kind]['kind'] + '.json') for kind in ('video', 'image')})
        result = preflight(model=args.model, sources=sources, package=args.package, core=args.core,
            operator_script=args.operator_script, benchmark_root=args.benchmark_root,
            gpu_uuids=allocated_uuids(), environment=os.environ)
        formal.write_new(args.output, result)
    else:
        result = prepare_inputs(args.preflight, bundle_path=args.bundle, runtime_facts_path=args.runtime_facts,
            calibration_paths=args.calibrations, package=args.package, core=args.core,
            benchmark_root=args.benchmark_root, gpu_uuids=allocated_uuids(), output_dir=args.output_dir)
    print(json.dumps(dict(command=args.command, schema=result.get('schema'), submitted=False, qualification_eligible=False)))


if __name__ == '__main__':
    main()
