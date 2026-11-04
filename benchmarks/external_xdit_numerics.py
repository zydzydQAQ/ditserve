"""Opt-in complete decoded-frame capture and preregistered SP1/SP4 comparison.

The space is float32 returned video AFTER pipeline postprocessing, not latents,
pre-clamp VAE output, semantic quality, or a performance benchmark. Run the full
comparison in a Kairos job container, never on its SSH login host.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess

OUTPUT_ROOT = Path('/shared/output/DITSERVE')
PACKAGE_ROOT = Path('/shared/data/DITSERVE/external-probes')
SPACE = 'returned decoded float32 frames after pipeline postprocessing'
ATOL = RTOL = 1e-5
TOTAL_BUDGET = 2 * 1024**3
MEMBER_BUDGET = 768 * 1024**2
CODE_FILES = ('external_xdit.py', 'external_xdit_native.py',
              'external_xdit_distributed.py', 'external_xdit_numerics.py')
IMAGE = 'sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05'


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def check_storage():
    result = subprocess.run(['df', '-BG', '/shared'], capture_output=True, text=True, timeout=20)
    try:
        row = result.stdout.splitlines()[-1].split()
        used, percent = int(row[2].rstrip('G')), int(row[4].rstrip('%'))
        if result.returncode or used > 800 or percent >= 80:
            raise ValueError('shared-storage stop threshold reached')
    except (IndexError, ValueError) as exc:
        raise ValueError('shared-storage gate failed') from exc
    return {'used_gib': used, 'percent': percent}


def safe_path(path, root=OUTPUT_ROOT):
    path, root = Path(path), Path(root)
    if (not path.is_absolute() or '..' in path.parts or path == root
            or not path.is_relative_to(root) or path.resolve() != path):
        raise ValueError('artifact path escapes project output or traverses a symlink')
    return path


def publish(path, value):
    check_storage()
    with Path(path).open('xb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def dimensions(request):
    shape = [request[k] for k in ('num_frames', 'height', 'width')] + [3]
    if any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('positive integer output dimensions required')
    size = math.prod(shape) * 4
    if size + 1024**2 > MEMBER_BUDGET or 2 * (size + 1024**2) > TOTAL_BUDGET:
        raise ValueError('complete decoded output exceeds preregistered storage budget')
    return shape, size


def read_bound(path, expected):
    if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected):
        raise ValueError('SHA256 binding is required')
    path = Path(path)
    if path.resolve() != path or path.is_symlink():
        raise ValueError('bound document traverses a symlink')
    if not path.is_file() or path.stat().st_size > 2*1024**2:
        raise ValueError('bound document must be a small regular file')
    raw = path.read_bytes()
    if sha_bytes(raw) != expected:
        raise ValueError('bound document SHA256 differs')
    return json.loads(raw)


def validate_contract(contract):
    if (contract.get('schema') != 'ditserve.xdit-decoded-contract.v1'
            or contract.get('space') != SPACE
            or type(contract.get('atol')) is not float or contract['atol'] != ATOL
            or type(contract.get('rtol')) is not float or contract['rtol'] != RTOL
            or type(contract.get('max_total_bytes')) is not int or contract['max_total_bytes'] != TOTAL_BUDGET
            or type(contract.get('member_budget_bytes')) is not int or contract['member_budget_bytes'] != MEMBER_BUDGET
            or contract.get('image_digest') != IMAGE or contract.get('build') != '51740c0'):
        raise ValueError('preregistered numerical/storage contract differs')
    dimensions(contract['request'])
    if set(contract.get('members', {})) != {'sp1', 'sp4'}:
        raise ValueError('exact SP1 reference and SP4 candidate members required')
    names = []
    for role, world in [('sp1', 1), ('sp4', 4)]:
        member = contract['members'][role]
        name = member.get('run_id', '')
        if (not re.fullmatch('[a-z0-9][a-z0-9-]{0,51}', name)
                or type(member.get('world_size')) is not int or member['world_size'] != world):
            raise ValueError('invalid run identity or topology')
        expected = OUTPUT_ROOT / 'external-tasks' / name / 'result'
        if member.get('output_dir') != str(expected):
            raise ValueError('output directory must be the exact registered job result')
        safe_path(expected, OUTPUT_ROOT)
        names.append(name)
    if len(set(names)) != 2:
        raise ValueError('two distinct run identities required')
    for key in ('protocol_sha256', 'official_source_manifest_sha256', 'model_manifest_sha256'):
        if not re.fullmatch('[a-f0-9]{64}', contract.get(key, '')):
            raise ValueError('source/model/protocol binding missing')
    if set(contract.get('code_sha256', {})) != set(CODE_FILES):
        raise ValueError('exact adapter source inventory required')
    for name, expected in contract['code_sha256'].items():
        if sha_bytes(Path(__file__).with_name(name).read_bytes()) != expected:
            raise ValueError('actual adapter source differs: ' + name)
    return contract


def prepare(contract_path, expected, output, request, identity, source, protocol_sha256, world, rank):
    contract_path = Path(contract_path)
    contract = validate_contract(read_bound(contract_path, expected))
    role = 'sp1' if world == 1 else 'sp4' if world == 4 else None
    if role is None or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('capture supports SP1 and SP4 only')
    member = contract['members'][role]
    output = Path(output)
    if (str(output) != member['output_dir'] or request != contract['request']
            or identity.get('manifest_sha256') != contract['model_manifest_sha256']
            or source.get('manifest_sha256') != contract['official_source_manifest_sha256']
            or protocol_sha256 != contract['protocol_sha256']):
        raise ValueError('execution differs from preregistered inputs')
    safe_path(output, OUTPUT_ROOT)
    if (os.environ.get('DITSERVE_RUN_ID') != member['run_id']
            or os.environ.get('DITSERVE_BUILD') != contract['build']
            or os.environ.get('DITSERVE_EXTERNAL_IMAGE_DIGEST') != contract['image_digest']):
        raise ValueError('actual launch identity differs')
    package = Path(os.environ.get('DITSERVE_EXTERNAL_PROBE_ROOT', ''))
    if (not package.is_absolute() or package.parent != PACKAGE_ROOT
            or package.resolve() != package or contract_path.parent != package):
        raise ValueError('contract must be inside the frozen task package')
    manifest_sha = os.environ.get('DITSERVE_EXTERNAL_PROBE_MANIFEST_SHA256')
    manifest = read_bound(package / 'package-manifest.json', manifest_sha)
    if (manifest.get('image_digest') != IMAGE or manifest['files'].get(contract_path.name) != expected
            or any(manifest['files'].get(n) != h for n, h in contract['code_sha256'].items())):
        raise ValueError('contract/code are not bound by the launch package')
    return dict(contract=contract, contract_sha256=expected, role=role, rank=rank,
                owner_rank=0, world_size=world, output_dir=str(output),
                run_id=member['run_id'], package_manifest_sha256=manifest_sha,
                request=request, model_identity=identity, source=source,
                protocol_sha256=protocol_sha256)


def begin(context):
    """Protect even a future partial artifact before any model execution."""
    if context['rank'] != context['owner_rank']:
        return
    directory = safe_path(Path(context['output_dir']) / 'decoded-numerics', OUTPUT_ROOT)
    check_storage()
    directory.mkdir(exist_ok=False)
    publish(directory / 'DEPENDENCY-PROTECTED.json', dict(
        schema='ditserve.numerical-dependency-protection.v1', protected=True,
        dependency='pending complete SP1/SP4 decoded numerical acceptance',
        cleanup_allowed=False, automatic_cleanup=False,
        contract_sha256=context['contract_sha256'], run_id=context['run_id'],
        expected_payload_bytes=dimensions(context['request'])[1],
        protected_names=['decoded.float32.raw.partial', 'decoded.float32.raw', 'artifact.json']))


def capture(videos, summary, context):
    """Write exactly one full C-order float32 payload, without affecting inference."""
    import numpy as np
    if context['rank'] != 0:
        raise ValueError('only the selected output owner may retain decoded frames')
    shape, size = dimensions(context['request'])
    if (summary.get('shape') != shape or summary.get('dtype') != 'float32'
            or summary.get('finite') is not True or summary.get('outputs') != 1
            or len(videos) != 1 or len(videos[0]) != shape[0]):
        raise ValueError('complete finite float32 decoded output required')
    directory = safe_path(Path(context['output_dir']) / 'decoded-numerics', OUTPUT_ROOT)
    protection = json.loads((directory / 'DEPENDENCY-PROTECTED.json').read_bytes())
    if (protection.get('protected') is not True or protection.get('cleanup_allowed') is not False
            or protection.get('contract_sha256') != context['contract_sha256']):
        raise ValueError('pending numerical dependency is not protected')
    target, pending = directory / 'decoded.float32.raw', directory / 'decoded.float32.raw.partial'
    if target.exists() or (directory / 'artifact.json').exists():
        raise FileExistsError('previous numerical artifact must be preserved')
    payload_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    output_hash.update(json.dumps({'shape': shape, 'dtype': 'float32'}, sort_keys=True).encode())
    total = 0
    check_storage()
    with pending.open('xb') as stream:
        for frame in videos[0]:
            check_storage()
            array = np.asarray(frame)
            if (list(array.shape) != shape[1:] or array.dtype != np.dtype('<f4')
                    or not np.isfinite(array).all()):
                raise ValueError('frame geometry, float32 dtype or finiteness differs')
            raw = np.ascontiguousarray(array).tobytes()
            total += len(raw)
            if total > size or total + 1024**2 > MEMBER_BUDGET:
                raise ValueError('decoded payload exceeds complete-output budget')
            stream.write(raw)
            payload_hash.update(raw)
            output_hash.update(raw)
        stream.flush()
        os.fsync(stream.fileno())
    check_storage()
    if total != size or pending.stat().st_size != size or output_hash.hexdigest() != summary['sha256']:
        raise ValueError('captured bytes do not reproduce the native output digest')
    pending.rename(target)
    artifact = dict(schema='ditserve.xdit-decoded-artifact.v1', status='complete', space=SPACE,
        contract_sha256=context['contract_sha256'], run_id=context['run_id'], role=context['role'],
        world_size=context['world_size'], owner_rank=0, shape=shape, dtype='<f4', order='C',
        payload_bytes=size, file_name=target.name, file_sha256=payload_hash.hexdigest(),
        native_output_sha256=output_hash.hexdigest(),
        native_output_digest_definition='SHA256(sorted JSON shape/dtype prefix + C-order frame bytes)',
        package_manifest_sha256=context['package_manifest_sha256'], request=context['request'],
        model_identity=context['model_identity'], source=context['source'],
        protocol_sha256=context['protocol_sha256'], protected=True,
        cleanup_allowed=False, qualification_eligible=False, quality_evaluated=False)
    publish(directory / 'artifact.json', artifact)
    return dict(manifest_path=str(directory / 'artifact.json'),
                manifest_sha256=sha_bytes(json_bytes(artifact)), file_sha256=artifact['file_sha256'],
                native_output_sha256=artifact['native_output_sha256'], payload_bytes=size,
                owner_rank=0, contract_sha256=context['contract_sha256'])


def capture_receipt(context):
    return dict(enabled=True, contract_sha256=context['contract_sha256'], role=context['role'],
                owner_rank=0, world_size=context['world_size'],
                artifact_expected_on_this_rank=context['rank'] == 0)



def validate_rank_capture(rows, world):
    """Every rank opts in identically; only rank zero may publish one payload."""
    first = rows[0]
    reference = first.get('decoded_capture', {})
    if world != 4 or reference.get('enabled') is not True or reference.get('role') != 'sp4':
        raise ValueError('decoded capture requires complete SP4 receipts')
    for row in rows:
        expected = dict(reference, artifact_expected_on_this_rank=row['rank'] == 0)
        if (row.get('decoded_capture') != expected or expected.get('world_size') != world
                or type(expected.get('owner_rank')) is not int or expected['owner_rank'] != 0
                or row.get('performance_eligible') is not False):
            raise ValueError('rank capture contract/owner differs')
        if row['rank'] == 0:
            artifact = row.get('decoded_artifact', {})
            if (artifact.get('owner_rank') != 0 or artifact.get('contract_sha256') != reference.get('contract_sha256')
                    or artifact.get('native_output_sha256') != row['output']['sha256']
                    or artifact.get('payload_bytes') != dimensions(row['request'])[1]
                    or row['output'].get('retained_arrays') is not True):
                raise ValueError('selected owner did not retain the complete output')
        elif 'decoded_artifact' in row or row['output'].get('retained_arrays') is not False:
            raise ValueError('nonowner rank retained a duplicate output')


def verify_artifact(contract, contract_sha, role, expected_manifest_sha):
    member = contract['members'][role]
    directory = safe_path(Path(member['output_dir']) / 'decoded-numerics', OUTPUT_ROOT)
    artifact = read_bound(directory / 'artifact.json', expected_manifest_sha)
    shape, size = dimensions(contract['request'])
    expected = dict(schema='ditserve.xdit-decoded-artifact.v1', status='complete', space=SPACE,
        contract_sha256=contract_sha, run_id=member['run_id'], role=role, world_size=member['world_size'],
        owner_rank=0, shape=shape, dtype='<f4', order='C', payload_bytes=size,
        file_name='decoded.float32.raw', request=contract['request'],
        protocol_sha256=contract['protocol_sha256'], protected=True, cleanup_allowed=False,
        qualification_eligible=False, quality_evaluated=False)
    if any(artifact.get(k) != v for k, v in expected.items()):
        raise ValueError('artifact identity, geometry or protection differs')
    if (artifact['source'].get('manifest_sha256') != contract['official_source_manifest_sha256']
            or artifact['model_identity'].get('manifest_sha256') != contract['model_manifest_sha256']):
        raise ValueError('artifact source/model differs')
    target = safe_path(directory / artifact['file_name'], OUTPUT_ROOT)
    if target.stat().st_size != size:
        raise ValueError('raw payload byte count differs')
    protection = json.loads((directory / 'DEPENDENCY-PROTECTED.json').read_bytes())
    if (protection.get('protected') is not True or protection.get('contract_sha256') != contract_sha
            or protection.get('cleanup_allowed') is not False or protection.get('automatic_cleanup') is not False
            or protection.get('run_id') != member['run_id'] or protection.get('expected_payload_bytes') != size):
        raise ValueError('artifact protection differs')
    # Completion of inference and torchrun must be established independently of
    # capture: the capture occurs before the original native cleanup runs.
    outer = json.loads((directory.parent.parent / 'summary.json').read_bytes())
    child = json.loads((directory.parent / 'summary.json').read_bytes())
    execution = outer.get('execution', {})
    if (outer.get('run_id') != member['run_id'] or execution.get('returncode') != 0
            or execution.get('timed_out') is not False or execution.get('child_summary') != child
            or execution.get('child_summary_sha256') != sha_bytes((directory.parent / 'summary.json').read_bytes())
            or outer.get('binding', {}).get('manifest_sha256') != artifact['package_manifest_sha256']
            or outer.get('binding', {}).get('image') != '192.168.1.114:30500/jjjkairos/ditserve@' + IMAGE
            or child.get('completed_requests') != 1 or child.get('requested_requests') != 1
            or child.get('request') != contract['request'] or child.get('source') != artifact['source']
            or child.get('model_identity') != artifact['model_identity']
            or child.get('output', {}).get('sha256') != artifact['native_output_sha256']):
        raise ValueError('completed native outer/child execution binding differs')
    if role == 'sp4':
        import external_xdit_distributed as distributed
        if distributed.aggregate(child['rank_receipts'], 4) != child:
            raise ValueError('complete SP4 rank receipts do not reproduce aggregate')
    elif child.get('status') != 'native_request_completed':
        raise ValueError('SP1 did not complete')
    owner = child['rank_receipts'][0] if role == 'sp4' else child
    saved = owner.get('decoded_artifact', {})
    if saved.get('manifest_sha256') != expected_manifest_sha or saved.get('file_sha256') != artifact['file_sha256']:
        raise ValueError('native result does not bind the captured artifact')
    return artifact, target


def compare_files(reference_path, candidate_path, shape, expected):
    """Full denominator, bounded memory; fp64 accumulation, fixed SP1 reference."""
    import numpy as np
    def fingerprint(path):
        value = Path(path).stat()
        if not stat.S_ISREG(value.st_mode):
            raise ValueError('raw payload must be a regular file')
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    before = [fingerprint(p) for p in (reference_path, candidate_path)]
    count = math.prod(shape)
    if any(x[2] != count*4 for x in before):
        raise ValueError('raw payload byte count differs from complete denominator')
    frame_bytes = math.prod(shape[1:]) * 4
    total, failed, sum_abs, sum_sq, maximum = 0, 0, 0., 0., 0.
    hashes = [hashlib.sha256(), hashlib.sha256()]
    outputs = [hashlib.sha256(), hashlib.sha256()]
    prefix = json.dumps({'shape': shape, 'dtype': 'float32'}, sort_keys=True).encode()
    for h in outputs: h.update(prefix)
    with Path(reference_path).open('rb') as reference, Path(candidate_path).open('rb') as candidate:
        for _ in range(shape[0]):
            check_storage()
            raw = [reference.read(frame_bytes), candidate.read(frame_bytes)]
            if any(len(x) != frame_bytes for x in raw):
                raise ValueError('truncated full-frame raw payload')
            arrays = [np.frombuffer(x, dtype='<f4') for x in raw]
            if any(not np.isfinite(x).all() for x in arrays):
                raise ValueError('nonfinite raw payload')
            for i in range(2):
                hashes[i].update(raw[i]); outputs[i].update(raw[i])
            a, b = (x.astype(np.float64) for x in arrays)
            error = np.abs(a-b)
            failed += int(np.count_nonzero(error > ATOL + RTOL * np.abs(a)))
            total += int(error.size)
            maximum = max(maximum, float(error.max()))
            sum_abs += float(np.sum(error, dtype=np.float64))
            sum_sq += float(np.sum(error * error, dtype=np.float64))
        if reference.read(1) or candidate.read(1) or total != count:
            raise ValueError('raw payload exceeds complete denominator')
    if before != [fingerprint(p) for p in (reference_path, candidate_path)]:
        raise ValueError('raw payload changed during comparison')
    for i in range(2):
        if (hashes[i].hexdigest() != expected[i]['file_sha256']
                or outputs[i].hexdigest() != expected[i]['native_output_sha256']):
            raise ValueError('raw file or native-prefixed SHA256 differs')
    return dict(total_elements=total, failed_elements=failed, max_abs=maximum,
                mean_abs=sum_abs/total, rmse=math.sqrt(sum_sq/total),
                strict_decoded_consistency_passed=failed == 0,
                payload_file_sha256=[h.hexdigest() for h in hashes],
                native_output_sha256=[h.hexdigest() for h in outputs])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--sp1-manifest-sha256', required=True)
    parser.add_argument('--sp4-manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    check_storage()
    contract = validate_contract(read_bound(args.contract, args.contract_sha256))
    output = safe_path(args.output_dir, OUTPUT_ROOT)
    if any(output.is_relative_to(Path(m['output_dir'])) or Path(m['output_dir']).is_relative_to(output)
           for m in contract['members'].values()):
        raise ValueError('comparison output must be disjoint from protected captures')
    def receipt_hashes():
        return {role: {str(p): sha_bytes(p.read_bytes()) for p in (
            Path(member['output_dir']) / 'summary.json', Path(member['output_dir']).parent / 'summary.json',
            Path(member['output_dir']) / 'decoded-numerics' / 'DEPENDENCY-PROTECTED.json')}
            for role, member in contract['members'].items()}
    input_receipts = receipt_hashes()
    artifacts = [verify_artifact(contract, args.contract_sha256, role, sha)
                 for role, sha in [('sp1', args.sp1_manifest_sha256), ('sp4', args.sp4_manifest_sha256)]]
    if sum(a['payload_bytes'] for a, _ in artifacts) + 2*1024**2 > TOTAL_BUDGET:
        raise ValueError('pair exceeds preregistered total budget')
    metrics = compare_files(artifacts[0][1], artifacts[1][1], dimensions(contract['request'])[0],
                            [a for a, _ in artifacts])
    if receipt_hashes() != input_receipts:
        raise ValueError('execution/protection receipts changed during comparison')
    result = dict(schema='ditserve.xdit-decoded-comparison.v1', status='comparison_completed',
        space=SPACE, reference='sp1', candidate='sp4', atol=ATOL, rtol=RTOL,
        formula='abs(candidate-reference) <= atol + rtol*abs(SP1 reference)',
        contract_sha256=args.contract_sha256, request=contract['request'],
        shape=dimensions(contract['request'])[0], dtype='float32',
        run_ids={role: member['run_id'] for role, member in contract['members'].items()},
        source_bindings={key: contract[key] for key in ('build', 'image_digest', 'code_sha256',
            'protocol_sha256', 'official_source_manifest_sha256', 'model_manifest_sha256')},
        payload_bytes=sum(a['payload_bytes'] for a, _ in artifacts), max_total_bytes=TOTAL_BUDGET,
        artifact_manifest_sha256=[args.sp1_manifest_sha256, args.sp4_manifest_sha256],
        input_receipts=input_receipts,
        **metrics, qualification_eligible=False, semantic_quality_evaluated=False,
        performance_eligible=False, latent_comparison=False, protected_inputs_retained=True,
        platform_succeeded_must_be_checked_independently=True)
    check_storage(); output.mkdir(parents=True, exist_ok=False)
    publish(output / 'summary.json', result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    # A measured numerical difference is a completed diagnostic, not a tool fault.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
