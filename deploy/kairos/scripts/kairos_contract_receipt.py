#!/usr/bin/env python3
"""Publish CONTRACT eligibility only after completed, hash-bound test evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from stage_gates import canonical_hash, evaluate


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def finalize(run_dir, protocol, job_spec):
    root = Path(run_dir).resolve()
    info = json.loads((root / 'BUILD_INFO.json').read_bytes())
    smoke = json.loads((root / 'runtime-smoke.json').read_bytes())
    spec = json.loads(Path(job_spec).read_bytes())
    spec_ref = reference(job_spec)
    if (smoke.get('image_identity_source') != 'submitted_job_spec_digest_pin'
            or smoke.get('image_job_spec_sha256') != spec_ref['sha256']
            or spec.get('name') != smoke.get('run_id')
            or spec.get('image') != smoke.get('execution_image')
            or not spec.get('image', '').endswith('@' + smoke.get('image_digest', 'missing'))):
        raise ValueError('contract requires the exact digest-pinned submitted job spec')
    environment = {item['name']: item['value'] for item in spec.get('env', [])}
    protocol_hash = canonical_hash(json.loads(Path(protocol).read_bytes()))
    if environment.get('DITSERVE_PROTOCOL_SHA256') != protocol_hash:
        raise ValueError('submitted job protocol differs from frozen protocol')
    if (spec.get('workers') != 1 or spec.get('gpus_per_worker') != 1
            or spec.get('mount_shared') is not True
            or environment.get('DITSERVE_TEST_REVISION') != info['commit']):
        raise ValueError('contract requires one GPU and tests from the same full commit')
    manifest = {'schema': 'ditserve.stage-evidence.v1',
        'build': {key: info[key] for key in ('commit', 'core_sha256')},
        'protocol': reference(protocol),
        'contract': {'runtime_smoke': reference(root / 'runtime-smoke.json'),
                     'log': reference(root / 'contract.log'), 'job_spec': spec_ref},
        'runs': []}
    manifest['build']['image_digest'] = smoke.get('image_digest')
    result = evaluate(manifest, root)
    if result['stages']['CONTRACT']['status'] != 'passed':
        raise ValueError('full contract did not qualify: ' + result['stages']['CONTRACT'].get('reason', 'unknown'))
    # Do not create any passed receipt until full evidence evaluation succeeds.
    manifest_path = root / 'contract-evidence.json'
    receipt_path = root / 'contract-receipt.json'
    if manifest_path.exists() or receipt_path.exists():
        raise FileExistsError('contract evidence already exists; use a new run ID')
    with manifest_path.open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    result['source_manifest'] = reference(manifest_path)
    with receipt_path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'status': 'passed', 'stage': 'CONTRACT', 'receipt': str(receipt_path),
            'image_identity_scope': 'submitted image digest pin; not a pod imageID measurement'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--job-spec', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_dir, args.protocol, args.job_spec)), flush=True)


if __name__ == '__main__':
    main()
