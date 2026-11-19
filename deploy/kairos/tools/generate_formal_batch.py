#!/usr/bin/env python3
"""Prepare immutable formal specs; never publish or submit them.

The digest-pinned image reference still requires validation against the current
Kairos user documentation before submission. No fallback to a mutable tag.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1] / 'ditserve'
IMAGE_DIGEST = 'sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05'
IMAGE = '192.168.1.114:30500/jjjkairos/ditserve@' + IMAGE_DIGEST


def prepare(commit, protocol, stage, *, date, gpus=8, model_manifests=None, model=None, p0_case=None,
            protocol_relpath='configs/benchmarks/formal-mixed-v1.json'):
    short = commit[:7]
    operator_root = f'/shared/data/DITSERVE/scripts/releases/{short}'
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if stage != 'contract':
        path = PurePosixPath(model_manifests or '')
        if '..' in path.parts or not path.is_relative_to('/shared/data/DITSERVE'):
            raise ValueError('provide an absolute frozen model-manifest mapping under /shared/data/DITSERVE')
    result = []
    if model not in (None, 'wan21', 'wan22') or p0_case not in (None, 'contention', 'no_preemption'):
        raise ValueError('unknown model or P0 case')
    if p0_case and stage != 'p0' or model and stage == 'contract':
        raise ValueError('model/case selection does not apply to this stage')
    models = [None] if stage == 'contract' else [model] if model else ['wan21', 'wan22']
    for selected_model in models:
        case_tag = '' if not p0_case else '-stop' if p0_case == 'contention' else '-idle'
        name = f'd{date}-formal-{stage}-{selected_model or "runtime"}{case_tag}-{short}-r1'
        if len(name) > 52:
            raise ValueError('job name exceeds platform limit')
        env = {'DITSERVE_BUILD': short, 'DITSERVE_RUN_ID': name,
               'DITSERVE_KAIROS_RUNTIME_ROOT': operator_root,
               'DITSERVE_TEST_REVISION': commit,
               'DITSERVE_FORMAL_PROTOCOL': f'/shared/code/DITSERVE/builds/{short}/share/ditserve/{protocol_relpath}',
               'DITSERVE_PROTOCOL_SHA256': protocol_hash,
               'DITSERVE_IMAGE_DIGEST': IMAGE_DIGEST, 'DITSERVE_EXECUTION_IMAGE': IMAGE,
               'DITSERVE_JOB_SPEC_PATH': f'/shared/data/DITSERVE/job-specs/{date}-formal-{short}/{name}.json'}
        if selected_model:
            env.update(DITSERVE_BENCHMARK_MODE='formal', DITSERVE_FORMAL_ACTION=stage,
                       DITSERVE_STAGE_ACTION=('no_preemption' if p0_case == 'no_preemption' else 'preemption') if stage == 'p0' else 'shape_calibration',
                       WAN_MODEL='wan' if selected_model == 'wan21' else 'wan2.2-ti2v-5b',
                       DITSERVE_MODEL_MANIFESTS=model_manifests)
            if p0_case:
                env['DITSERVE_P0_CASE'] = p0_case
        else:
            env['DITSERVE_TEST_ROOT'] = f'/shared/data/DITSERVE/tests/{short}/tests/python'
        result.append({'name': name, 'image': IMAGE,
                       'command': 'bash ' + operator_root + '/scripts/' + (
                           'kairos_formal_benchmark.sh' if selected_model else 'kairos_targeted_tests.sh'),
                       'workers': 1, 'gpus_per_worker': gpus if selected_model else 1, 'mount_shared': True,
                       'env': [{'name': k, 'value': v} for k, v in env.items()]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit', nargs='?', default='HEAD')
    parser.add_argument('--stage', choices=['contract', 'p0', 'calibration'], default='contract')
    parser.add_argument('--gpus', type=int, choices=[2, 4, 6, 8], default=8)
    parser.add_argument('--date', default=datetime.now().strftime('%Y%m%d'))
    parser.add_argument('--model-manifests')
    parser.add_argument('--model', choices=['wan21', 'wan22'])
    parser.add_argument('--p0-case', choices=['contention', 'no_preemption'])
    args = parser.parse_args()
    datetime.strptime(args.date, '%Y%m%d')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--verify', args.commit+'^{commit}'], text=True).strip()
    # Generate from the committed protocol that will actually be deployed.
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show', commit+':configs/benchmarks/formal-mixed-v1.json'])
    specs = prepare(commit, json.loads(raw), args.stage, date=args.date, gpus=args.gpus,
                    model_manifests=args.model_manifests, model=args.model, p0_case=args.p0_case)
    destination = ROOT / 'job-specs' / f'{args.date}-formal-{commit[:7]}'
    destination.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        path = destination / (spec['name']+'.json')
        content = json.dumps(spec, indent=2)+'\n'
        if path.exists():
            if path.read_text() != content:
                raise ValueError('existing spec differs; use a new identity: '+str(path))
        else:
            path.write_text(content)
    print(json.dumps({'specs': len(specs), 'directory': str(destination), 'submitted': False,
                      'platform_digest_reference': 'pending_documentation_validation'}))


if __name__ == '__main__':
    main()
