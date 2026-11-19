#!/usr/bin/env python3
"""Freeze P1 jobs from committed files; requires a passed P0 gate to submit."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from generate_formal_batch import REPO, ROOT, prepare


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare_p1(commit, protocol, plan, *, date, gpus, model_manifests, pair_id=None, role=None):
    if plan.get('protocol_sha256') != canonical_hash(protocol) or plan.get('schema') != 'ditserve.p1-plan.v1':
        raise ValueError('P1 plan and committed formal protocol differ')
    pairs = [item for item in plan['shape_pairs'] if pair_id is None or item['id'] == pair_id]
    if not pairs or role is not None and role not in plan['roles']:
        raise ValueError('unknown P1 pair/role')
    result = []
    for pair in pairs:
        for selected_role in ([role] if role else plan['roles']):
            spec = prepare(commit, protocol, 'calibration', date=date, gpus=gpus,
                           model_manifests=model_manifests, model=pair['model'])[0]
            tag = 'stop' if selected_role == 'contention' else 'idle'
            name = f'd{date}-formal-p1-{pair["id"]}-{tag}-{commit[:7]}-r1'
            if len(name) > 52:
                raise ValueError('P1 job name exceeds platform limit')
            values = {item['name']: item['value'] for item in spec['env']}
            values.update(DITSERVE_RUN_ID=name, DITSERVE_FORMAL_ACTION='p1',
                DITSERVE_STAGE_ACTION='p1_diagnostics', DITSERVE_P1_PAIR=pair['id'],
                DITSERVE_P1_ROLE=selected_role, DITSERVE_P1_PLAN_SHA256=canonical_hash(plan),
                DITSERVE_JOB_SPEC_PATH=f'/shared/data/DITSERVE/job-specs/{date}-formal-{commit[:7]}/{name}.json')
            spec.update(name=name, env=[{'name': key, 'value': value} for key, value in values.items()])
            result.append(spec)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit', nargs='?', default='HEAD')
    parser.add_argument('--gpus', type=int, choices=[2, 4, 6, 8], default=2)
    parser.add_argument('--date', default=datetime.now().strftime('%Y%m%d'))
    parser.add_argument('--model-manifests', required=True)
    parser.add_argument('--pair')
    parser.add_argument('--role', choices=['contention', 'no_preemption'])
    args = parser.parse_args()
    datetime.strptime(args.date, '%Y%m%d')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--verify', args.commit+'^{commit}'], text=True).strip()
    values = [json.loads(subprocess.check_output(['git', '-C', str(REPO), 'show', commit+':configs/benchmarks/'+name]))
              for name in ('formal-mixed-v1.json', 'p1-qualification-v1.json')]
    specs = prepare_p1(commit, *values, date=args.date, gpus=args.gpus,
                       model_manifests=args.model_manifests, pair_id=args.pair, role=args.role)
    destination = ROOT / 'job-specs' / f'{args.date}-formal-{commit[:7]}'
    destination.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        path = destination / (spec['name']+'.json')
        raw = json.dumps(spec, indent=2)+'\n'
        if path.exists():
            if path.read_text() != raw:
                raise ValueError('immutable P1 spec differs: ' + str(path))
        else:
            with path.open('x') as stream:
                stream.write(raw)
    print(json.dumps({'specs': len(specs), 'directory': str(destination), 'submitted': False,
                      'requires_stage': 'P0', 'cost_scope': 'same-allocation calibration and three independent seed replays per job'}))


if __name__ == '__main__':
    main()
