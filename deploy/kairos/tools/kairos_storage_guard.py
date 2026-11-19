#!/usr/bin/env python3
"""Enforce the project's storage stop line for an explicit job allowlist only."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from kairos_queue import TERMINAL, checked_get, env_values, load_api, storage_gate


def selected_specs(paths):
    result = {}
    for path in paths:
        spec = json.loads(path.read_bytes())
        env = env_values(spec)
        name = spec['name']
        if (env.get('DITSERVE_RUN_ID') != name or not env.get('DITSERVE_BUILD')
                or not name.startswith(('d20261103-formal-', 'd20261104-formal-'))
                or not spec['image'].startswith('192.168.1.114:30500/jjjkairos/ditserve@sha256:')
                or not spec['command'].startswith('bash /shared/data/DITSERVE/scripts/releases/')):
            raise ValueError('guard refuses a spec outside this explicit DitServe release')
        if name in result:
            raise ValueError('duplicate guarded job')
        result[name] = spec
    return result


def stop_selected(api, specs):
    jobs = checked_get(api, '/api/jobs')
    selected = [job for job in jobs if job.get('namespace') == 'user-jjjkairos'
                and job.get('name') in specs and job.get('state') not in TERMINAL]
    for job in selected:
        # The documented API exposes DELETE as job termination. Shared evidence
        # is not deleted. Names were frozen by explicit local submission specs.
        code, _ = api.req('DELETE', '/api/jobs/user-jjjkairos/' + job['name'])
        if code not in (200, 202, 204, 404):
            raise RuntimeError('termination failed: ' + job['name'] + ' HTTP ' + str(code))
        print(json.dumps({'storage_stop_terminated': job['name'], 'http_status': code}), flush=True)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specs', nargs='+', type=Path)
    parser.add_argument('--transfer-guard', type=Path)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('interval must be between 1 and 60 seconds')
    specs = selected_specs(args.specs)
    api = load_api()
    print(json.dumps({'guarded_jobs': sorted(specs), 'action': 'terminate only on verified storage threshold'}), flush=True)
    while True:
        try:
            storage_gate()
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            if not isinstance(error, RuntimeError) or str(error) != 'shared-storage stop threshold reached':
                print(json.dumps({'storage_status': 'unknown', 'storage_check_unavailable': str(error),
                    'error_type': type(error).__name__, 'retry_seconds': args.interval}), flush=True)
                time.sleep(args.interval)
                continue
            if args.transfer_guard:
                result = subprocess.run(['python3', str(args.transfer_guard), 'storage threshold observed by job guard'])
                if result.returncode:
                    print(json.dumps({'transfer_guard_exit': result.returncode}), flush=True)
            # After the confirmed stop line, never resume normal monitoring or
            # submissions. Retry only the explicit termination if API is busy.
            while True:
                try:
                    stop_selected(api, specs)
                    print('storage stop completed; manual investigation required', flush=True)
                    return
                except Exception as failure:
                    print(json.dumps({'termination_retry': str(failure)}), flush=True)
                    time.sleep(5)
        try:
            jobs = checked_get(api, '/api/jobs')
        except Exception as error:
            print(json.dumps({'jobs_check_unavailable': str(error)}), flush=True)
            time.sleep(args.interval)
            continue
        relevant = [job for job in jobs if job.get('namespace') == 'user-jjjkairos' and job.get('name') in specs]
        if len(relevant) == len(specs) and all(job.get('state') in TERMINAL for job in relevant):
            print('all guarded jobs terminal', flush=True)
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
