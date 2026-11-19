#!/usr/bin/env python3
"""Read-only Kairos allocation and shared-storage monitor; no job mutation."""
import argparse
import datetime
import json
import subprocess
import time

from kairos_queue import checked_get, live_capacity, load_api, storage_gate, TERMINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--namespace', default='user-jjjkairos')
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('--interval must be in [1, 60] seconds')
    api = load_api()
    while True:
        try:
            storage = storage_gate()
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            threshold = isinstance(error, RuntimeError) and str(error) == 'shared-storage stop threshold reached'
            print(json.dumps({'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'storage_status': 'threshold_reached' if threshold else 'unknown',
                'storage': None, 'capacity': None, 'storage_check_unavailable': str(error),
                'error_type': type(error).__name__,
                'retry_seconds': args.interval if args.watch and not threshold else None}), flush=True)
            # This monitor is read-only. The separate explicit-allowlist guard
            # handles verified threshold termination; unknown is never safe.
            if threshold or not args.watch:
                raise SystemExit(2 if threshold else 1)
            time.sleep(args.interval)
            continue
        jobs = checked_get(api, '/api/jobs')
        capacity = live_capacity(api, jobs, args.namespace)
        print(json.dumps({'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'storage': storage, 'capacity': capacity,
            'active_account_jobs': [{k: job[k] for k in ('name', 'state', 'total_gpus')}
                for job in jobs if job.get('namespace') == args.namespace and job.get('state') not in TERMINAL]},
            ensure_ascii=False), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
