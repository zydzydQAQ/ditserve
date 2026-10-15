#!/usr/bin/env python3
"""Materialize the pinned sparse checkout, including root ignore files, for offline use."""
import argparse
from pathlib import Path
import subprocess
import time

REVISION = 'c6a27e771e803b1c12ffa28b82ba69b42472020e'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('destination')
p.add_argument('--proxy', default='')
a = p.parse_args()
source = Path(a.destination)
subprocess.run(['git', 'init', str(source)], check=True)
git = ['git', '-C', str(source)]
subprocess.run(git + ['remote', 'add', 'origin', 'https://github.com/NVlabs/LongLive.git'], check=True)
subprocess.run(git + ['sparse-checkout', 'init', '--cone'], check=True)
subprocess.run(git + ['sparse-checkout', 'set', 'LongLive1.0'], check=True)
network_git = git + ['-c', 'http.proxy=' + a.proxy, '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=30']

def retry(arguments, capture=False):
    for attempt in range(5):
        try:
            return subprocess.run(network_git + arguments, check=True, text=True,
                                  capture_output=capture, timeout=180).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 4:
                raise
            time.sleep(2)

retry(['fetch', '--depth', '1', '--filter=blob:none', 'origin', REVISION])
retry(['checkout', '--detach', 'FETCH_HEAD'])
assert subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == REVISION
# Status may need the root .gitignore blob; materialize it now, while online.
assert not retry(['status', '--porcelain'], capture=True).strip(), 'dirty upstream checkout'
