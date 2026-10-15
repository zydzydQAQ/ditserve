#!/usr/bin/env python3
"""Install an isolated, fully hash-checked environment from a committed artifact list."""
import argparse
import json
import os
import urllib.parse
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('environment')
p.add_argument('manifest')
a = p.parse_args()
subprocess.run(['python', '-m', 'venv', a.environment], check=True)
python = str(Path(a.environment) / 'bin/python')
entries = json.loads(Path(a.manifest).read_text())['artifacts']
# Optional preseeded cache: every byte is still checked against the official SHA256.
# A missing cache artifact is a build error; it never silently resolves another version.
wheelhouse = os.environ.get('DITSERVE_WHEELHOUSE_URL', '').rstrip('/')
if wheelhouse:
    if urllib.parse.urlsplit(wheelhouse).scheme not in {'http', 'https', 'file'}:
        raise SystemExit('DITSERVE_WHEELHOUSE_URL must be an http, https or file URL')
    entries = [dict(e, url=wheelhouse + '/' + urllib.parse.quote(e['filename'])) for e in entries]
with tempfile.TemporaryDirectory() as directory:
    for label, group in [('bootstrap', [e for e in entries if e['name'] in {'pip', 'setuptools', 'wheel'}]),
                         ('dependencies', entries)]:
        lock = Path(directory) / (label + '.lock')
        lock.write_text('\n'.join(f"{e['name']} @ {e['url']} --hash=sha256:{e['sha256']}" for e in group) + '\n')
        subprocess.run([python, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir',
                        '--no-compile', '--require-hashes', '--no-deps', '--no-build-isolation', '-r', str(lock)], check=True)
subprocess.run([python, '-m', 'pip', 'check'], check=True)
