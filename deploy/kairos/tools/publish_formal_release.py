#!/usr/bin/env python3
"""Publish one prepared test/operator snapshot; no image or job operations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from kairos_queue import storage_gate

ROOT = Path(__file__).resolve().parents[1]
SSH = ['ssh', '-o', 'ClearAllForwardings=yes', '-o', 'ControlMaster=auto', '-o', 'ControlPath=/tmp/ssh-jjjkairos', '-o', 'ControlPersist=30m', '-o', 'BatchMode=yes', 'jjjkairos']


def remote(command, **kwargs):
    return subprocess.run(SSH + [command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('short_commit')
    args = parser.parse_args()
    short = args.short_commit
    if not re.fullmatch(r'[0-9a-f]{7}', short):
        raise ValueError('expected seven-character prepared commit')
    root = ROOT / 'formal-releases' / short
    raw = (root / 'input-manifest.json').read_text()
    manifest = json.loads(raw)
    if not manifest['commit'].startswith(short):
        raise ValueError('source commit differs')
    for name, digest in manifest['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('prepared source changed: ' + name)
    base = '/data/share/jjjkairos/data/DITSERVE'
    targets = {'tests': f'{base}/tests/{short}', 'operator': f'{base}/scripts/releases/{short}'}
    storage_gate()
    for destination in targets.values():
        remote('test ! -e ' + shlex.quote(destination))
    for kind, destination in targets.items():
        storage_gate()
        temporary = destination + '.pending'
        remote('mkdir -p ' + shlex.quote(temporary))
        subprocess.run(['rsync', '-rlpt', '--chmod=Dgo=rx,Fgo=rX', '-e', 'ssh -o ClearAllForwardings=yes -o ControlMaster=auto -o ControlPath=/tmp/ssh-jjjkairos -o ControlPersist=30m -o BatchMode=yes',
                        str(root / kind) + '/', 'jjjkairos:' + temporary + '/'], check=True)
    verification = '''import hashlib,json,pathlib
manifest=json.loads(%r)
targets=json.loads(%r)
for name,digest in manifest['files'].items():
    kind,relative=name.split('/',1)
    path=pathlib.Path(targets[kind]+'.pending')/relative
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise SystemExit('uploaded source SHA differs: '+name)
print('verified',len(manifest['files']),'files')
''' % (raw, json.dumps(targets))
    remote('python3 -', input=verification, text=True)
    for destination in targets.values():
        storage_gate()
        remote('test ! -e ' + shlex.quote(destination) + ' && mv ' + shlex.quote(destination+'.pending') + ' ' + shlex.quote(destination))
    storage_gate()
    subprocess.run(['rsync', '-rlpt', '--chmod=Fgo=r', '-e', 'ssh -o ClearAllForwardings=yes -o ControlMaster=auto -o ControlPath=/tmp/ssh-jjjkairos -o ControlPersist=30m -o BatchMode=yes',
                    str(root / 'input-manifest.json'), 'jjjkairos:' + targets['operator'] + '/input-manifest.json'], check=True)
    print(json.dumps({'commit': manifest['commit'], 'published': targets, 'verified_files': len(manifest['files']),
                      'input_manifest_sha256': hashlib.sha256(raw.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
