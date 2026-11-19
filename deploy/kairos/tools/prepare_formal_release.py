#!/usr/bin/env python3
"""Stage committed test inputs and an immutable operator snapshot locally."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1] / 'ditserve'


def stage_benchmark_snapshot(committed, operator):
    """Keep benchmark-relative resources beside the exact committed tools."""
    for directory in ('benchmarks', 'configs'):
        shutil.copytree(Path(committed) / directory, Path(operator) / directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commit')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--verify', args.commit+'^{commit}'], text=True).strip()
    destination = ROOT / 'formal-releases' / commit[:7]
    destination.mkdir(parents=True, exist_ok=False)
    tests = destination / 'tests'
    tests.mkdir()
    raw = subprocess.check_output(['git', '-C', str(REPO), 'archive', commit, 'tests', 'benchmarks', 'configs'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(tests, filter='data')
    operator = destination / 'operator'
    # Analysis/compilation tools may evolve independently of the selected GPU
    # core and workers. Keep committed benchmark sources and their adjacent
    # configs in the immutable manifest alongside the explicit runtime tools.
    stage_benchmark_snapshot(tests, operator)
    for directory in ('scripts', 'tools'):
        for source in (ROOT / directory).rglob('*'):
            if source.is_file() and source.suffix in ('.py', '.sh') and '__pycache__' not in source.parts:
                target = operator / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    files = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(destination.rglob('*')) if path.is_file()}
    manifest = {'schema': 'ditserve.formal-release-inputs.v1', 'commit': commit, 'files': files,
                'remote_test_root': f'/shared/data/DITSERVE/tests/{commit[:7]}',
                'remote_operator_root': f'/shared/data/DITSERVE/scripts/releases/{commit[:7]}'}
    (destination / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'destination': str(destination), 'commit': commit, 'files': len(files),
                      'bytes': sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
                      'published': False}))


if __name__ == '__main__':
    main()
