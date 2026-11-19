#!/usr/bin/env python3
"""One CUDA arithmetic probe plus five CPU tooling modules; no stage receipt."""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
import unittest

CORE_COMMIT = '51740c05944b063e39bc325b64241cd611a73b41'
CORE_SHORT = CORE_COMMIT[:7]
CORE_SHA = 'e358e29d1a3ca238bbf2b7f981db71e951519715aedf46d58454738aabcbd42b'
PYTHON_SHA = 'fc493414a9ff589850ec25303c9166fb86e138af340f40dfd4caa62ed1b6995a'
DATA = Path('/shared/data/DITSERVE')
BUILDS = Path('/shared/code/DITSERVE/builds')
MODULES = ('test_20261103_native_config_identity', 'test_20261103_p1_isolated_batch',
           'test_20261103_p1_isolated_review', 'test_20261103_p1_isolated_operator',
           'test_20261103_p1_isolated_shell')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plain(path):
    path = Path(path)
    if not path.is_file() or path.resolve() != path.absolute():
        raise ValueError('missing or linked source: ' + str(path))
    return path


def inventory(root):
    root = Path(root)
    files = sorted(root.rglob('*.py'))
    if not files:
        raise ValueError('empty Python source inventory')
    return {p.relative_to(root).as_posix(): sha(plain(p)) for p in files}


def storage():
    try:
        result = subprocess.run(['df', '-BG', '/shared'], check=True, capture_output=True,
                                text=True, timeout=15)
        row = result.stdout.splitlines()[-1].split()
        if int(row[2].rstrip('G')) > 790 or int(row[4].rstrip('%')) >= 80:
            raise ValueError('storage threshold reached')
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        # The caller must not write a shared failure receipt or flush logs.
        raise SystemExit(75)


def verify_sources():
    env = os.environ
    runtime = Path(env['DITSERVE_KAIROS_RUNTIME_ROOT'])
    tests = Path(env['DITSERVE_TEST_ROOT']).parents[1]
    benchmarks = Path(env['DITSERVE_BENCHMARK_ROOT'])
    for root, prefix in ((runtime, DATA / 'scripts/releases'), (tests, DATA / 'tests')):
        if root.parent != prefix or root.resolve() != root:
            raise ValueError('tool snapshot escapes its fixed shared root')
    if (benchmarks != runtime / 'benchmarks' or Path(__file__).resolve() != runtime / 'tools/kairos_tools_validation.py'
            or Path(env['DITSERVE_TEST_ROOT']) != tests / 'tests/python'):
        raise ValueError('tool/test path differs from the submitted snapshot')
    manifest_path = plain(runtime / 'input-manifest.json')
    if sha(manifest_path) != env['DITSERVE_OPERATOR_MANIFEST_SHA256']:
        raise ValueError('operator manifest byte SHA differs')
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get('schema') != 'ditserve.formal-release-inputs.v1'
            or manifest.get('remote_test_root') != str(tests)
            or manifest.get('remote_operator_root') != str(runtime)
            or manifest.get('commit') != env['DITSERVE_TEST_REVISION']):
        raise ValueError('manifest roots/revision differ')
    mapped = {}
    for name, digest in manifest['files'].items():
        key = PurePosixPath(name)
        if (str(key) != name or key.is_absolute() or '..' in key.parts or '\\' in name
                or key.parts[0] not in ('operator', 'tests') or len(key.parts) < 2
                or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('invalid snapshot manifest path/hash')
        path = (runtime if key.parts[0] == 'operator' else tests).joinpath(*key.parts[1:])
        if sha(plain(path)) != digest:
            raise ValueError('snapshot source hash differs: ' + name)
        mapped[str(path)] = digest
    # A source added outside the frozen manifest must not become an import.
    for root in (runtime, tests):
        for path in root.rglob('*'):
            if (path.is_file() and path.suffix in ('.py', '.sh', '.json', '.txt')
                    and path != manifest_path and '__pycache__' not in path.parts
                    and str(path) not in mapped):
                raise ValueError('unmanifested snapshot source: ' + str(path))
    operator_inventory, test_inventory = inventory(benchmarks), inventory(tests / 'benchmarks')
    if operator_inventory != test_inventory:
        raise ValueError('operator and test benchmark copies differ')
    if sha(plain(runtime / 'tools/p1_isolated_job.py')) != operator_inventory.get('p1_isolated_job.py'):
        raise ValueError('actual operator helper differs from the tested helper bytes')
    for module in MODULES:
        plain(tests / 'tests/python' / (module + '.py'))
    build = BUILDS / env['DITSERVE_BUILD']
    info = json.loads(plain(build / 'BUILD_INFO.json').read_bytes())
    package = build / 'share/ditserve/python'
    sources = {}
    for name in ('ditserve', 'xfuser'):
        sources.update({name + '/' + k: v for k, v in inventory(package / name).items()})
    if (env['DITSERVE_BUILD'] != CORE_SHORT or info.get('commit') != CORE_COMMIT
            or sha(plain(build / 'bin/ditserve')) != CORE_SHA or info.get('core_sha256') != CORE_SHA
            or canonical(sources) != PYTHON_SHA):
        raise ValueError('tools diagnostic must retain the frozen 517 core and worker inventory')
    spec_path = plain(env['DITSERVE_JOB_SPEC_PATH'])
    if not spec_path.is_relative_to(DATA):
        raise ValueError('submitted spec is outside the project data root')
    spec = json.loads(spec_path.read_bytes())
    entries = spec.get('env', [])
    declared = {item['name']: item['value'] for item in entries}
    pinned = ('DITSERVE_BUILD', 'DITSERVE_RUN_ID', 'DITSERVE_KAIROS_RUNTIME_ROOT',
              'DITSERVE_TEST_ROOT', 'DITSERVE_BENCHMARK_ROOT', 'DITSERVE_TEST_REVISION',
              'DITSERVE_OPERATOR_MANIFEST_SHA256', 'DITSERVE_TOOLS_RUNNER_SHA256',
              'DITSERVE_JOB_SPEC_PATH', 'DITSERVE_IMAGE_DIGEST')
    if (len(declared) != len(entries) or any(declared.get(k) != env[k] for k in pinned)
            or spec.get('name') != env['DITSERVE_RUN_ID'] or spec.get('workers') != 1
            or spec.get('gpus_per_worker') != 1 or spec.get('mount_shared') is not True
            or not re.fullmatch('sha256:[0-9a-f]{64}', env['DITSERVE_IMAGE_DIGEST'])
            or not spec.get('image', '').endswith('@' + env['DITSERVE_IMAGE_DIGEST'])
            or spec.get('command') != 'bash ' + str(runtime / 'scripts/kairos_tools_validation.sh')
            or sha(Path(__file__)) != env['DITSERVE_TOOLS_RUNNER_SHA256']):
        raise ValueError('actual tooling job differs from its immutable submitted spec')
    return dict(core_commit=CORE_COMMIT, core_sha256=CORE_SHA, python_sources_sha256=PYTHON_SHA,
        core_path=str(build / 'bin/ditserve'), worker_package=str(package),
        build_info_sha256=sha(build / 'BUILD_INFO.json'), operator_manifest_sha256=sha(manifest_path),
        spec_sha256=sha(spec_path), image=spec['image'], image_identity_scope='submitted digest-pinned spec; not pod imageID',
        benchmark_sources_sha256=canonical(operator_inventory), tested_benchmark_root=str(tests / 'benchmarks'),
        actual_p1_benchmark_root=str(benchmarks), test_revision=manifest['commit'], snapshot_files=len(mapped))


class BoundedLog(io.StringIO):
    def __init__(self):
        super().__init__(); self.remaining = 256 * 1024; self.dropped = 0

    def write(self, value):
        retained = value[:self.remaining]
        super().write(retained)
        self.remaining -= len(retained); self.dropped += len(value) - len(retained)
        return len(value)


def suite(output):
    tests = Path(os.environ['DITSERVE_TEST_ROOT']).parents[1]
    package = BUILDS / CORE_SHORT / 'share/ditserve/python'
    sys.path[:0] = [str(tests / 'benchmarks'), str(package), str(tests / 'tests/python')]
    os.environ['DITSERVE_CORE'] = str(package.parents[2] / 'bin/ditserve')
    log = BoundedLog()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        loaded = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        result = unittest.TextTestRunner(stream=log, verbosity=1).run(loaded)
    imported = {}
    benchmark_names = {Path(name).stem for name in inventory(tests / 'benchmarks')}
    for name, module in sys.modules.items():
        file = getattr(module, '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        if name in benchmark_names:
            if path.parent != tests / 'benchmarks':
                raise ValueError('tool tests imported a different benchmark tree: ' + str(path))
            imported[name] = dict(path=str(path), sha256=sha(path))
    required = {'native_config_identity', 'p1_isolated_batch', 'p1_isolated_review', 'p1_isolated_job'}
    if not required <= imported.keys():
        raise ValueError('expected tested benchmark modules were not imported')
    for name in MODULES:
        if Path(sys.modules[name].__file__).resolve() != tests / 'tests/python' / (name + '.py'):
            raise ValueError('test module came from outside the frozen test snapshot')
    for name, module in sys.modules.items():
        file = getattr(module, '__file__', None)
        if file and name.split('.')[0] in ('ditserve', 'xfuser'):
            if not Path(file).resolve().is_relative_to(package):
                raise ValueError('CPU tests imported a worker outside the frozen 517 package')
    value = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skips=len(result.skipped), expected_failures=len(result.expectedFailures),
        unexpected_successes=len(result.unexpectedSuccesses), modules=list(MODULES), imports=imported,
        log_truncated_characters=log.dropped)
    Path(output).write_text(json.dumps(value))
    print(log.getvalue())
    if not result.wasSuccessful() or result.skipped or result.expectedFailures or result.testsRun == 0:
        raise SystemExit(1)


def run():
    before = verify_sources()
    storage()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('expected exactly one allocated CUDA device')
    vector = torch.arange(1024, device='cuda', dtype=torch.float32)
    value = (vector + 1).sum().item()
    matrix = torch.ones((32, 32), device='cuda') @ torch.ones((32, 32), device='cuda')
    torch.cuda.synchronize()
    if value != 524800 or not torch.equal(matrix, torch.full_like(matrix, 32)):
        raise ValueError('CUDA arithmetic probe failed')
    cuda = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                device_count=1, sum=value, matmul_value=32, scope='arithmetic only; no model qualification')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='ditserve-tools-') as directory:
        output = Path(directory) / 'suite.json'
        child = subprocess.Popen([sys.executable, __file__, 'suite', '--output', str(output)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        start = last_storage = time.monotonic()
        try:
            while True:
                try:
                    log, _ = child.communicate(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - last_storage >= 15:
                        storage(); last_storage = time.monotonic()
                    if time.monotonic() - start > 1800:
                        raise TimeoutError('tool suite exceeded 1800 seconds')
        except BaseException:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        storage()
        tests = json.loads(output.read_bytes()) if output.exists() else dict(status='runner_failed')
        status = 'passed' if child.returncode == 0 else 'failed'
        # A fresh process verifies the actual P1 operator import path separately
        # from tests, which deliberately import the identical test snapshot.
        code = '''import importlib.util,json,os,pathlib,sys
r=pathlib.Path(os.environ['DITSERVE_KAIROS_RUNTIME_ROOT']); b=r/'benchmarks'
sys.path.insert(0,str(b))
s=importlib.util.spec_from_file_location('p1_operator_helper',r/'tools/p1_isolated_job.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
identity=m.benchmark_identity(b)
print(json.dumps(dict(helper_path=str(pathlib.Path(m.__file__).resolve()),benchmark_root=identity['root'],imports=identity['imports'],benchmark_sources=identity['sources'])))
'''
        loaded = json.loads(subprocess.check_output([sys.executable, '-c', code], env=env, text=True, timeout=30))
        if canonical(loaded.pop('benchmark_sources')) != before['benchmark_sources_sha256']:
            raise ValueError('fresh actual P1 helper imports differ from tested benchmarks')
        storage()
        if verify_sources() != before:
            raise ValueError('source/spec identity changed during tooling tests/import probe')
        result = dict(schema='ditserve.tools-validation.v1', status=status,
            qualification_eligible=False, allows=[], scope='benchmark/operator tools only; not CONTRACT, P0, P1 or model quality',
            identity=before, cuda_probe=cuda, cpu_tests=tests, actual_p1_imports=loaded,
            dependencies_closed=False, deletion_authorized=False)
        target = Path('/shared/output/DITSERVE/acceptance') / os.environ['DITSERVE_RUN_ID'] / 'tools-validation.json'
        with target.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
        print(log[:256 * 1024])
        print(json.dumps(dict(status=status, tests=tests.get('tests'), qualification_eligible=False)))
        if status != 'passed':
            raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run', 'suite'))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.action == 'preflight': verify_sources()
    elif args.action == 'suite': suite(args.output)
    else: run()
