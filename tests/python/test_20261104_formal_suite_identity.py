"""Formal suite names survive the native CLI without changing the experiment inputs."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import formal_benchmark as formal
import test_20261102_formal_benchmark as fixtures
from ditserve.evaluation.common import seal

BINARY = Path(os.environ.get('DITSERVE_CORE', ROOT / '.cache/build/ditserve'))
SUFFIXES = {'main': 'main', 'ablation': 'ablation', 'main+ablation': 'main-ablation'}


class FormalSuiteIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FormalBenchmarkTest.setUpClass()
        cls.fixture = fixtures.FormalBenchmarkTest()
        cls.fixture.uuids = cls.fixture.uuids[:4]
        cls.fixture.manifests = copy.deepcopy(cls.fixture.manifests)
        cls.fixture.manifests['wan22'] = dict(
            path='/shared/data/DITSERVE/models/wan22/manifest.json', sha256='d' * 64)

    def build(self, suite, model='wan22'):
        f = self.fixture
        bundle = f.bundle()
        if model == 'wan22':
            bundle['model'] = model
            for old, new in (('wan21-480p', 'wan22-small'), ('wan21-720p', 'wan22-720p')):
                entry = bundle['classes'].pop(old)
                entry['spec'] = formal.shape_spec(f.p, new)
                bundle['classes'][new] = entry
        return formal.serving(f.p, model, f.uuids, seal(bundle), f.pools, f.manifests,
                              seed=301, load=1.2, video_fraction=.5, pattern='bursty', suite=suite)

    def test_suite_suffixes_preserve_policies_and_the_mixed_trace(self):
        f = self.fixture
        before = copy.deepcopy((f.p, f.uuids, f.pools, f.manifests))
        for model in ('wan21', 'wan22'):
            reference = self.build('main', model)
            for suite, suffix in SUFFIXES.items():
                with self.subTest(model=model, suite=suite):
                    config = self.build(suite, model)
                    self.assertEqual(config['experiment'],
                        f'formal-{model}-test-bursty-s301-l1p2-v0p5-{suffix}')
                    self.assertEqual(config['benchmark']['requested_policies'], formal.SUITES[suite])
                    phase = config['phases'][0]
                    self.assertCountEqual(phase['matrix']['/policy'], formal.SUITES[suite])
                    self.assertEqual(config['benchmark']['trace'], reference['benchmark']['trace'])
                    self.assertEqual(phase['matrix']['/workload'], reference['phases'][0]['matrix']['/workload'])
                    self.assertEqual(phase['profile'], reference['phases'][0]['profile'])
                    self.assertEqual(phase['options'], reference['phases'][0]['options'])
                    self.assertEqual(config['workers'], reference['workers'])
                    self.assertEqual(config['benchmark']['trace']['manifest']['counts'],
                                     {'video': 100, 'image': 100})
                    self.assertEqual(sorted(config['benchmark']['trace']['manifest']['class_counts'].values()),
                                     [50, 50, 50, 50])
        self.assertEqual((f.p, f.uuids, f.pools, f.manifests), before)
        self.assertEqual(len(formal.SUITES['main+ablation']), 13)

    @unittest.skipUnless(BINARY.is_file(), 'build the native ditserve executable first')
    def test_native_run_cli_validates_full_configs_before_the_missing_resume_guard(self):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        # A missing resume checkpoint stops after native name/config validation,
        # before source freezing, budget admission, model loading or GPU work.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for suite, suffix in SUFFIXES.items():
                with self.subTest(suite=suite):
                    config = self.build(suite)
                    path = root / (suffix + '.json')
                    path.write_text(json.dumps(config, allow_nan=False), encoding='utf-8')
                    command = [str(BINARY), '--root', str(root), '--package-root', str(ROOT / 'workers'),
                               '--python', sys.executable, 'run', '--config', str(path),
                               '--run-id', suffix, '--resume']
                    result = subprocess.run(command, capture_output=True, text=True,
                                            env=environment, timeout=30)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn('resume needs an existing run', result.stderr)
                    self.assertNotIn('safe path component', result.stderr)
                    work = root / '.cache/results-work' / config['experiment'] / suffix
                    self.assertTrue(work.is_dir())
                    self.assertFalse((work / 'run.json').exists())
                    self.assertFalse((work / 'sources').exists())
            # The original failing name must still be rejected by that same CLI.
            config = self.build('main+ablation')
            config['experiment'] = config['experiment'].removesuffix('main-ablation') + 'main+ablation'
            path = root / 'original.json'
            path.write_text(json.dumps(config, allow_nan=False), encoding='utf-8')
            command = [str(BINARY), '--root', str(root), '--package-root', str(ROOT / 'workers'),
                       '--python', sys.executable, 'run', '--config', str(path),
                       '--run-id', 'original', '--resume']
            rejected = subprocess.run(command, capture_output=True, text=True,
                                      env=environment, timeout=30)
            self.assertEqual(rejected.returncode, 2, rejected.stdout + rejected.stderr)
            self.assertIn('experiment must be a safe path component:', rejected.stderr)
            self.assertFalse((root / '.cache/results-work' / config['experiment']).exists())


class FormalOperatorBindingTest(unittest.TestCase):
    def test_committed_launcher_selects_only_the_bound_serving_compiler(self):
        source = (ROOT / 'benchmarks/kairos_formal_benchmark.sh').read_text()
        # Execute the actual pre-calibration selection block without sourcing
        # the GPU environment or starting a job.
        selection = source[source.index('action=${DITSERVE_FORMAL_ACTION:-p0}'):
                           source.index('case "$action" in')]
        selection += '\nprintf "SELECTED=%s\\nPACKAGE=%s\\n" "$benchmarks" "$PACKAGE"\n'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operator = root / 'operator'
            benchmark = operator / 'benchmarks'
            benchmark.mkdir(parents=True)
            compiler = benchmark / 'formal_benchmark.py'
            compiler.write_text('# bound compiler fixture\n')
            package = root / 'verified-core'
            base = dict(os.environ, PACKAGE=str(package), SCRIPTS=str(operator / 'scripts'),
                        DITSERVE_KAIROS_RUNTIME_ROOT=str(operator),
                        DITSERVE_BENCHMARK_ROOT=str(benchmark),
                        DITSERVE_FORMAL_BENCHMARK_SHA256=hashlib.sha256(compiler.read_bytes()).hexdigest())
            for action in ('serve', 'p1_isolated', 'p0', 'p1'):
                with self.subTest(action=action):
                    result = subprocess.run(['bash', '-eu', '-c', selection],
                        env=dict(base, DITSERVE_FORMAL_ACTION=action), capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    selected = benchmark if action in ('serve', 'p1_isolated') else package / 'benchmarks'
                    self.assertIn('SELECTED=' + str(selected), result.stdout)
                    self.assertIn('PACKAGE=' + str(package), result.stdout)
            for change in ({'DITSERVE_BENCHMARK_ROOT': str(root / 'other/benchmarks')},
                           {'SCRIPTS': str(root / 'other/scripts')},
                           {'DITSERVE_FORMAL_BENCHMARK_SHA256': '0' * 64},
                           {'DITSERVE_BENCHMARK_ROOT': ''}):
                with self.subTest(rejected=change):
                    result = subprocess.run(['bash', '-eu', '-c', selection],
                        env=dict(base, DITSERVE_FORMAL_ACTION='serve', **change),
                        capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('SELECTED=', result.stdout)


if __name__ == '__main__':
    unittest.main()
