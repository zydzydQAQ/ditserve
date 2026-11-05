"""Regression tests for 2026-11-05 review findings 8, 12, 14 and 15.

8  P0 diagnostics build under both frozen protocols; only the pattern that is
   actually executed is validated, and inconsistency is reported explicitly.
12 Cross-SP gates check every declared threshold; SSIM is a real windowed SSIM.
14 The superseded external-serving recipe cannot default to the deleted base.
15 Provenance for deleted files is pinned to git history with matching hashes.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / 'benchmarks', ROOT / 'workers', ROOT / 'tests' / 'python'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import external_wan_topology_step as wan_topo
import external_xdit_numerics as xdit_num
import formal_benchmark as formal
from ditserve.evaluation.common import canonical_hash, seal
from test_20261104_external_wan_topology_step import rows as wan_rows

DELETED_SLIM_DIGEST = '2009ea26f63bcf0360ce7b4d71ed508c91c0702c97dd97e372cecfb672a818a1'
R3_DIGEST = 'sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05'


def fixture(protocol_path, model):
    p = formal.protocol(protocol_path)
    uuids = [f'GPU-{i}' for i in range(8)]
    pools = {kind: seal(dict(kind=expected['kind'], source=expected,
                             samples=[dict(sample_id=f'{kind}-{i}', prompt=f'{kind} sample {i}')
                                      for i in range(1200)]))
             for kind, expected in p['datasets'].items()}
    manifests = {name: dict(path=f'/shared/data/DITSERVE/models/{name}/manifest.json', sha256='a' * 64)
                 for name in (model, 'sd35')}
    entries = {}
    for sid, shape in p['shapes'].items():
        if shape['scope'] != 'development' or shape['model'] not in (model, 'sd35'):
            continue
        if shape['kind'] == 'video':
            cost = dict(kind='video', steps=shape['steps'], blocks_per_step=60, prepare_ms=0,
                        encode_ms=20, step_compute_ms=1000, finish_ms=100, comm_mb_per_step=100,
                        collectives_per_step=240, cfg_exchange_mb=1, decode_chunks=21,
                        links={str(d): dict(gbps=5, latency_ms=.1) for d in (2, 3, 4, 6, 8)})
        else:
            cost = dict(kind='image', service_ms=250.0)
        entries[sid] = dict(spec=formal.shape_spec(p, sid), cost=cost,
                            source=dict(run_sha256='b' * 64, core_sha256='c' * 64, resize_ms=10,
                                        quantum_overhead_ms=1, gpu_uuids=uuids))
    bundle = seal(dict(schema=formal.BUNDLE_SCHEMA, model=model, protocol_sha256=canonical_hash(p),
                       gpu_uuids=uuids, classes=entries, model_manifests=manifests))
    return p, uuids, bundle, pools, manifests


class P0DiagnosticProtocolTest(unittest.TestCase):
    CASES = ((formal.DEFAULT_PROTOCOL, 'wan21'), (formal.DEFAULT_V2_PROTOCOL, 'wan22'))

    def build(self, p, uuids, bundle, pools, manifests, model, diagnostic, **kwargs):
        return formal.serving(p, model, uuids, bundle, pools, manifests, split='development',
                              seed=201, diagnostic=diagnostic, **kwargs)

    def test_p0_diagnostics_build_under_v1_and_v2_with_their_own_controls(self):
        for path, model in self.CASES:
            p, uuids, bundle, pools, manifests = fixture(path, model)
            for diagnostic in ('contention', 'no_preemption'):
                with self.subTest(protocol=p['id'], diagnostic=diagnostic):
                    controls = p['p0_diagnostics'][diagnostic]
                    # Exactly the arguments the operator script now passes for P0.
                    cfg = self.build(p, uuids, bundle, pools, manifests, model, diagnostic)
                    benchmark = cfg['benchmark']
                    self.assertEqual(benchmark['p0_diagnostic_controls'], controls)
                    for field in ('count', 'video_fraction', 'load', 'pattern',
                                  'video_slo_scale', 'image_slo_scale'):
                        self.assertEqual(benchmark[field], controls[field], field)
                    window = benchmark['trace']['manifest']['windows'][0]
                    self.assertEqual(window['pattern'], controls['pattern'])
                    self.assertEqual(window['load'], controls['load'])
                    self.assertEqual(benchmark['requested_workload_before_diagnostic_override'],
                                     dict(load=None, video_fraction=None, pattern=None))
                    self.assertTrue(cfg['experiment'].endswith('-' + diagnostic))
                    self.assertIn(f"-{controls['pattern']}-", cfg['experiment'])

    def test_v2_contention_uses_declared_gamma_not_removed_bursty(self):
        p, uuids, bundle, pools, manifests = fixture(formal.DEFAULT_V2_PROTOCOL, 'wan22')
        self.assertNotIn('bursty', [x['name'] for x in p['patterns']])
        self.assertEqual(p['p0_diagnostics']['contention']['pattern'], 'gamma')
        cfg = self.build(p, uuids, bundle, pools, manifests, 'wan22', 'contention')
        self.assertEqual(cfg['benchmark']['trace']['manifest']['windows'][0]['pattern'], 'gamma')

    def test_legacy_requested_values_are_recorded_but_not_validated_against_matrix(self):
        # The pre-fix operator passed --load 1.2 --pattern bursty; under v2 neither
        # is in the formal matrix, yet the diagnostic must still build and record them.
        p, uuids, bundle, pools, manifests = fixture(formal.DEFAULT_V2_PROTOCOL, 'wan22')
        cfg = self.build(p, uuids, bundle, pools, manifests, 'wan22', 'contention',
                         load=1.2, video_fraction=.5, pattern='bursty')
        self.assertEqual(cfg['benchmark']['requested_workload_before_diagnostic_override'],
                         dict(load=1.2, video_fraction=.5, pattern='bursty'))
        self.assertEqual(cfg['benchmark']['pattern'], 'gamma')

    def test_undeclared_diagnostic_pattern_is_an_explicit_protocol_inconsistency(self):
        p, uuids, bundle, pools, manifests = fixture(formal.DEFAULT_V2_PROTOCOL, 'wan22')
        broken = copy.deepcopy(p)
        broken['p0_diagnostics']['contention']['pattern'] = 'bursty'
        bundle = copy.deepcopy(bundle)
        bundle.pop('sha256')
        bundle['protocol_sha256'] = canonical_hash(broken)
        with self.assertRaisesRegex(ValueError, r"p0_diagnostics 'contention' uses arrival pattern "
                                                r"'bursty'.*internally inconsistent"):
            self.build(broken, uuids, seal(bundle), pools, manifests, 'wan22', 'contention')

    def test_non_diagnostic_cells_still_validate_the_formal_matrix(self):
        p, uuids, bundle, pools, manifests = fixture(formal.DEFAULT_V2_PROTOCOL, 'wan22')
        with self.assertRaisesRegex(ValueError, 'arrival pattern is outside frozen protocol'):
            formal.serving(p, 'wan22', uuids, bundle, pools, manifests, split='development',
                           seed=201, pattern='bursty')
        with self.assertRaisesRegex(ValueError, 'load/mix is outside frozen protocol'):
            formal.serving(p, 'wan22', uuids, bundle, pools, manifests, split='development',
                           seed=201, load=1.2)

    def test_operator_p0_branch_passes_no_ignored_workload_flags(self):
        source = (ROOT / 'benchmarks/kairos_formal_benchmark.sh').read_text()
        branch = source[source.index('if [ "$action" = p0 ]; then\n    #'):]
        branch = branch[:branch.index('  else')]
        self.assertIn('--diagnostic "$diagnostic"', branch)
        for flag in ('--pattern', '--load', '--video-fraction'):
            self.assertNotIn(flag, branch)


def global_ssim(x, y, data_range=1.0):
    """Reference standard global SSIM (means, variances, covariance)."""
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy, cov = x.var(), y.var(), np.mean((x - mx) * (y - my))
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


class CrossSPQualityGateTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        storage = patch.object(xdit_num, 'check_storage', return_value={'used_gib': 1, 'percent': 1})
        storage.start()
        self.addCleanup(storage.stop)

    def compare(self, reference, candidate):
        shape = list(reference.shape)
        prefix = json.dumps({'shape': shape, 'dtype': 'float32'}, sort_keys=True).encode()
        paths, expected = [], []
        for name, array in (('sp1.raw', reference), ('sp4.raw', candidate)):
            raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
            path = self.root / name
            path.write_bytes(raw)
            paths.append(path)
            expected.append(dict(file_sha256=hashlib.sha256(raw).hexdigest(),
                                 native_output_sha256=hashlib.sha256(prefix + raw).hexdigest()))
        return xdit_num.compare_files(paths[0], paths[1], shape, expected)

    def test_identical_frames_have_ssim_one_and_pass(self):
        frames = np.random.default_rng(1).uniform(0.05, 0.95, (3, 24, 32, 3)).astype(np.float32)
        metrics = self.compare(frames, frames)
        self.assertEqual(metrics['ssim'], 1.0)
        self.assertTrue(metrics['cross_sp_quality_gate_passed'])
        self.assertNotIn('ssim_global', metrics)
        self.assertIn('Gaussian-window SSIM', metrics['ssim_method'])
        self.assertEqual(metrics['metrics_not_computed'], ['LPIPS', 'VBench'])

    def test_windowed_ssim_matches_scipy_reference_definition(self):
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            self.skipTest('scipy unavailable')
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 1, (19, 27, 3))
        y = np.clip(x + rng.normal(0, .05, x.shape), 0, 1)
        expected = []
        for c in range(3):
            a, b = x[..., c], y[..., c]
            f = lambda v: gaussian_filter(v, 1.5, truncate=3.0)
            ma, mb = f(a), f(b)
            va, vb, cv = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
            expected.append(((2 * ma * mb + 1e-4) * (2 * cv + 9e-4)
                             / ((ma * ma + mb * mb + 1e-4) * (va + vb + 9e-4))).mean())
        total, count = xdit_num.windowed_ssim_sum(x, y)
        self.assertAlmostEqual(total / count, float(np.mean(expected)), places=12)

    def test_structural_distortion_with_low_relative_l2_fails_on_ssim(self):
        # A bright, flat-ish reference plus a small checkerboard perturbation keeps
        # PSNR, cosine and relative L2 inside their bounds but destroys local structure.
        rng = np.random.default_rng(3)
        reference = (0.8 + 0.01 * rng.standard_normal((2, 32, 32, 3))).astype(np.float32)
        checker = np.where((np.indices((32, 32)).sum(axis=0) % 2) == 0, 1.0, -1.0)
        candidate = (reference + 0.08 * checker[None, :, :, None]).astype(np.float32)
        metrics = self.compare(reference, candidate)
        limits = xdit_num.CROSS_SP_QUALITY_THRESHOLDS
        self.assertLessEqual(metrics['relative_l2_error'], limits['relative_l2_max'])
        self.assertGreaterEqual(metrics['cosine_similarity'], limits['cosine_similarity_min'])
        self.assertGreaterEqual(metrics['psnr_db'], limits['psnr_db_min'])
        self.assertLess(metrics['ssim'], limits['ssim_min'])
        self.assertLess(global_ssim(reference, candidate), limits['ssim_min'])
        self.assertFalse(metrics['cross_sp_quality_gate_passed'])
        # The removed raw-second-moment formula would have passed this case.
        n = reference.size
        a, b = reference.astype(np.float64), candidate.astype(np.float64)
        legacy = (2 * np.sum(a * b) / n + 9e-4) / (np.sum(a * a) / n + np.sum(b * b) / n + 9e-4)
        self.assertGreater(legacy, limits['ssim_min'])

    def test_quality_gate_fails_closed_without_ssim(self):
        quality = xdit_num.evaluate_cross_sp_quality(0., 0., 0., 1., 1., 1., 1)
        self.assertFalse(quality['cross_sp_quality_gate_passed'])
        self.assertIsNone(quality['ssim'])
        for bad in (math.nan, 1.5, True):
            self.assertFalse(xdit_num.evaluate_cross_sp_quality(
                0., 0., 0., 1., 1., 1., 1, ssim=bad)['cross_sp_quality_gate_passed'])
        self.assertTrue(xdit_num.evaluate_cross_sp_quality(
            0., 0., 0., 1., 1., 1., 1, ssim=0.9)['cross_sp_quality_gate_passed'])
        self.assertFalse(xdit_num.evaluate_cross_sp_quality(
            0., 0., 0., 1., 1., 1., 1, ssim=0.84)['cross_sp_quality_gate_passed'])


class WanTopologyGateTest(unittest.TestCase):
    def aggregate(self, comparison):
        rows = wan_rows(2)
        for boundary in rows[0]['boundaries']:
            boundary['comparison'] = dict(boundary['comparison'], **comparison)
        return wan_topo.aggregate(rows, rows[0]['binding'], 2, {'pin': 'immutable'}, 'a' * 64)

    def test_bitwise_equal_boundaries_pass(self):
        gate = self.aggregate({})['cross_sp_acceptance_gate']
        self.assertTrue(gate['passed'])
        self.assertEqual(gate['failed_boundaries'], {})
        self.assertEqual((gate['max_relative_l2'], gate['min_cosine_similarity']), (0.0, 1.0))
        self.assertEqual(gate['metrics_not_computed'], ['PSNR', 'SSIM', 'LPIPS', 'VBench'])

    def test_measured_cosine_below_minimum_fails_even_with_low_relative_l2(self):
        gate = self.aggregate(dict(bitwise_equal=False, within_tolerance=False,
                                   relative_l2=0.05, cosine_similarity=0.90))['cross_sp_acceptance_gate']
        self.assertFalse(gate['passed'])
        self.assertEqual(set(next(iter(gate['failed_boundaries'].values()))), {'cosine_similarity'})
        self.assertEqual(gate['min_cosine_similarity'], 0.90)

    def test_cosine_lower_bound_from_relative_l2_is_enforced(self):
        # Without a measured cosine, relL2=0.24 only guarantees cos >= 0.9708 (passes);
        # relL2=0.25 still passes relL2 but its bound 0.9682 is checked, never skipped.
        passed = self.aggregate(dict(bitwise_equal=False, within_tolerance=False, relative_l2=0.24))
        gate = passed['cross_sp_acceptance_gate']
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(gate['min_cosine_similarity'], math.sqrt(1 - 0.24 ** 2))
        result = wan_topo.boundary_gate(dict(bitwise_equal=False, relative_l2=0.2),
                                        dict(relative_l2_max=0.25, cosine_similarity_min=0.99))
        self.assertFalse(result['passed'])
        self.assertEqual(result['failed_thresholds'], ['cosine_similarity'])
        self.assertEqual(result['cosine_basis'], 'lower_bound_from_relative_l2')

    def test_within_tolerance_alone_no_longer_bypasses_thresholds(self):
        gate = self.aggregate(dict(bitwise_equal=False, within_tolerance=True,
                                   relative_l2=None))['cross_sp_acceptance_gate']
        self.assertFalse(gate['passed'])
        self.assertEqual(set(next(iter(gate['failed_boundaries'].values()))),
                         {'relative_l2', 'cosine_similarity'})


class ImageRecipeAndProvenanceTest(unittest.TestCase):
    def test_external_serving_recipe_has_no_default_base_and_rejects_deleted_slim(self):
        text = (ROOT / 'docker/Dockerfile').read_text(encoding='utf-8')
        args = re.findall(r'^ARG BASE_IMAGE(.*)$', text, re.MULTILINE)
        self.assertTrue(args)
        self.assertTrue(all(rest.strip() == '' for rest in args), 'BASE_IMAGE must have no default')
        directives = [line for line in text.splitlines() if line and not line.startswith('#')]
        self.assertFalse(any(DELETED_SLIM_DIGEST in line and 'case' not in line and '*' not in line
                             for line in directives))
        guard = re.search(r'RUN case "\$\{BASE_IMAGE\}" in(.*?)esac', text, re.DOTALL).group(1)
        script = 'case "$BASE_IMAGE" in' + guard.replace('\\\n', '\n') + 'esac'
        def check(reference):
            return subprocess.run(['sh', '-c', script], env={'BASE_IMAGE': reference},
                                  capture_output=True, text=True, timeout=10)
        for refused in ('192.168.1.114:30500/jjjkairos/ditserve@sha256:' + DELETED_SLIM_DIGEST,
                        '192.168.1.114:30500/jjjkairos/ditserve:xdit-dev-016e1ca-slim',
                        'registry/ditserve:any-tag', ''):
            with self.subTest(refused=refused):
                result = check(refused)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('AGENTS.md', result.stderr)
        self.assertEqual(check('example/base@' + R3_DIGEST).returncode, 0)

    def test_handoff_is_superseded_and_points_to_r3(self):
        handoff = json.loads((ROOT / 'configs/benchmarks/external-image-build-handoff-v1.json')
                             .read_text(encoding='utf-8'))
        self.assertEqual(handoff['status'], 'superseded_historical_do_not_build')
        self.assertFalse(handoff['supersession']['build_allowed'])
        self.assertEqual(handoff['supersession']['superseded_by']['manifest_digest'], R3_DIGEST)
        self.assertEqual(handoff['base_image']['status'], 'deleted_2026-11-04_do_not_use')
        for name, item in handoff['build_context']['files'].items():
            data = item['content'].encode('utf-8')
            self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (item['bytes'], item['sha256']), name)
            self.assertNotIn('ditserve/requirements/', item['source_path'])

    def git_blob(self, reference):
        match = re.fullmatch(r'git:([0-9a-f]{40}):(.+)', reference)
        self.assertIsNotNone(match, reference)
        try:
            result = subprocess.run(['git', '-C', str(ROOT), 'show', f'{match[1]}:{match[2]}'],
                                    capture_output=True, timeout=30)
        except FileNotFoundError:
            self.skipTest('git unavailable')
        if result.returncode != 0:
            self.skipTest('git history unavailable in this checkout')
        return result.stdout

    def test_deleted_requirement_provenance_is_git_pinned_with_matching_hashes(self):
        followup = json.loads((ROOT / 'configs/benchmarks/external-runtime-r3-followup-v1.json')
                              .read_text(encoding='utf-8'))
        blob = self.git_blob(followup['requirements']['path'])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), followup['requirements']['sha256'])
        handoff = json.loads((ROOT / 'configs/benchmarks/external-image-build-handoff-v1.json')
                             .read_text(encoding='utf-8'))
        for name in ('external-serving-common.txt', 'external-serving-constraints.txt'):
            item = handoff['build_context']['files'][name]
            blob = self.git_blob(item['source_path'])
            self.assertEqual((len(blob), hashlib.sha256(blob).hexdigest()), (item['bytes'], item['sha256']))
            self.assertEqual(blob.decode('utf-8'), item['content'])

    def test_owned_files_do_not_reference_deleted_paths(self):
        deleted = re.compile(r'docs/container\.md|docs/20261102-[\w-]+\.md|docs/dit_serving_\w+\.(?:bib|tex)'
                             r'|(?<!git:[0-9a-f]{40}:)requirements/external-serving-[\w-]+\.txt')
        owned = [ROOT / 'README.md', ROOT / 'docker/Dockerfile',
                 ROOT / 'configs/benchmarks/external-image-build-handoff-v1.json',
                 ROOT / 'configs/benchmarks/external-runtime-r3-followup-v1.json',
                 *sorted((ROOT / 'docs').glob('*.md'))]
        for path in owned:
            with self.subTest(path=path.name):
                text = path.read_text(encoding='utf-8')
                if path.suffix == '.json':
                    # Embedded historical file contents are hash-bound; check only the keys we own.
                    value = json.loads(text)
                    files = value.get('build_context', {}).get('files', {})
                    text = json.dumps([value.get('requirements', {}).get('path')]
                                      + [item.get('source_path') for item in files.values()])
                self.assertIsNone(deleted.search(text), deleted.search(text) and deleted.search(text)[0])


if __name__ == '__main__':
    unittest.main()
