"""Guarded reporting and whole-trajectory inference, without CUDA."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ditserve.evaluation import workflow_serving_report as report


def serving(seed=1, *, slo=.5, latency=100., placement='dual_cpu', policy='shared_fifo', confirmation=False, repeat=0):
    spec = dict(request_id='r' + str(seed), seed=seed, frames=17, prompt='boat')
    frozen = dict(request_id=spec['request_id'], spec=spec, arrival_ms=0, slo_ms=1000)
    output = dict(sha256='output', finite=True, dtype='float32', shape=[1, 2])
    request = dict(request_id=spec['request_id'], spec=spec, spec_sha256=report.digest(spec), status='completed',
                   latent=copy.deepcopy(output), video=copy.deepcopy(output))
    config = dict(placement=placement, policy=policy, confirmation=confirmation, repeat=repeat, load_fraction=.8,
                  worker_cpu_sets=[[0, 1], [60, 61]], worker_devices=[0, 1], controller_cpu=59, queue_limit=64, prefetch_depth=1,
                  observations=[dict(id='trace', requests=[frozen], arrival_window_ms=1000, observation_cutoff_ms=3000)])
    if confirmation:
        config['selection'] = dict(family='online')
    obs = dict(id='trace', status='completed', performance_evidence=True, capacity_gate=dict(status='passed'),
               memory_margin_gate=dict(status='passed'), requests=[request],
               summary=dict(completed=1, unfinished=0, rejected=0, slo_attainment=slo, latency_p50_ms=latency))
    return dict(schema='ditserve.workflow.serving.v1', kind='online', status='completed', performance_evidence=True,
                configuration=config, observations=[obs])


def quality():
    metric = dict(finite=True, reference_finite=True, shape_match=True, exact=False, max_abs=.1, rmse=.01, cosine=.999)
    rows = [dict(variant=v, id='case-s7-f17', case='case', seed=7, frames=17, spec=dict(prompt='A <boat>.'), baseline_bridge_exact=True,
                 comparison=dict(latent=copy.deepcopy(metric), video=copy.deepcopy(metric))) for v in ('T0', 'T1', 'T2')]
    return dict(kind='quality', status='completed', qualification=dict(cpu_bridge_exact=True, finite_and_shape=True), requests=rows)


def install(root, name, value, **job_fields):
    path = root / 'measurements' / name
    path.mkdir(parents=True)
    (path / 'run.json').write_text(json.dumps(value))
    launch = root / 'launches' / name
    launch.mkdir(parents=True)
    (launch / 'launch.json').write_text(json.dumps(dict(status='completed')))
    return dict(id=name, output=str(path), **job_fields)


def save_study(root, jobs, **fields):
    (root / 'study.json').write_text(json.dumps(dict(jobs=jobs, **fields)))


class ServingReportTests(unittest.TestCase):
    def test_final_report_reuses_pruned_archive_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = dict(schema='ditserve.run.v1', created_utc='original', metrics=dict(stable=1))
            (root / 'run.json').write_text(json.dumps(saved))
            (root / 'ANALYSIS.html').write_text('original report')
            before = (root / 'run.json').read_bytes()
            with mock.patch.object(report, 'archived_run', return_value=saved), \
                 mock.patch.object(report, 'collect', side_effect=AssertionError('must not rediscover pruned inputs')):
                self.assertEqual(report.build_report(root, final=True), saved)
            self.assertEqual((root / 'run.json').read_bytes(), before)
            self.assertEqual((root / 'ANALYSIS.html').read_text(), 'original report')

    def test_preview_cli_never_finalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(report, 'finalize_workflow') as finalize:
                report.main(['--root', str(root)])
            finalize.assert_not_called()
            self.assertFalse((root / 'run.json').exists())
            self.assertTrue((root / 'ANALYSIS-LIVE.html').is_file())

    def test_final_report_preserves_existing_pending_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = dict(path='quality.npy', bytes=1, sha256='a' * 64, kind='raw',
                            dependencies=[dict(closed=False, reason='blind review pending')])
            (root / 'run.json').write_text(json.dumps(dict(schema='ditserve.run.v1',
                retention=dict(pending_dependencies=['quality']), artifacts=[artifact])))
            run = report.build_report(root, final=True)
            self.assertEqual(run['retention']['pending_dependencies'], ['quality'])
            self.assertIn(artifact, run['artifacts'])

    @unittest.skipUnless(importlib.util.find_spec('openpyxl'), 'openpyxl is required for archive round-trip')
    def test_final_cli_verifies_updated_inventory_and_can_repeat(self):
        from ditserve.results import load
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run02'
            (root / 'traces').mkdir(parents=True)
            capture = b'retained profiler capture'
            (root / 'traces/sample.nsys-rep').write_bytes(capture)
            sqlite = root / 'traces/sample.sqlite'
            sqlite.write_bytes(b'redundant export')
            (root / 'traces/sample.parsed.json').write_text(json.dumps(dict(
                schema_version='ditserve.workflow_trace.v1', status='completed',
                postprocess=dict(capture=dict(bytes=len(capture), sha256=hashlib.sha256(capture).hexdigest())))))
            destination = Path(directory) / 'archive'
            args = ['--root', str(root), '--final', '--archive', str(destination)]
            report.main(args)
            self.assertFalse(sqlite.exists())
            canonical = (root / 'run.json').read_bytes()
            self.assertEqual(load(destination)['run'], json.loads(canonical))
            analysis = (root / 'ANALYSIS.html').read_bytes()
            report.main(args)
            self.assertEqual((root / 'run.json').read_bytes(), canonical)
            self.assertEqual((root / 'ANALYSIS.html').read_bytes(), analysis)

    def test_empty_preview_is_available_without_canonical_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = report.build_report(root)
            self.assertEqual(run['status'], 'partial')
            self.assertTrue((root / 'ANALYSIS-LIVE.html').is_file())
            self.assertFalse((root / 'run.json').exists())
            self.assertEqual(run['metrics']['confirmation']['status'], 'not_started')
            self.assertIsNone(run['config']['analysis_source'])
            source = dict(path='sources/analysis-01', digest='test-digest', files=[dict(path='workflow_serving_report.py', sha256='test-sha')])
            (root / 'analysis-source.json').write_text(json.dumps(source))
            self.assertEqual(report.build_report(root)['config']['analysis_source'], source)

    def test_serving_requires_explicit_performance_and_both_memory_gates(self):
        self.assertTrue(report.performance_valid(serving())['eligible'])
        for field in ('capacity_gate', 'memory_margin_gate'):
            row = serving()
            del row['observations'][0][field]
            self.assertFalse(report.performance_valid(row)['eligible'])
        row = serving(); row['performance_evidence'] = False
        self.assertFalse(report.performance_valid(row)['eligible'])

    def test_scheduling_requires_both_hashes_specs_and_complete_ids(self):
        left, right = serving(), serving(policy='least_work_fifo')
        args = dict(family='online', load=.8, repeat=0, reference='a', candidate='b', quality=report.quality_evidence({}))
        self.assertTrue(report._pair(left, right, **args)['eligible'])
        right['observations'][0]['requests'][0]['latent']['sha256'] = 'changed'
        self.assertFalse(report._pair(left, right, **args)['eligible'])
        right = serving(); right['observations'][0]['requests'][0]['spec']['seed'] = 9
        self.assertFalse(report.exact_outputs(left, right)['passed'])
        right = serving(); right['observations'][0]['requests'] = []
        self.assertFalse(report._pair(left, right, **args)['eligible'])

    def test_placement_differences_are_compared_not_exact(self):
        left, right = serving(latency=100), serving(latency=90, placement='dedicated_fp32')
        right['observations'][0]['requests'][0]['latent']['sha256'] = 'different_device'
        args = dict(family='placement', load=.5, repeat=0, reference='a', candidate='b')
        row = report._pair(left, right, quality=report.quality_evidence({'quality': quality()}), **args)
        self.assertTrue(row['screening_pass'])
        self.assertEqual(row['numeric']['status'], 'compared')
        self.assertIsNone(row['numeric']['passed'])
        self.assertFalse(report._pair(left, right, quality=report.quality_evidence({}), **args)['eligible'])

    def test_latency_screen_is_five_percent_reduction_from_baseline(self):
        args = dict(family='placement', load=.5, repeat=0, reference='a', candidate='b',
                    quality=report.quality_evidence({'quality': quality()}))
        left = serving(latency=100)
        accepted = report._pair(left, serving(latency=95, placement='dedicated_fp32'), **args)
        self.assertAlmostEqual(accepted['improvement'], .05)
        self.assertTrue(accepted['screening_pass'])
        # A 5.15% baseline/candidate ratio gain is only 4.9% latency reduction.
        rejected = report._pair(left, serving(latency=95.1, placement='dedicated_fp32'), **args)
        self.assertFalse(rejected['screening_pass'])
        self.assertAlmostEqual(rejected['improvement'], .049)

    def test_flat_quality_recipe_keeps_case_and_does_not_need_performance_memory_gate(self):
        record = quality()
        evidence = report.quality_evidence({'quality': record})
        self.assertEqual(evidence['status'], 'compared')
        self.assertEqual(evidence['comparisons'][0]['recipe']['id'], 'case-s7-f17')
        self.assertEqual(evidence['comparisons'][0]['recipe']['spec']['prompt'], 'A <boat>.')
        self.assertFalse(report.performance_valid(record)['eligible'])
        self.assertEqual(report.performance_valid(record)['reasons'], ['numerical evidence only'])
        nested = dict(recipe=dict(id='legacy', spec=dict(prompt='legacy')))
        self.assertEqual(report.quality_recipe(nested), nested['recipe'])

    def test_gallery_uses_actual_flat_recipe_container_paths_and_hashes(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = quality()
            baseline = dict(record['requests'][0], variant='cpu_direct', status='reference')
            record['requests'].insert(0, baseline)
            job = install(root, 'quality', record)
            save_study(root, [job])
            for row in record['requests']:
                path = root / 'measurements/quality/outputs' / row['variant'] / row['id'] / 'video.npy'
                path.parent.mkdir(parents=True)
                array = np.linspace(0, 1, 90, dtype=np.float32).reshape(1, 5, 2, 3, 3)
                np.save(path, array, allow_pickle=False)
                row['artifacts'] = dict(video=dict(path=str(Path('/data/workflow') / root.name / path.relative_to(root)),
                    file_sha256=report._file_hash(path), finite=True, shape=list(array.shape), dtype='float32'))
            (root / 'measurements/quality/run.json').write_text(json.dumps(record))
            result = report.generate_quality_gallery(root)
            self.assertEqual(result['clips'], 3)
            page = Path(result['page']).read_text()
            self.assertIn('A &lt;boat&gt;.', page)
            self.assertNotIn('cpu_direct', page)
            self.assertNotIn('T1', page)
            images = list((root / 'quality-gallery').glob('*.png'))
            self.assertEqual(len(images), 15)
            self.assertTrue(all(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for path in images))
            mapping = json.loads(Path(result['mapping']).read_text())
            self.assertEqual({row['variant'] for row in mapping['mappings']}, {'cpu_direct', 'T1', 'T2'})
            Path(mapping['mappings'][0]['source']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                report.generate_quality_gallery(root)

    def test_gallery_reports_opt_in_when_only_video_evidence_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = quality()
            for row in record['requests']:
                row['artifacts'] = dict(video=dict(retained=False, finite=True, sha256='a' * 64))
            job = install(root, 'quality', record)
            save_study(root, [job])
            with self.assertRaisesRegex(ValueError, 'save_videos=true'):
                report.generate_quality_gallery(root)
            self.assertFalse((root / 'quality-gallery').exists())

    def test_screening_requires_each_of_three_blocks_to_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root, jobs = Path(directory), []
            for repeat in range(3):
                for policy, value in [('shared_fifo', .5), ('least_work_fifo', .56)]:
                    name = f'dual_cpu-{policy}-l0p8-r{repeat}'
                    jobs.append(install(root, name, serving(repeat, slo=value, policy=policy)))
            save_study(root, jobs)
            self.assertEqual(report.select_extension(root)['family'], 'online')
            path = root / 'measurements' / 'dual_cpu-least_work_fifo-l0p8-r2' / 'run.json'
            path.write_text(json.dumps(serving(2, slo=.51, policy='least_work_fifo')))
            self.assertIsNone(report.select_extension(root))

    def test_superseded_history_kept_and_current_logical_alias_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [install(root, 'old', dict(kind='P5', status='failed'), logical_id='diagnostic', superseded='retry'),
                    install(root, 'retry', dict(kind='P5', status='qualification_failed', performance_evidence=False), logical_id='diagnostic', attempt=2)]
            save_study(root, jobs)
            study, results, phases, sources, failures = report.collect(root)
            self.assertEqual(list(results), ['diagnostic'])
            self.assertEqual(phases[0]['status'], 'superseded')
            self.assertEqual(len(sources), 2)
            self.assertEqual(len(failures), 1)
            run = report.build_report(root)
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(phases[1]['qualification']['scopes']['numeric']['status'], 'failed')
            self.assertEqual(phases[1]['qualification']['scopes']['performance']['status'], 'unverified')

    def test_profiler_job_cannot_enter_normal_p4_or_request_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_study(root, [install(root, 'trace', serving(), profiling=True, family='extension')])
            run = report.build_report(root)
            self.assertFalse(run['requests'][0]['include_in_performance'])
            self.assertEqual(run['phases'][0]['qualification']['scopes']['performance']['status'], 'unverified')

    def test_p4_trace_has_hashed_independent_evidence_and_missing_is_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = install(root, 'trace-overlap', serving(), profiling=True)
            study = dict(jobs=[job])
            self.assertEqual(report.cuda_overlap_trace(root, study)['status'], 'unavailable')
            (root / 'traces').mkdir()
            path = root / 'traces/trace-overlap.parsed.json'
            parsed = dict(schema_version='ditserve.workflow_trace.v1', source='/traces/trace-overlap.sqlite',
                availability=dict(cuda_kernels=True), arms=[dict(name='P4/timed/streams/frames.17/peer_frames.17/r0',
                devices={'1': dict(dit_vae_kernel_overlap_available=True, dit_vae_kernel_overlap_ms=12.3,
                    dit_a_kernel_union_ms=100., vae_b_kernel_union_ms=40.)})])
            path.write_text(json.dumps(parsed))
            evidence = report.cuda_overlap_trace(root, study)
            self.assertEqual(evidence['status'], 'available')
            self.assertEqual(evidence['source']['sha256'], report._file_hash(path))
            self.assertEqual(evidence['arms'][0]['dit_vae_kernel_overlap_ms'], 12.3)
            self.assertFalse(evidence['include_in_performance'])
            parsed['availability']['cuda_kernels'] = False
            path.write_text(json.dumps(parsed))
            self.assertIsNone(report.cuda_overlap_trace(root, study)['arms'][0]['dit_vae_kernel_overlap_ms'])
            parsed['availability']['cuda_kernels'] = True
            path.write_text(json.dumps(parsed))
            (root / 'launches/trace-overlap/launch.json').write_text('{"status":"interrupted"}')
            self.assertIsNone(report.cuda_overlap_trace(root, study)['arms'][0]['dit_vae_kernel_overlap_ms'])
            job['superseded'] = 'trace-overlap-attempt2'
            retry = install(root, 'trace-overlap-attempt2', serving(), logical_id='trace-overlap', profiling=True)
            study['jobs'].append(retry)
            self.assertEqual(report.cuda_overlap_trace(root, study)['status'], 'unavailable')
            parsed['source'] = '/traces/trace-overlap-attempt2.sqlite'
            retry_path = root / 'traces/trace-overlap-attempt2.parsed.json'
            retry_path.write_text(json.dumps(parsed))
            evidence = report.cuda_overlap_trace(root, study)
            self.assertEqual(evidence['status'], 'available')
            self.assertEqual(evidence['source']['path'], str(retry_path))

    def test_missing_guarded_completion_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); job = install(root, 'x', serving())
            (root / 'launches/x/launch.json').write_text('{"status":"interrupted"}')
            save_study(root, [job])
            self.assertFalse(report.collect(root)[1])
            self.assertEqual(report.build_report(root)['status'], 'partial')

    def test_telemetry_compaction_preserves_cpu_micro_samples(self):
        micro = dict(kind='cpu_micro', samples=[dict(encode_ms=123, process_average_cores=7)])
        self.assertEqual(report.compact(micro), micro)
        result = report.compact(dict(kind='prefetch', samples=[dict(timestamp_ns=1)], phase_records=[dict(name='x')]))
        self.assertEqual(result['samples'], dict(retained_in_source=True, count=1))
        self.assertEqual(report.compact(dict(nvml_samples=[1]))['nvml_samples']['count'], 1)

    def test_prefetch_comparison_is_same_core_and_exact(self):
        def pref(cores, depth):
            row = serving()['observations'][0]['requests'][0]
            row['outputs'] = dict(latent=row.pop('latent'), video=row.pop('video'))
            return dict(kind='prefetch', status='completed', configuration=dict(cpu_threads=cores, depth=depth),
                        qualification=dict(finite_and_shape=True, repeated_exact=True), requests=[row],
                        queues=[dict(repeat=r, wall_ms=100 - depth * 10) for r in range(3)])
        records = {'a': pref(8, 0), 'b': pref(8, 1), 'c': pref(4, 1)}
        rows = report.supplemental_rows(records)['prefetch']
        row = next(row for row in rows if row['cpu_threads'] == 8)
        self.assertTrue(row['eligible']); self.assertEqual(len(row['paired_speedups']), 3)
        self.assertEqual(row['capacity_qualification'], 'unverified')
        self.assertEqual(next(row for row in rows if row['cpu_threads'] == 4)['status'], 'depth0_missing')
        records['b']['requests'][0]['outputs']['video']['sha256'] = 'changed'
        self.assertFalse(next(row for row in report.supplemental_rows(records)['prefetch'] if row['cpu_threads'] == 8)['eligible'])

    def test_confirmation_requires_all_six_fresh_valid_paired_blocks(self):
        records = {}
        for repeat in range(6):
            for policy, slo in [('shared_fifo', .5), ('least_work_fifo', .6)]:
                records[f'{repeat}-{policy}'] = serving(100 + repeat, slo=slo, policy=policy, confirmation=True, repeat=repeat)
        summary = report.confirmation_summary(records)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['interval']['blocks'], 6)
        self.assertIn('entire paired trajectory', summary['interval']['resampling_unit'])
        records['screening'] = serving(100)
        self.assertIsNone(report.confirmation_summary(records)['interval'])
        del records['screening']; del records['5-least_work_fifo']
        self.assertIsNone(report.confirmation_summary(records)['interval'])

    def test_bootstrap_is_deterministic_and_rejects_nonfinite(self):
        a = report.bootstrap_paired([.04, .05, .06], draws=100)
        self.assertEqual(a, report.bootstrap_paired([.04, .05, .06], draws=100))
        self.assertAlmostEqual(a['estimate'], .05)
        self.assertIsNone(report.bootstrap_paired([1, 2]))
        with self.assertRaises(ValueError):
            report.bootstrap_paired([1, 2, float('nan')])

    def test_active_finalization_fails_before_creating_final_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); save_study(root, [], active_job='still_running')
            with self.assertRaises(RuntimeError):
                report.build_report(root, final=True)
            self.assertFalse((root / 'run.json').exists())
            self.assertFalse((root / 'ANALYSIS.html').exists())


if __name__ == '__main__':
    unittest.main()
