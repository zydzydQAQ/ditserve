"""Frozen workload generation and retry namespace tests; no GPU or Docker."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from ditserve.evaluation.workflow_serving import validate_manifest
from ditserve.evaluation.workflow_serving_study import (base_manifest, digest, initial_jobs,
    online_jobs, requests, retry_block, prefetch_jobs, trace_jobs)


class StudyTests(unittest.TestCase):
    def study(self, root):
        return dict(root=str(root), run_id='test', source=str(root / 'sources/frozen'),
                    source_digest='test', image='sha256:test', budget_seconds=28800,
                    jobs=[], decisions=[])

    def test_seeded_arrivals_are_sorted_and_frame_prompt_pairs_balanced(self):
        rows = requests(42, rate=.05, slo_ms={'17': 50000, '33': 90000})
        self.assertEqual(rows, requests(42, rate=.05, slo_ms={'17': 50000, '33': 90000}))
        self.assertNotEqual(rows, requests(43, rate=.05))
        self.assertEqual([r['arrival_ms'] for r in rows], sorted(r['arrival_ms'] for r in rows))
        self.assertEqual(sum(r['spec']['frames'] == 17 for r in rows), 8)
        self.assertEqual(len({(r['spec']['frames'], r['spec']['prompt']) for r in rows}), 8)

    def test_all_initial_and_online_manifests_validate_and_reuse_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.study(root)
            initial_jobs(study)
            calibration = dict(lambda0=.07, slo_ms={'17': 50000, '33': 90000},
                               estimates_ms={'17': dict(encode=10000, gpu=14000),
                                             '33': dict(encode=10000, gpu=31000)})
            online_jobs(study, calibration)
            original = copy.deepcopy(study['jobs'])
            online_jobs(study, calibration)
            self.assertEqual(original, study['jobs'])
            for job in study['jobs']:
                if 'ditserve.evaluation.workflow_serving' in job['command']:
                    validate_manifest(json.loads(Path(job['input']).read_text()))
                elif 'ditserve.evaluation.workflow_serving_encoder' in job['command']:
                    from ditserve.evaluation.workflow_serving_encoder import validate_config
                    command = job['command']
                    validate_config(command[command.index('--kind') + 1], json.loads(Path(job['input']).read_text()))
            left = json.loads((root / 'inputs/dual_cpu-shared_fifo-l0p5-r0.json').read_text())
            right = json.loads((root / 'inputs/dedicated_fp32-shared_fifo-l0p5-r0.json').read_text())
            self.assertEqual(left['observations'], right['observations'])

    def test_retry_changes_all_output_namespaces_and_keeps_failed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.study(root)
            initial_jobs(study)
            prefetch_jobs(study, 2)
            previous = [copy.deepcopy(row) for row in study['jobs'] if row['block'] == 'prefetch-c8']
            retry_block(study, 'prefetch-c8')
            revised = [row for row in study['jobs'] if row['block'] == 'prefetch-c8-attempt2']
            self.assertEqual(len(previous), len(revised))
            for a, b in zip(previous, revised):
                self.assertNotEqual(a['output'], b['output'])
                self.assertEqual(json.loads(Path(a['input']).read_text()), json.loads(Path(b['input']).read_text()))
            # Rebuilding a frozen plan keeps the explicit replacement attempts.
            prefetch_jobs(study, 2)
            self.assertEqual(len(revised), 3)

    def test_trace_has_timed_capture_and_retries_into_fresh_capture_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self.study(root)
            original = trace_jobs(study)[0]
            self.assertTrue(original['profiling'])
            self.assertIn('--capture-range-end=none', original['command'])
            self.assertIn('--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0', original['command'])
            retry_block(study, 'trace-overlap')
            replacement = trace_jobs(study)[0]
            self.assertEqual(replacement['attempt'], 2)
            before = original['command'][original['command'].index('--output') + 1]
            after = replacement['command'][replacement['command'].index('--output') + 1]
            self.assertNotEqual(before, after)


if __name__ == '__main__':
    unittest.main()
