"""Checker plans bind the actual formal shape and cannot imply serving success."""
import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_sd35_round_check as check


class SD35RoundCheckTests(unittest.TestCase):
    def protocol(self):
        return json.loads(check.DEFAULT_PROTOCOL.read_bytes())

    def test_formal_1024_selection_and_geometry_binding(self):
        req = check.request_plan(self.protocol(), 'A red fox in snow.', 101, 'sd35-1024')
        self.assertEqual((req.height, req.width, req.steps, req.seed, req.guidance_scale),
                         (1024, 1024, 28, 101, 4.5))
        protocol = self.protocol()
        protocol['shapes']['sd35-1024']['width'] = 512
        with self.assertRaisesRegex(ValueError, 'shape differs'):
            check.request_plan(protocol, 'A fox.', 101, 'sd35-1024')
        with self.assertRaisesRegex(ValueError, 'unsupported frozen'):
            check.request_plan(self.protocol(), 'A fox.', 101, 'wan21-480p')

    def test_shape_is_frozen_512_28_steps_and_cpu_pairing(self):
        req = check.request_plan(self.protocol(), 'A red fox in snow.', 101)
        self.assertEqual((req.height, req.width, req.steps, req.seed, req.guidance_scale),
                         (512, 512, 28, 101, 4.5))
        self.assertEqual(req.max_sequence_length, 256)
        self.assertEqual(req.negative_prompt, 'blurry, low quality')
        self.assertEqual((check.ATOL, check.RTOL), (1e-5, 1e-5))

    def test_modified_protocol_and_invalid_seed_or_prompt_rejected(self):
        for key, value in [('height', 1024), ('steps', 2), ('frames', 81), ('guidance_scale', 7.0)]:
            protocol = self.protocol()
            protocol['shapes']['sd35-512'][key] = value
            with self.assertRaisesRegex(ValueError, 'shape differs'):
                check.request_plan(protocol, 'A fox.', 101)
        for prompt, seed in [('', 101), (' ', 101), ('fox', True), ('fox', -1)]:
            with self.assertRaises(ValueError):
                check.request_plan(self.protocol(), prompt, seed)

    def test_storage_threshold_and_unparseable_result_stop(self):
        from types import SimpleNamespace
        def output(used, percent):
            return SimpleNamespace(returncode=0, stdout=f'Filesystem 1G-blocks Used Available Use% Mounted\n/dev/sdb 1024G {used}G 200G {percent}% /shared\n')
        with patch.object(check.subprocess, 'run', return_value=output(790, 79)):
            self.assertEqual(check.check_storage(), {'used_gib': 790, 'percent': 79})
        for value in [output(791, 79), output(787, 80), SimpleNamespace(returncode=1, stdout='bad')]:
            with patch.object(check.subprocess, 'run', return_value=value):
                with self.assertRaises(SystemExit) as caught:
                    check.check_storage()
                self.assertEqual(caught.exception.code, 75)
        with patch.object(check.subprocess, 'run', side_effect=check.subprocess.TimeoutExpired('df', 20)):
            with self.assertRaises(SystemExit) as caught:
                check.check_storage()
            self.assertEqual(caught.exception.code, 75)

    def test_plan_does_not_load_weights_or_create_output_and_has_no_qualification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_manifest = root / 'model.json'
            model_manifest.write_text('{}')
            output = root / 'output'
            argv = ['plan', '--shared-python-root', str(root), '--model-root', str(root / 'weights'),
                    '--model-manifest', str(model_manifest), '--model-manifest-sha256', check.digest(model_manifest),
                    '--protocol-sha256', check.digest(check.DEFAULT_PROTOCOL), '--prompt', 'A red fox.',
                    '--output-dir', str(output)]
            capture = io.StringIO()
            # Isolate only the filesystem source-binding check; execute must never
            # run. This does not simulate model outputs or CUDA.
            with patch.object(check.adapter, 'verify_shared_sources', return_value=root), \
                 patch.object(check, 'execute', side_effect=AssertionError('must not execute')) as execute, \
                 patch.object(check, 'check_storage', side_effect=AssertionError('plan is local/read-only')), \
                 contextlib.redirect_stdout(capture):
                self.assertEqual(check.main(argv), 0)
            execute.assert_not_called()
            result = json.loads(capture.getvalue())
            self.assertEqual(result['status'], 'planned')
            self.assertEqual(result['planned_trajectories'], 3)
            self.assertEqual(result['completed_trajectories'], 0)
            self.assertEqual(result['intervals'], [[0, 2], [2, 28]])
            self.assertEqual(result['generator_device'], 'cpu')
            self.assertFalse(result['model_weights_loaded'])
            self.assertFalse(result['numerical_check_passed'])
            self.assertFalse(result['qualification_eligible'])
            self.assertFalse(result['serving_executed'])
            self.assertFalse(result['tetriserve_scheduler_executed'])
            self.assertFalse(result['cross_gpu_transfer_tested'])
            self.assertEqual(result['allows'], [])
            self.assertFalse(output.exists())

    def test_bad_protocol_pin_cannot_reach_execution_or_touch_existing_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            previous = output / 'summary.json'
            previous.write_text('old evidence\n')
            argv = ['plan', '--shared-python-root', str(root), '--model-root', str(root),
                    '--model-manifest', str(root / 'missing.json'), '--protocol-sha256', '0' * 64,
                    '--prompt', 'A red fox.', '--output-dir', str(output)]
            capture = io.StringIO()
            with patch.object(check, 'execute', side_effect=AssertionError('must not execute')) as execute, \
                 contextlib.redirect_stdout(capture):
                self.assertEqual(check.main(argv), 1)
            execute.assert_not_called()
            self.assertEqual(previous.read_text(), 'old evidence\n')
            self.assertEqual(json.loads(capture.getvalue())['completed_trajectories'], 0)


if __name__ == '__main__':
    unittest.main()
