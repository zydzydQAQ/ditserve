"""Fail-closed source/model and actual-import contracts for the GF-DiT probe."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('external_gfdit', ROOT / 'benchmarks/external_gfdit.py')
gfdit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gfdit)


class ExternalGFDiTTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / 'source'
        self.shared = self.base / 'shared'
        for name in gfdit.EVIDENCE_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
        for file, cls, model in [('wan.py', 'WanRuntimeV2Adapter', 'WanPipeline'),
                                 ('qwen_image.py', 'QwenImageRuntimeV2Adapter', 'QwenImagePipeline')]:
            path = self.source / 'vllm_omni/diffusion/runtime_v2/adapters' / file
            path.write_text(f'class {cls}:\n    model_class_name = {model!r}\n')
        # These package files are inventory fixtures only; tests never import them.
        for package in ('gfc', 'xfuser'):
            path = self.shared / package / '__init__.py'
            path.parent.mkdir(parents=True)
            path.write_text('raise RuntimeError("inventory fixture must never execute")\n')
        self.manifest = self.base / 'source.json'
        self.metadata = {'schema': 'ditserve.official-source-archive.v1',
                         'repository': gfdit.REPOSITORY, 'commit': gfdit.COMMIT,
                         'archive_sha256': gfdit.ARCHIVE_SHA256,
                         'files': {p.relative_to(self.source).as_posix(): gfdit.digest(p.read_bytes())
                                   for p in self.source.rglob('*') if p.is_file()}}
        self.repin()
        self.protocol = self.base / 'protocol.json'
        self.protocol.write_text(json.dumps({'schema': 'ditserve.formal-protocol.v1', 'id': 'formal-mixed-v1'}))

    def repin(self):
        self.manifest.write_text(json.dumps(self.metadata))
        self.manifest_sha = gfdit.digest(self.manifest.read_bytes())

    def audit(self):
        return gfdit.audit(self.source, self.manifest, self.manifest_sha, self.shared, self.protocol)

    def test_static_admission_never_claims_model_or_serving_success(self):
        result = self.audit()
        self.assertTrue(result['current_models']['wan22']['runtime_v2_registered'])
        self.assertFalse(result['current_models']['sd35']['runtime_v2_registered'])
        self.assertFalse(result['current_models']['wan22']['model_execution_verified'])
        self.assertFalse(result['qualification_eligible'])
        self.assertFalse(result['shared_backend_port']['implemented'])
        self.assertEqual(result['measurements'], [])
        self.assertFalse(result['serving_executed'])
        self.assertEqual(result['native_limits']['dynamic_cfg'], 1)

    def test_source_identity_inventory_hash_and_traversal_are_bound(self):
        with self.assertRaisesRegex(ValueError, 'manifest SHA256'):
            gfdit.verify_source(self.source, self.manifest, '0' * 64)
        self.metadata['commit'] = '0' * 40
        self.repin()
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.audit()
        self.metadata['commit'] = gfdit.COMMIT
        self.metadata['files']['../outside.py'] = '0' * 64
        self.repin()
        with self.assertRaisesRegex(ValueError, 'inventory entry'):
            self.audit()
        del self.metadata['files']['../outside.py']
        self.repin()
        extra = self.source / 'sitecustomize.py'
        extra.write_text('')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            self.audit()
        extra.unlink()
        (self.source / gfdit.EVIDENCE_FILES[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source SHA256'):
            self.audit()

    def test_source_symlink_is_rejected(self):
        target = self.source / gfdit.EVIDENCE_FILES[0]
        target.unlink()
        outside = self.base / 'outside'
        outside.write_text('')
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.audit()

    def test_actual_model_geometry_required_and_hashed_without_weights(self):
        model = self.base / 'model'
        values = {'model_index.json': {'_class_name': 'WanPipeline', 'expand_timesteps': True},
                  'transformer/config.json': {'patch_size': [1, 2, 2], 'num_attention_heads': 24},
                  'vae/config.json': {'scale_factor_spatial': 16, 'scale_factor_temporal': 4},
                  'scheduler/scheduler_config.json': {'_class_name': 'UniPCMultistepScheduler'}}
        for name, value in values.items():
            path = model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        result = gfdit.model_configs({'wan22': model})['wan22']
        self.assertTrue(result['TI2V_expansion_path_required'])
        self.assertEqual(result['geometry']['spatial_scale'], 16)
        self.assertFalse(result['weights_loaded'])
        self.assertEqual(len(result['files_sha256']), 4)
        (model / 'vae/config.json').write_text('{}')
        unresolved = gfdit.model_configs({'wan22': model})['wan22']
        self.assertTrue(unresolved['formal_execution_blocked'])
        self.assertIsNone(unresolved['geometry'])
        (model / 'vae/config.json').write_text('{"scale_factor_spatial":false,"scale_factor_temporal":4}')
        with self.assertRaisesRegex(ValueError, 'VAE geometry'):
            gfdit.model_configs({'wan22': model})
        (model / 'model_index.json').write_text('{"_class_name":"StableDiffusion3Pipeline"}')
        with self.assertRaisesRegex(ValueError, 'model class differs'):
            gfdit.model_configs({'wan22': model})

    def test_real_isolated_import_records_missing_module_without_stubs(self):
        missing = 'ditserve_nonexistent_external_dependency_20261103'
        command = [sys.executable, '-I', '-B', '-c', gfdit._IMPORT_ONE,
                   str(self.source), str(self.shared), missing]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = gfdit.read_json(proc.stdout.removeprefix('GFDIT_IMPORT_JSON='))
        self.assertEqual(result['module'], missing)
        self.assertEqual(result['status'], 'import_failed')
        self.assertEqual(result['error_type'], 'ModuleNotFoundError')
        self.assertEqual(list(self.source.rglob('*.pyc')), [])
        self.assertEqual(list(self.shared.rglob('*.pyc')), [])

    def test_timeout_and_failed_subprocess_cannot_be_promoted(self):
        for value in (True, 0, -1, float('nan'), float('inf'), 301):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gfdit.import_probe(self.source, self.shared, timeout=value)
        failed = subprocess.CompletedProcess([], 1, '', 'real process failed')
        with mock.patch.object(gfdit.subprocess, 'run', return_value=failed):
            result = gfdit.import_probe(self.source, self.shared)
        self.assertEqual(result['status'], 'imports_failed')
        self.assertFalse(result['collectives_executed'])
        self.assertEqual(len(result['imports']), len(gfdit.IMPORT_TARGETS))
        forged = subprocess.CompletedProcess([], 0, 'GFDIT_IMPORT_JSON={"module":"wrong","status":"imported"}', '')
        with mock.patch.object(gfdit.subprocess, 'run', return_value=forged), self.assertRaisesRegex(ValueError, 'identity'):
            gfdit.import_probe(self.source, self.shared)

    def test_cli_has_distinct_inspected_and_invalid_exit_codes(self):
        args = ['audit', '--source-root', str(self.source), '--source-manifest', str(self.manifest),
                '--source-manifest-sha256', self.manifest_sha, '--shared-python-root', str(self.shared),
                '--protocol', str(self.protocol)]
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            self.assertEqual(gfdit.main(args), 2)
        self.assertFalse(json.loads(capture.getvalue())['qualification_eligible'])
        args[args.index(self.manifest_sha)] = '0' * 64
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            self.assertEqual(gfdit.main(args), 1)
        self.assertEqual(json.loads(capture.getvalue())['status'], 'invalid_evidence')


if __name__ == '__main__':
    unittest.main()
