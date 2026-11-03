"""Official-source binding and honest unsupported-model/import failure reporting."""
import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_tetriserve as ts


class TetriServeProbeTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        root = base / 'official'
        shared = base / 'shared' / 'xfuser'
        modules = {
            'tetriserve/__init__.py': "SUPPORTED_MODEL_DICT = {'sd3': 'stabilityai/stable-diffusion-3-medium-diffusers'}\n",
            'tetriserve/engine/runner.py': "class FluidServeRunner:\n    MODEL_REGISTRY = {'stabilityai/stable-diffusion-3-medium-diffusers': 'SD3Pipeline'}\n",
            'tetriserve/server/launcher.py': '# Real importable source fixture; never launch a server.\n',
            'tetriserve/server/api_server.py': 'class RequestInput:\n    prompt: str\n    guidance_scale: float\n    latency_threshold: float\ndef generate(request):\n    return InputConfig(prompt=request.prompt)\n',
            'tetriserve/xfuser/config/config.py': 'class InputConfig:\n    prompt: str\n    num_frames: int\n',
            'tetriserve/xfuser/' + ts.SD3_PIPELINE: 'class xFuserStableDiffusion3Pipeline:\n' + ''.join('    def ' + n + '(self): pass\n' for n in ts.ROUND_METHODS),
        }
        for relative in ['tetriserve/engine/runtime.py', 'tetriserve/engine/proc_launcher.py',
                         'tetriserve/scheduler/dyn_step_scheduler.py', 'tetriserve/scheduler/perf_model.py',
                         'tetriserve/light_ray/latent_manager.py', 'tetriserve/distributed/globals.py']:
            modules[relative] = '# Isolated source inventory fixture.\n'
        for relative, source in modules.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        for relative, source in {
                '__init__.py': '# Actual isolated import target.\n',
                ts.SD3_PIPELINE: 'class xFuserStableDiffusion3Pipeline:\n    def __call__(self): pass\n',
                ts.GROUP_COORDINATOR: 'class SequenceParallelGroup:\n    pass\n',
        }.items():
            path = shared / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        manifest = base / 'source.json'
        record = {'schema': 'ditserve.official-source-archive.v1', 'repository': ts.REPOSITORY,
                  'commit': ts.COMMIT, 'archive_sha256': ts.ARCHIVE_SHA256,
                  'files': {p.relative_to(root).as_posix(): ts.digest(p.read_bytes())
                            for p in root.rglob('*') if p.is_file()}}
        manifest.write_text(json.dumps(record))
        return root, manifest, ts.digest(manifest.read_bytes()), shared

    def test_current_models_and_shared_backend_remain_blocked(self):
        with TemporaryDirectory() as directory:
            args = self.fixture(directory)
            result = ts.audit(*args)
            self.assertEqual(set(result['current_models']), {'wan21', 'wan22', 'sd35'})
            self.assertTrue(all(not v['registered'] and not v['runnable'] for v in result['current_models'].values()))
            self.assertEqual(result['ignored_api_fields'], ['guidance_scale'])
            self.assertEqual(set(result['shared_xfuser']['missing_round_methods']), set(ts.ROUND_METHODS))
            self.assertFalse(result['shared_xfuser']['dynamic_group_update'])
            self.assertFalse(result['serving_executed'])
            self.assertFalse(result['qualification_eligible'])
            self.assertEqual(result['measurements'], [])
            self.assertEqual(result['allows'], [])
            self.assertEqual(result['official_schedule_logic'], 'dyn_slo_schedule_global_dp')

    def test_manifest_pin_file_identity_and_unlisted_module_fail_closed(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'manifest SHA256 differs'):
                ts.verify_source(root, manifest, '0' * 64)
            runner = root / 'tetriserve/engine/runner.py'
            previous = runner.read_bytes()
            runner.write_bytes(previous + b'\n# mutated\n')
            with self.assertRaisesRegex(ValueError, 'file SHA256 differs'):
                ts.verify_source(root, manifest, sha)
            runner.write_bytes(previous)
            (root / 'tetriserve/unlisted.py').write_text('raise RuntimeError("injection")')
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                ts.verify_source(root, manifest, sha)

    def test_wrong_commit_path_escape_and_symlink_rejected(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            original = json.loads(manifest.read_text())
            for change in ('commit', 'escape'):
                record = json.loads(json.dumps(original))
                if change == 'commit':
                    record['commit'] = '0' * 40
                else:
                    record['files']['../outside.py'] = '0' * 64
                manifest.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    ts.verify_source(root, manifest, ts.digest(manifest.read_bytes()))
            manifest.write_text(json.dumps(original))
            (root / 'linked').symlink_to(shared, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                ts.verify_source(root, manifest, ts.digest(manifest.read_bytes()))

    def test_real_subprocess_imports_and_missing_dependency_are_distinct(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            result = ts.import_probe(root, shared, timeout=30)
            self.assertEqual(result['status'], 'imports_passed')
            self.assertTrue(all(v['status'] == 'imported' for v in result['imports']))
            self.assertFalse(result['serving_executed'])
            ts.verify_source(root, manifest, sha)
            # This is a real failing import in a fresh process, not a mocked success.
            (root / 'tetriserve/server/launcher.py').write_text('import ditserve_test_missing_dependency_20261103\n')
            result = ts.import_probe(root, shared, timeout=30)
            self.assertEqual(result['status'], 'imports_failed')
            launcher = next(i for i in result['imports'] if i['module'] == 'tetriserve.server.launcher')
            self.assertEqual(launcher['error_type'], 'ModuleNotFoundError')
            self.assertIn('ditserve_test_missing_dependency_20261103', launcher['error'])

    def test_cli_never_returns_success_for_unsupported_formal_models(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = ts.main(['audit', '--source-root', str(root), '--source-manifest', str(manifest),
                                '--source-manifest-sha256', sha, '--shared-xfuser-root', str(shared)])
            result = json.loads(output.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(result['status'], 'blocked_model_backend_port')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = ts.main(['audit', '--source-root', str(root), '--source-manifest', str(manifest),
                                '--source-manifest-sha256', '0' * 64, '--shared-xfuser-root', str(shared)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())['status'], 'invalid_evidence')

    def test_nonfinite_duplicate_json_and_invalid_timeout_rejected(self):
        for value in ['{"x": NaN}', '{"x": 1, "x": 2}']:
            with self.assertRaises(ValueError):
                ts.read_json(value)
        for value in [True, 0, -1, 301, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                ts.import_probe('.', '.', timeout=value)


if __name__ == '__main__':
    unittest.main()
