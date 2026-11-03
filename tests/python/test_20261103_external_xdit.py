"""Source provenance, backend differences and real isolated xDiT import failures."""
import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import external_xdit as xd


class ExternalXDiTTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        root, shared = base / "official", base / "shared" / "xfuser"
        files = {
            "xfuser/__init__.py": "# isolated upstream fixture\n",
            "xfuser/runner.py": "MODEL_REGISTRY = {'Wan2.1': object}\n",
            "xfuser/cli.py": "def main(): raise RuntimeError('must not launch')\n",
            "entrypoints/launch.py": "# imports only; no service launched\n",
            xd.RUNNER_FILES[0]: "def register_model(n): return lambda c: c\n@register_model('" + xd.MODEL_IDS['wan21'] + "')\nclass Wan21: pass\n@register_model('" + xd.MODEL_IDS['wan22'] + "')\nclass Wan22: pass\n",
            xd.RUNNER_FILES[1]: "def register_model(n): return lambda c: c\n@register_model('stabilityai/stable-diffusion-3.5-large')\nclass SD35: pass\n",
            "xfuser/extra.py": "# officially available, not vendored\n",
        }
        for name, contents in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        shared_files = {
            "__init__.py": "# modified serving fork\n",
            "runner.py": files["xfuser/runner.py"],
            "model_executor/models/transformers/transformer_wan.py": "# shared Wan transformer\n",
            "model_executor/models/transformers/transformer_sd3.py": "# shared SD3 transformer\n",
            "model_executor/pipelines/pipeline_stable_diffusion_3.py": "# shared pipeline\n",
        }
        for name, contents in shared_files.items():
            path = shared / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        manifest = base / "source.json"
        value = {"schema": "ditserve.official-source-archive.v1", "repository": xd.REPOSITORY,
                 "commit": xd.COMMIT, "archive_sha256": xd.ARCHIVE_SHA256,
                 "files": {n: xd.digest((root / n).read_bytes()) for n in files}}
        manifest.write_text(json.dumps(value))
        return root, manifest, xd.digest(manifest.read_bytes()), shared

    def test_full_inventory_and_backend_fork_differences(self):
        with TemporaryDirectory() as directory:
            args = self.fixture(directory)
            result = xd.audit(*args)
            diff = result['shared_xfuser']
            self.assertEqual(diff['modified_count'], 1)
            self.assertEqual(diff['identical_count'], 1)
            self.assertEqual(diff['omitted_count'], 4)
            self.assertEqual(diff['shared_only_count'], 3)
            self.assertFalse(diff['identical_python_source'])
            self.assertFalse(diff['numerical_equivalence_verified'])
            self.assertFalse(diff['revision_label_is_sufficient_identity'])
            self.assertTrue(result['current_models']['wan21']['registered'])
            self.assertTrue(result['current_models']['wan22']['registered'])
            self.assertFalse(result['current_models']['sd35']['registered'])
            self.assertIn('task=t2v', result['current_models']['wan22']['native_parallelism'])
            self.assertFalse(result['qualification_eligible'])
            self.assertFalse(result['serving_executed'])
            self.assertEqual(result['measurements'], [])

    def test_manifest_source_modification_and_extra_files_fail_closed(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'manifest SHA256 differs'):
                xd.verify_source(root, manifest, '0' * 64)
            original = (root/'xfuser/runner.py').read_bytes()
            (root/'xfuser/runner.py').write_bytes(original + b'# changed')
            with self.assertRaisesRegex(ValueError, 'file SHA256 differs'):
                xd.verify_source(root, manifest, sha)
            (root/'xfuser/runner.py').write_bytes(original)
            (root/'xfuser/unlisted.py').write_text('raise RuntimeError("unexpected")')
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                xd.verify_source(root, manifest, sha)

    def test_wrong_identity_traversal_duplicate_json_and_symlink_fail_closed(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            original = json.loads(manifest.read_text())
            for field, value in [('commit', '0' * 40), ('repository', 'unofficial/xDiT'),
                                 ('archive_sha256', '0' * 64)]:
                record = dict(original, **{field: value})
                manifest.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    xd.verify_source(root, manifest, xd.digest(manifest.read_bytes()))
            original['files']['../outside.py'] = '0' * 64
            manifest.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'invalid source inventory'):
                xd.verify_source(root, manifest, xd.digest(manifest.read_bytes()))
            (shared/'outside').symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                xd.python_inventory(shared)
        for raw in ['{"a": 1, "a": 2}', '{"a": NaN}']:
            with self.assertRaises(ValueError):
                xd.read_json(raw)

    def test_actual_fresh_imports_do_not_mix_upstream_and_shared(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            for location, mode in [(root, 'official'), (shared, 'shared')]:
                result = xd.import_one(location, mode, timeout=30)
                self.assertEqual(result['status'], 'imports_passed', result)
                self.assertFalse(result['source_violations'])
                self.assertTrue(all(i['status'] == 'imported' for i in result['imports']))
                expected = root/'xfuser' if mode == 'official' else shared
                self.assertTrue(all(Path(i['path']).is_relative_to(expected)
                    for i in result['loaded_xfuser_modules'].values()))
                self.assertFalse(result['serving_executed'])
            xd.verify_source(root, manifest, sha)

    def test_missing_dependency_and_partial_import_are_not_success(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            (root/'entrypoints/launch.py').write_text('import ditserve_missing_xdit_dependency_20261103\n')
            result = xd.import_one(root, 'official', timeout=30)
            self.assertEqual(result['status'], 'imports_failed')
            failed = next(i for i in result['imports'] if i['module'] == 'entrypoints.launch')
            self.assertEqual(failed['error_type'], 'ModuleNotFoundError')
            self.assertIn('ditserve_missing_xdit_dependency_20261103', failed['error'])
            self.assertTrue(any(i['status'] == 'imported' for i in result['imports']))

    def test_cli_qualification_and_shared_inventory_pin(self):
        with TemporaryDirectory() as directory:
            root, manifest, sha, shared = self.fixture(directory)
            command = ['audit', '--source-root', str(root), '--source-manifest', str(manifest),
                       '--source-manifest-sha256', sha, '--shared-xfuser-root', str(shared)]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = xd.main(command)
            self.assertEqual(code, 2)
            self.assertFalse(json.loads(output.getvalue())['qualification_eligible'])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = xd.main(command + ['--shared-xfuser-inventory-sha256', '0' * 64])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())['status'], 'invalid_evidence')

    def test_invalid_timeout_and_mode_rejected_before_subprocess(self):
        for timeout in [True, False, 0, -1, 301, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                xd.import_one('.', 'official', timeout=timeout)
        with self.assertRaises(ValueError):
            xd.import_one('.', 'mixed')


if __name__ == '__main__':
    unittest.main()
