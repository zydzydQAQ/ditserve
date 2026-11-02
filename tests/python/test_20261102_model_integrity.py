"""Frozen model inventory must be verified before granting weight identity."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ditserve.runtime.model_integrity import verify_model_manifest
from ditserve.runtime.worker import Worker, bind_generation_identity


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        (self.root / "weights.bin").write_bytes(b"weights")
        self.manifest = Path(self.temp.name) / "manifest.json"
        self.data = {"schema": "ditserve.model-manifest.v1", "revision": "frozen-upstream-revision",
                     "files": [{"path": "weights.bin", "bytes": 7,
                                "sha256": hashlib.sha256(b"weights").hexdigest()}]}

    def verify(self):
        self.manifest.write_text(json.dumps(self.data))
        return verify_model_manifest(self.root, self.manifest, hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_success_binds_real_verification(self):
        proof = self.verify()
        result = {"results": [{"generation": {}}]}
        bind_generation_identity(result, {}, proof)
        self.assertEqual(result["results"][0]["generation"]["model_integrity"], "manifest_verified")
        self.assertEqual(proof["model_verified_files"], 1)

    def test_same_size_corruption_fails(self):
        (self.root / "weights.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.verify()

    def test_missing_and_unlisted_files_fail(self):
        (self.root / "config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "complete snapshot"):
            self.verify()
        (self.root / "weights.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_manifest_tampering_and_traversal_fail(self):
        self.verify()
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            verify_model_manifest(self.root, self.manifest, "0" * 64)
        self.data["files"][0]["path"] = "../weights.bin"
        with self.assertRaisesRegex(ValueError, "file path"):
            self.verify()

    def test_external_symlink_and_duplicate_fail(self):
        target = Path(self.temp.name) / "outside"
        target.write_bytes(b"weights")
        (self.root / "weights.bin").unlink()
        (self.root / "weights.bin").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.verify()
        (self.root / "weights.bin").unlink()
        (self.root / "weights.bin").write_bytes(b"weights")
        self.data["files"].append(dict(self.data["files"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify()

    def test_worker_uses_verified_revision_and_rejects_a_conflicting_declaration(self):
        proof = self.verify()
        config = dict(model_path=str(self.root), model_manifest_path=str(self.manifest),
                      model_manifest_sha256=proof['model_manifest_sha256'])
        loaded = []
        class Backend:
            def __init__(self, options):
                loaded.append(options['revision'])
            def sample(self, request):
                return {'generation': {'revision': loaded[-1]}}
        worker = Worker('wan', config)
        result = worker._model_call(Backend, 'sample', {})
        self.assertEqual(loaded, [self.data['revision']])
        self.assertEqual(result['generation']['revision'], result['generation']['model_verified_revision'])
        self.assertEqual(result['generation']['model_path'], str(self.root))
        conflicting = Worker('wan', dict(config, revision='another-revision'))
        with self.assertRaisesRegex(ValueError, 'revision differs'):
            conflicting._model_call(Backend, 'sample', {})
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
