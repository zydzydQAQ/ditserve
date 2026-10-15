"""Small offline fixtures exercise provisioning integrity and restart behavior."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.migration import models


class ModelMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "models"
        self.root.mkdir()
        self.lock_path = self.base / "models.lock.json"
        self.assets = {"config.json": b'{"model_type":"test"}',
                       "model-1.safetensors": b"locked model fixture",
                       "model.safetensors.index.json": b'{"weight_map":{"x":"model-1.safetensors"}}'}
        self.lock = {"schema_version": "ditserve.migration.models.v1", "models": [
            {"name": "wan", "group": "core", "directory": "wan", "repo_id": "test/model",
             "revision": "a" * 40, "files": [self.item(name, value) for name, value in self.assets.items()]},
            {"name": "mistral", "group": "sessions", "directory": "mistral", "repo_id": "test/session",
             "revision": "b" * 40, "files": [self.item("config.json", b"{}")]}]}
        self.save_lock()

    def item(self, path, value):
        return {"path": path, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}

    def save_lock(self):
        self.lock_path.write_text(json.dumps(self.lock))

    def install_core(self):
        (self.root / "wan").mkdir(exist_ok=True)
        for name, value in self.assets.items():
            (self.root / "wan" / name).write_bytes(value)

    def verify(self, groups=("core",)):
        return models.verify_models(self.root, groups, lock_path=self.lock_path)

    def test_missing_reports_every_required_file(self):
        result = self.verify()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["errors"]), 3)
        self.assertTrue(all(error["reason"] == "missing" for error in result["errors"]))

    def test_core_does_not_require_optional_models_or_write(self):
        self.install_core()
        result = self.verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual([m["name"] for m in result["models"]], ["wan"])
        self.assertFalse((self.root / ".provisioning").exists())
        self.assertFalse(result["gpu_work_started"])

    def test_explicit_optional_group_is_verified(self):
        self.install_core()
        result = self.verify(("core", "sessions"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"][0]["model"], "mistral")

    def test_same_size_corruption_and_size_mismatch_fail(self):
        self.install_core()
        (self.root / "wan/config.json").write_bytes(b"x" * len(self.assets["config.json"]))
        (self.root / "wan/model-1.safetensors").write_bytes(b"truncated")
        self.assertEqual({e["reason"] for e in self.verify()["errors"]}, {"sha256_mismatch", "size_mismatch"})

    def test_misleading_verified_manifest_does_not_override_content(self):
        self.install_core()
        directory = self.root / ".provisioning"
        directory.mkdir()
        (directory / "wan.json").write_text('{"status":"verified","revision":"wrong"}')
        (self.root / "wan/config.json").write_bytes(b"wrong revision")
        self.assertEqual(self.verify()["status"], "failed")

    def test_index_cannot_reference_unlocked_shard(self):
        self.install_core()
        value = b'{"weight_map":{"x":"unlocked.safetensors"}}'
        self.lock["models"][0]["files"][2] = self.item("model.safetensors.index.json", value)
        self.save_lock()
        (self.root / "wan/model.safetensors.index.json").write_bytes(value)
        self.assertEqual(self.verify()["errors"][0]["reason"], "shard_not_in_lock")

    def test_unlisted_weights_and_loader_overrides_fail_before_gpu_work(self):
        self.install_core()
        for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json",
                     "tokenizer.json", "added_tokens.json", "generation_config.json",
                     "chat_template.jinja"):
            with self.subTest(name=name):
                extra = self.root / "wan" / name
                extra.write_bytes(b"unlocked override")
                result = self.verify()
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["gpu_work_started"])
                self.assertEqual(result["errors"], [{"model": "wan", "path": name,
                                                   "reason": "unexpected_loader_file"}])
                extra.unlink()

    def test_component_loader_directories_are_checked_for_wan_sana_and_mistral(self):
        for name, component, group in (("wan", "text_encoder", "core"),
                                       ("sana", "tokenizer", "core"),
                                       ("mistral", "", "sessions")):
            with self.subTest(name=name):
                prefix = component + "/" if component else ""
                assets = {prefix + path: value for path, value in self.assets.items()}
                self.lock["models"] = [{"name": name, "group": group, "directory": name,
                    "repo_id": "test/model", "revision": "a" * 40,
                    "files": [self.item(path, value) for path, value in assets.items()]}]
                self.save_lock()
                for path, value in assets.items():
                    target = self.root / name / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(value)
                extra = self.root / name / component / "model.safetensors"
                extra.write_bytes(b"preferred over locked shards")
                result = self.verify((group,))
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["errors"][0]["path"], prefix + "model.safetensors")
                self.assertEqual(result["errors"][0]["reason"], "unexpected_loader_file")

    def test_documentation_metadata_and_cache_are_harmless(self):
        self.install_core()
        directory = self.root / "wan"
        for name in ("README.md", "LICENSE", "verification.json"):
            (directory / name).write_text("metadata")
        (directory / ".cache").mkdir()
        (directory / ".cache/model.safetensors").write_bytes(b"not a loader input")
        self.assertEqual(self.verify()["status"], "verified")

    def test_shared_lpips_and_explicit_longlive_paths_allow_unrelated_weights(self):
        for name in ("lpips", "longlive"):
            with self.subTest(name=name):
                model = dict(self.lock["models"][0], name=name)
                self.install_core()
                (self.root / "wan/unrelated.pth").write_bytes(b"shared asset")
                result = models._inspect_model(self.root, model)
                self.assertEqual(result["status"], "verified")

    def test_lock_rejects_traversal_duplicate_and_floating_revision(self):
        original = json.loads(json.dumps(self.lock))
        for mode in ("traversal", "duplicate", "revision"):
            self.lock = json.loads(json.dumps(original))
            if mode == "traversal":
                self.lock["models"][0]["files"][0]["path"] = "../escape"
            elif mode == "duplicate":
                self.lock["models"][0]["files"].append(self.lock["models"][0]["files"][0])
            else:
                self.lock["models"][0]["revision"] = "main"
            self.save_lock()
            with self.assertRaises(ValueError):
                self.verify()

    def test_invalid_group_is_error(self):
        with self.assertRaises(ValueError):
            self.verify(("cor",))

    def test_reuse_publishes_native_manifest_without_network(self):
        self.install_core()
        with patch.object(models, "_download_url", side_effect=AssertionError("network accessed")):
            result = models.download_models(self.root, lock_path=self.lock_path)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["models"][0]["reused_files"], 3)
        manifest = json.loads((self.root / ".provisioning/wan.json").read_text())
        self.assertEqual(manifest["status"], "verified")
        self.assertEqual(manifest["model"]["path"], str(self.root / "wan"))
        self.assertEqual(len(manifest["files"]), 3)
        self.assertEqual(manifest["lock_sha256"], models._sha256(self.lock_path))

    def test_failed_repair_invalidates_previous_ready_manifest(self):
        self.install_core()
        models.download_models(self.root, lock_path=self.lock_path)
        (self.root / "wan/config.json").unlink()
        with patch.object(models, "_download_file", side_effect=OSError("private token must not be logged")):
            result = models.download_models(self.root, lock_path=self.lock_path)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("private token", json.dumps(result))
        self.assertEqual(json.loads((self.root / ".provisioning/wan.json").read_text())["status"], "pending")

    def test_valid_snapshot_symlink_is_reused_and_corrupt_one_is_not_modified(self):
        outside = self.base / "existing"
        outside.mkdir()
        for name, value in self.assets.items():
            (outside / name).write_bytes(value)
        (self.root / "wan").symlink_to(outside, target_is_directory=True)
        self.assertEqual(models.download_models(self.root, lock_path=self.lock_path)["status"], "verified")
        (outside / "config.json").write_bytes(b"corrupt")
        result = models.download_models(self.root, lock_path=self.lock_path)
        self.assertEqual(result["status"], "failed")
        self.assertEqual((outside / "config.json").read_bytes(), b"corrupt")

    def test_longlive_download_uses_locked_remote_mapping(self):
        model = {"directory": "longlive/weights", "revision": "a" * 40}
        item = dict(self.item("longlive_models/models/base.pt", b"checkpoint"),
                    repo_id="test/longlive", revision="b" * 40,
                    download_directory="longlive_models", remote_path="models/base.pt")
        calls = []
        def download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"checkpoint")
            return str(target)
        import types
        hub = types.SimpleNamespace(hf_hub_download=download)
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            self.assertEqual(models._download_file(self.root, model, item), "downloaded")
        self.assertEqual(calls[0]["revision"], "b" * 40)
        self.assertEqual(calls[0]["filename"], "models/base.pt")
        self.assertNotIn("token", calls[0])

    def test_url_resume_validates_and_atomically_publishes(self):
        import io
        data = b"complete asset bytes"
        item = self.item("alex.pth", data)
        target = self.root / "alex.pth"
        partial = target.with_name(target.name + "." + item["sha256"] + ".partial")
        partial.write_bytes(data[:4])
        response = io.BytesIO(data[4:])
        response.status = 206
        response.headers = {"Content-Range": "bytes 4-%d/%d" % (len(data) - 1, len(data))}
        with patch("urllib.request.urlopen", return_value=response) as opened:
            models._download_url("https://example.org/alex.pth", target, item)
        self.assertEqual(opened.call_args.args[0].get_header("Range"), "bytes=4-")
        self.assertEqual(target.read_bytes(), data)
        self.assertFalse(partial.exists())

    def test_url_bad_hash_is_not_published_and_partial_can_retry(self):
        import io
        item = self.item("alex.pth", b"good")
        target = self.root / "alex.pth"
        response = io.BytesIO(b"evil")
        response.status, response.headers = 200, {}
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(ValueError):
                models._download_url("https://example.org/alex.pth", target, item)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_longlive_missing_interpreter_is_failed_without_claiming_cpu_success(self):
        self.lock["longlive_source"] = {"revision": "a" * 40}
        self.save_lock()
        result = models.validate_longlive_environment(self.root, python="/missing/longlive/python",
                                                      lock_path=self.lock_path)
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.root / ".provisioning/longlive-environment.json").exists())

    def test_longlive_assets_do_not_reuse_old_environment_proof(self):
        self.install_core()
        self.lock["models"][0].update(name="longlive", group="longlive")
        self.lock["longlive_source"] = {"revision": "c" * 40}
        self.save_lock()
        directory = self.root / ".provisioning"
        directory.mkdir()
        (directory / "longlive.json").write_text(json.dumps({"status": "verified",
            "environment_validation": {"status": "passed", "python": "/old/host/python"}}))
        with patch.object(models, "validate_longlive_environment", return_value={
                "status": "failed", "reason": "unavailable"}) as validate:
            result = models.download_models(self.root, ("longlive",), lock_path=self.lock_path,
                                           longlive_python="/new/python", source_path="/new/source")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(validate.call_args.kwargs["python"], "/new/python")
        manifest = json.loads((directory / "longlive.json").read_text())
        self.assertEqual(manifest["environment_validation"]["status"], "failed")
        self.assertNotIn("/old/host", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
