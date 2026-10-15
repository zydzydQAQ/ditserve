import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research import dual_protocol as protocol


class DualProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "snapshots" / protocol.REVISION
        self.artifacts = self.root / "artifacts"
        self.config = read_json(protocol.DEFAULT_CONFIG)
        self.config.update(model_path=str(self.model), artifact_root=str(self.artifacts))
        self.config_path = self.root / "configs/research_two_gpu.json"
        atomic_json(self.config_path, self.config)
        for field, value in (("MODEL_PATH", self.model), ("ARTIFACT_ROOT", self.artifacts),
                             ("DEFAULT_CONFIG", self.config_path)):
            mocked = patch.object(protocol, field, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.make_model()

    def make_model(self):
        for name in protocol.REQUIRED_METADATA:
            target = self.model / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"{}" if name.endswith(".json") else b"tokenizer fixture")
        atomic_json(self.model / "model_index.json", dict(_class_name="WanPipeline", **protocol.MODEL_COMPONENTS))
        atomic_json(self.model / "transformer/config.json", dict(_class_name="WanTransformer3DModel",
            num_layers=30, num_attention_heads=12, attention_head_dim=128,
            in_channels=16, out_channels=16, patch_size=[1, 2, 2]))
        for component, (index, prefix, count) in protocol.WEIGHT_GROUPS.items():
            weights = {}
            for i in range(1, count + 1):
                name = "%s-%05d-of-%05d.safetensors" % (prefix, i, count)
                (self.model / component / name).write_bytes(bytes([i]) * 128)
                weights["tensor_%d" % i] = name
            atomic_json(self.model / component / index, dict(metadata=dict(total_size=count * 64), weight_map=weights))
        (self.model / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"v" * 128)

    def test_protocol_rejects_scientific_drift_and_model_fallback(self):
        protocol.validate_protocol(self.config)
        for key, value in (("steps", 31), ("model_path", "/tmp/other"), ("retention_policy", "all"),
                           ("transition_step", 15), ("budget_gpu_seconds", 28801),
                           ("parallel", dict(degrees=[2]))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_protocol(dict(self.config, **{key: value}))
        with self.assertRaisesRegex(ValueError, "no fallback"):
            protocol.inspect_model_snapshot(self.root / "other")
        with self.assertRaisesRegex(ValueError, "no fallback"):
            protocol.inspect_model_snapshot(self.model.parent / "wrong-revision")

    def test_shapes_splits_recipes_and_attention_are_explicit(self):
        self.assertEqual(self.config["shapes"]["main"], dict(height=480, width=832, frames=17, steps=32))
        self.assertEqual(self.config["shapes"]["fallback"], dict(height=256, width=448, frames=17, steps=32))
        self.assertEqual([p["id"] for p in self.config["prompts"] if p["split"] == "holdout"], ["tram", "aquarium"])
        self.assertEqual(len([p for p in self.config["prompts"] if p["split"] == "dev"]), 4)
        self.assertEqual([r["id"] for r in self.config["recipes"]], ["off", "t008"])
        self.assertEqual(protocol.attention_for_device(dict(name="NVIDIA A100-SXM4-40GB", compute_capability="8.0")), "flash")
        v100 = dict(name="Tesla V100-SXM2-32GB", compute_capability="7.0")
        self.assertEqual(protocol.attention_for_device(v100), "math")
        self.assertEqual(protocol.attention_for_device(v100, efficient_qualified=True), "efficient")
        with self.assertRaises(ValueError):
            protocol.attention_for_device(dict(name="A100", compute_capability="7.0"))

    def test_discovery_does_not_hash_weights_or_claim_freeze_eligibility(self):
        with patch.object(protocol, "file_digest", side_effect=AssertionError("hashing not allowed")):
            manifest = protocol.inspect_model_snapshot()
        self.assertEqual(manifest["status"], "inventory_only")
        self.assertTrue(all(row["sha256"] is None for row in manifest["files"].values()))
        with self.assertRaisesRegex(ValueError, "complete bound"):
            protocol.verify_model_manifest(manifest)

    def test_hash_manifest_binds_all_weights_and_rejects_drift(self):
        manifest = protocol.inspect_model_snapshot(hash_files=True)
        self.assertEqual(len([n for n in manifest["files"] if n.endswith(".safetensors")]), 8)
        self.assertEqual(protocol.verify_model_manifest(manifest), manifest)
        self.assertEqual(protocol.verify_model_manifest(manifest, rehash=True), manifest)
        shard = self.model / "text_encoder/model-00001-of-00005.safetensors"
        shard.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            protocol.verify_model_manifest(manifest)

    def test_missing_unknown_and_unsafe_shards_fail_closed(self):
        extra = self.model / "transformer/diffusion_pytorch_model.bin"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "weight file set"):
            protocol.inspect_model_snapshot()
        extra.unlink()
        path = self.model / "text_encoder/model.safetensors.index.json"
        index = read_json(path)
        index["weight_map"]["tensor_1"] = "../../outside.safetensors"
        atomic_json(path, index)
        with self.assertRaisesRegex(ValueError, "unsafe model shard"):
            protocol.inspect_model_snapshot()

    def test_missing_or_empty_model_files_and_wrong_architecture_fail_closed(self):
        path = self.model / "vae/diffusion_pytorch_model.safetensors"
        path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            protocol.inspect_model_snapshot()
        path.write_bytes(b"restored")
        config = read_json(self.model / "transformer/config.json")
        config["num_layers"] = 40
        atomic_json(self.model / "transformer/config.json", config)
        with self.assertRaisesRegex(ValueError, "architecture"):
            protocol.inspect_model_snapshot()

    def test_manifest_mutation_and_change_during_hash_are_rejected(self):
        manifest = protocol.inspect_model_snapshot(hash_files=True)
        manifest["files"]["model_index.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bound SHA256"):
            protocol.verify_model_manifest(manifest)
        original = protocol.file_digest
        def changing(path):
            result = original(path)
            if path.name == "model_index.json":
                path.write_text(path.read_text() + " ")
            return result
        with patch.object(protocol, "file_digest", side_effect=changing), self.assertRaisesRegex(ValueError, "changed while"):
            protocol.inspect_model_snapshot(hash_files=True)

    def test_freeze_archives_sources_binds_manifest_and_never_touches_ledger(self):
        source = self.root / "python/ditserve/a.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 1\n")
        ledger = self.root / "shared-ledger.json"
        ledger.write_text('{"active": "running; must not be touched"}\n')
        before = ledger.read_bytes()
        output = self.artifacts / "run01"
        manifest = protocol.inspect_model_snapshot(hash_files=True)
        with patch.object(protocol, "dependency_identity", return_value={}), patch.object(protocol, "quality_assets", return_value={}):
            frozen = protocol.freeze_protocol(self.config_path, output, ledger, root=self.root, model_manifest=manifest)
            self.assertEqual(ledger.read_bytes(), before)
            self.assertEqual(protocol.verify_protocol(output / "protocol.json", root=self.root), frozen)
            self.assertEqual(protocol.freeze_protocol(self.config_path, output, ledger, root=self.root, model_manifest=manifest), frozen)
            source.write_text("value = 2\n")
            with self.assertRaisesRegex(ValueError, "current implementation"):
                protocol.verify_protocol(output / "protocol.json", root=self.root)
            self.assertEqual(protocol.verify_protocol(output / "protocol.json", root=self.root, verify_current=False), frozen)
            (output / "sources/python/ditserve/a.py").write_text("archive drift")
            with self.assertRaisesRegex(ValueError, "archive mismatch"):
                protocol.verify_protocol(output / "protocol.json", root=self.root, verify_current=False)

    def test_freeze_requires_complete_manifest_and_deduplicates_structured_result_sources(self):
        with self.assertRaisesRegex(ValueError, "complete bound"):
            protocol.freeze_protocol(self.config_path, self.artifacts / "run", model_manifest=protocol.inspect_model_snapshot())
        manifest = protocol.inspect_model_snapshot(hash_files=True)
        with patch.object(protocol, "dependency_identity", return_value={}), patch.object(protocol, "quality_assets", return_value={}):
            output = self.root / "results/research-two-gpu"
            first = protocol.freeze_protocol(self.config_path, output / "run01", model_manifest=manifest, root=self.root)
            second = protocol.freeze_protocol(self.config_path, output / "run02", model_manifest=manifest, root=self.root)
            self.assertEqual(first, second)
            self.assertEqual((output / "run01/sources").resolve(), (output / "run02/sources").resolve())
            self.assertEqual(len(list((output / "source-snapshots").iterdir())), 1)
            self.assertEqual(protocol.verify_protocol(output / "run02/protocol.json", root=self.root), second)


if __name__ == "__main__":
    unittest.main()
