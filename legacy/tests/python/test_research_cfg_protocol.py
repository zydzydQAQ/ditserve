import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research import cfg_protocol as protocol
from ditserve.research import dual_protocol
from ditserve.research.cfg_policy import resolve_policy, verify_policy
from ditserve.research.common import atomic_json, digest, read_json


class CfgPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(protocol.DEFAULT_CONFIG)

    def test_catalog_is_explicit_and_unknown_legacy_alias_rejected(self):
        expected = {
            "off": (False, 0.0, "independent", "native"),
            "independent_008": (True, .08, "independent", "native"),
            "independent_004": (True, .04, "independent", "native"),
            "paired_008": (True, .08, "paired", "oracle"),
        }
        for key, values in expected.items():
            policy = resolve_policy(self.config, key)
            self.assertEqual(tuple(policy[k] for k in ("cache", "threshold", "cfg_mode", "implementation")), values)
            self.assertEqual(policy["max_continuous"], 1)
            self.assertEqual(policy["policy_id"], policy["id"])
            self.assertEqual(verify_policy(self.config, policy), policy)
        for key in ("t008", "unknown", "", None):
            with self.subTest(policy=key), self.assertRaisesRegex(ValueError, "unknown CFG policy"):
                resolve_policy(self.config, key)

    def test_implementation_changes_identity_and_does_not_qualify_it(self):
        oracle = resolve_policy(self.config, "paired_008")
        split = resolve_policy(self.config, "paired_008", "split_prefix")
        self.assertNotEqual(oracle["sha256"], split["sha256"])
        self.assertEqual(oracle["sha256"], digest({k: v for k, v in oracle.items() if k != "sha256"}))
        self.assertNotIn("qualified", split)
        for policy, implementation in (("paired_008", "native"), ("off", "oracle"),
                                       ("independent_004", "split_prefix"), ("paired_008", "optimized")):
            with self.subTest(policy=policy, implementation=implementation), self.assertRaisesRegex(ValueError, "implementation"):
                resolve_policy(self.config, policy, implementation)

    def test_returned_identity_is_independent_and_tampered_identity_rejected(self):
        before = copy.deepcopy(self.config)
        original = resolve_policy(self.config, "independent_004")
        altered = resolve_policy(self.config, "independent_004")
        altered["threshold"] = .08
        altered["sha256"] = digest({k: v for k, v in altered.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "identity/checksum"):
            verify_policy(self.config, altered)
        self.assertEqual(resolve_policy(self.config, "independent_004"), original)
        self.assertEqual(self.config, before)
        with self.assertRaisesRegex(ValueError, "dictionary"):
            verify_policy(self.config, None)

    def test_catalog_drift_and_numeric_bool_aliases_are_rejected(self):
        for field, value in (("id", "t008"), ("threshold", .03), ("max_continuous", True),
                             ("cache", 1), ("cfg_mode", "joint")):
            config = copy.deepcopy(self.config)
            config["recipes"][2][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_policy(config, "off")
        for recipes in (self.config["recipes"][:-1], self.config["recipes"][::-1],
                        self.config["recipes"] + [self.config["recipes"][-1]]):
            with self.assertRaises(ValueError):
                resolve_policy(dict(self.config, recipes=recipes), "off")


class CfgProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(protocol.DEFAULT_CONFIG)
        self.legacy = read_json(dual_protocol.DEFAULT_CONFIG)

    def test_recipe_locks_devices_model_shape_samples_and_budget(self):
        self.assertEqual(protocol.validate_protocol(self.config), self.config)
        self.assertEqual(dual_protocol.validate_protocol(self.config), self.config)
        self.assertEqual(self.config["attention"], {"A100": "flash", "V100": "efficient"})
        self.assertEqual(self.config["shapes"]["main"], dict(height=256, width=448, frames=17, steps=32))
        self.assertEqual(self.config["shapes"]["main"], self.config["shapes"]["fallback"])
        self.assertEqual(self.config["screening_inputs"], [dict(prompt_id="origami", seed=101),
            dict(prompt_id="origami", seed=102), dict(prompt_id="kettle", seed=101)])
        self.assertEqual(self.config["remaining_budget_shares"], self.legacy["remaining_budget_shares"])
        self.assertEqual(self.config["prompts"], self.legacy["prompts"])
        self.assertEqual(self.config["phases"], list(protocol.PHASES))
        self.assertEqual(self.config["migration_actions"], ["stay", "full", "cold"])
        self.assertIn("not_transferable", self.config["capacity"])
        for field, value in (("revision", "wrong"), ("model_path", "/tmp/model"),
                             ("artifact_root", "/tmp/raw"), ("budget_gpu_seconds", 30000),
                             ("attention", {"A100": "flash", "V100": "math"}),
                             ("cost_safety_factor", 1.0), ("screening_inputs", []),
                             ("shapes", self.legacy["shapes"]), ("retention_policy", "all")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "fixed CFG"):
                protocol.validate_protocol(dict(self.config, **{field: value}))

    def test_backend_has_no_silent_fallback(self):
        self.assertEqual(protocol.attention_for_device(dict(name="NVIDIA A100", compute_capability="8.0")), "flash")
        self.assertEqual(protocol.attention_for_device(dict(name="Tesla V100", compute_capability="7.0")), "efficient")
        with self.assertRaises(ValueError):
            protocol.attention_for_device(dict(name="Tesla V100", compute_capability="8.0"))

    def test_only_positive_stage_estimates_can_be_adjusted(self):
        config = copy.deepcopy(self.config)
        config["estimate_wall_seconds"]["calibration"] = 180
        config["timeout_wall_seconds"]["calibration"] = 181
        config["cleanup_wall_seconds"] = 20
        self.assertEqual(protocol.validate_protocol(config), config)
        for value in (0, -1, float("nan"), float("inf")):
            config["estimate_wall_seconds"]["calibration"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.validate_protocol(config)

    def test_new_freeze_has_separate_identity_and_old_archive_remains_verifiable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_config = root / "configs/research_two_gpu.json"
            new_config = root / "configs/research_cfg.json"
            atomic_json(old_config, self.legacy)
            source = root / "python/ditserve/fixture.py"
            source.parent.mkdir(parents=True)
            source.write_text("version = 1\n")
            ledger = root / "existing-ledger.json"
            ledger.write_text('{"historical_gpu_seconds":12196.38427163777,"active":null}\n')
            ledger_bytes = ledger.read_bytes()
            manifest = {"sha256": "fixture-model-manifest"}
            with patch.object(dual_protocol, "verify_model_manifest", return_value=manifest), \
                    patch.object(dual_protocol, "dependency_identity", return_value={}), \
                    patch.object(dual_protocol, "quality_assets", return_value={}):
                old_output = root / "results/old-run"
                old = dual_protocol.freeze_protocol(old_config, old_output, ledger, model_manifest=manifest, root=root)
                atomic_json(new_config, self.config)
                new_output = root / "results/cfg-run"
                new = protocol.freeze_protocol(new_config, new_output, ledger, model_manifest=manifest, root=root)
                self.assertEqual(new["schema_version"], protocol.FROZEN_SCHEMA)
                self.assertNotEqual(new["sha256"], old["sha256"])
                self.assertIn("configs/research_cfg.json", new["source_files"])
                self.assertEqual(protocol.verify_protocol(new_output / "protocol.json", root=root), new)
                self.assertEqual(protocol.freeze_protocol(new_config, new_output, ledger, model_manifest=manifest, root=root), new)
                self.assertEqual(dual_protocol.verify_protocol(old_output / "protocol.json", root=root, verify_current=False), old)
                with self.assertRaisesRegex(ValueError, "new CFG frozen"):
                    protocol.verify_protocol(old_output / "protocol.json", root=root, verify_current=False)
                self.assertEqual(ledger.read_bytes(), ledger_bytes)
                source.write_text("version = 2\n")
                with self.assertRaisesRegex(ValueError, "current implementation"):
                    protocol.verify_protocol(new_output / "protocol.json", root=root)
                self.assertEqual(protocol.verify_protocol(new_output / "protocol.json", root=root, verify_current=False), new)
                forged = copy.deepcopy(new)
                forged["schema_version"] = dual_protocol.FROZEN_SCHEMA
                forged["sha256"] = digest({k: v for k, v in forged.items() if k != "sha256"})
                atomic_json(new_output / "protocol.json", forged)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    dual_protocol.verify_protocol(new_output / "protocol.json", root=root, verify_current=False)


if __name__ == "__main__":
    unittest.main()
