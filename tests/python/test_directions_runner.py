import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ditserve.directions.protocol import load_config
from ditserve.directions.budget import Allocation, BudgetStop
from ditserve.directions.runner import execute, inspect, phase_result, report, model_inventory


class SupervisorTests(unittest.TestCase):
    def test_occupied_gpu_never_starts_worker_or_spends_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "ledger.json"
            with patch("ditserve.directions.runner.inspect", return_value={"status": "blocked"}), \
                 patch("ditserve.directions.runner.subprocess.Popen") as popen:
                result = execute(load_config(), root, ledger, 1, "profile")
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["gpu_work_started"])
            popen.assert_not_called()
            self.assertFalse(ledger.exists())

    def test_failed_correctness_cannot_advance_to_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "direction1"
            child.mkdir()
            (child / "worker-correctness.json").write_text(json.dumps({"status": "passed", "eligible": True}))
            (child / "supervisor-correctness.json").write_text(json.dumps({"status": "incomplete", "eligible": False}))
            with patch("ditserve.directions.runner.subprocess.Popen") as popen:
                result = execute(load_config(), root, root / "ledger.json", 1, "preemption")
            self.assertEqual(result["status"], "blocked")
            popen.assert_not_called()

    def test_missing_measurements_do_not_become_zero_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = report(root, root / "missing-ledger.json")
            self.assertEqual(result["budget"]["status"], "not_started")
            for direction in result["directions"]:
                self.assertTrue(all(p["result"]["status"] == "not_run" for p in direction["phases"]))

    def test_worker_report_alone_does_not_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker-profile.json").write_text(json.dumps({"status": "passed", "eligible": True}))
            self.assertEqual(phase_result(root, "profile")["status"], "incomplete")

    def test_direct_entry_cannot_skip_a_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("ditserve.directions.runner.inspect") as inspect, \
                 patch("ditserve.directions.runner.subprocess.Popen") as popen:
                result = execute(load_config(), root, root / "ledger.json", 2, "screen")
            self.assertEqual(result["status"], "blocked")
            inspect.assert_not_called()
            popen.assert_not_called()

    def test_d2_cost_requires_qualified_cache_and_pressure_requires_cost(self):
        for phase, prerequisite in (("state_cost", "qualify"), ("pressure", "state_cost")):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "direction1").mkdir()
                (root / "direction1/terminal.json").write_text('{"status":"stopped"}')
                child = root / "direction2"
                child.mkdir()
                (child / ("worker-%s.json" % prerequisite)).write_text(
                    json.dumps(dict(status="passed", eligible=True)))
                (child / ("supervisor-%s.json" % prerequisite)).write_text(
                    json.dumps(dict(status="incomplete", eligible=False)))
                with patch("ditserve.directions.runner.inspect") as inspect_gpu, \
                     patch("ditserve.directions.runner.subprocess.Popen") as popen:
                    result = execute(load_config(), root, root / "ledger.json", 2, phase)
                self.assertEqual(result["status"], "blocked")
                inspect_gpu.assert_not_called()
                popen.assert_not_called()
                self.assertFalse((root / "ledger.json").exists())

    def test_partial_snapshot_is_not_an_available_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_index.json").write_text("{}")
            (root / "text_encoder").mkdir()
            (root / "text_encoder/model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}}))
            result = model_inventory("sana", {"path": str(root), "revision": "fixed"})
            self.assertFalse(result["available"])
            self.assertIn("text_encoder/model-00001-of-00002.safetensors", result["missing"])


class LongLiveEnvironmentAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assets = self.root / "longlive"
        self.assets.mkdir()
        required = ["longlive_models/models/longlive_base.pt", "longlive_models/models/lora.pt",
                    "wan_models/Wan2.1-T2V-1.3B/config.json", "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model",
                    "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json"]
        files = []
        for name in required:
            target = self.assets / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"CPU fixture")
            files.append(dict(path=name, bytes=target.stat().st_size,
                              sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        self.python = self.assets / "venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"never execute this CPU fixture")
        self.manifest_path = self.assets / "provisioning.json"
        self.record_path = self.assets / "environment-cpu-verification.json"
        self.entry = dict(path=str(self.assets), python=str(self.python), revision="fixed",
                          verification_manifest=str(self.manifest_path))
        self.manifest = dict(status="verified", files=files)
        self.record = dict(status="verified_cpu_only", python=str(self.python), gpu_used=False,
                           cuda_initialized=False, native_attention_backends=dict(
                               FLASH_ATTN_2_AVAILABLE=True, FLASH_ATTN_3_AVAILABLE=False))
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def verified_environment(self):
        self.record_path.write_text(json.dumps(self.record))
        self.manifest["environment_validation"] = dict(status="passed", python=str(self.python),
            record=str(self.record_path), sha256=hashlib.sha256(self.record_path.read_bytes()).hexdigest())
        self.save_manifest()

    def inventory(self):
        return model_inventory("longlive", self.entry)

    def test_verified_weights_and_python_without_environment_record_are_unavailable(self):
        result = self.inventory()
        self.assertFalse(result["available"])
        self.assertEqual(result["missing"], ["<environment-not-validated>"])
        self.assertEqual(result["verification_manifest_sha256"],
                         hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())

    def test_verified_cpu_record_is_available_without_executing_environment(self):
        self.verified_environment()
        with patch("ditserve.directions.runner.subprocess.Popen") as popen:
            result = self.inventory()
        popen.assert_not_called()
        self.assertTrue(result["available"], result["missing"])
        self.assertEqual(result["environment_record_sha256"],
                         hashlib.sha256(self.record_path.read_bytes()).hexdigest())
        self.assertIn("CPU environment validation record", result["scope"])

    def test_invalid_validation_envelope_is_unavailable(self):
        cases = [("status", "building", "<environment-not-validated>"),
                 ("python", "/different/venv/bin/python", "<environment-python-mismatch>"),
                 ("python", "venv/bin/python", "<environment-python-mismatch>"),
                 ("record", None, "<environment-validation-record-not-absolute>"),
                 ("record", "environment-cpu-verification.json", "<environment-validation-record-not-absolute>"),
                 ("record", str(self.root / "missing.json"), "<environment-validation-record-missing>"),
                 ("sha256", "0" * 64, "<environment-validation-record-hash-mismatch>")]
        for field, value, marker in cases:
            with self.subTest(field=field, value=value):
                self.verified_environment()
                self.manifest["environment_validation"][field] = value
                self.save_manifest()
                result = self.inventory()
                self.assertFalse(result["available"])
                self.assertIn(marker, result["missing"])

    def test_rehashed_record_still_requires_cpu_identity_and_real_backend_validation(self):
        valid = dict(self.record)
        cases = [("status", "incomplete", "<environment-validation-record-not-verified>"),
                 ("python", "/different/venv/bin/python", "<environment-validation-record-identity-mismatch>"),
                 ("gpu_used", True, "<environment-validation-record-identity-mismatch>"),
                 ("cuda_initialized", True, "<environment-validation-record-identity-mismatch>"),
                 ("native_attention_backends", {}, "<environment-attention-backend-not-validated>"),
                 ("native_attention_backends", {"FLASH_ATTN_2_AVAILABLE": False},
                  "<environment-attention-backend-not-validated>")]
        for field, value, marker in cases:
            with self.subTest(field=field):
                self.record = dict(valid, **{field: value})
                self.verified_environment()
                result = self.inventory()
                self.assertFalse(result["available"])
                self.assertIn(marker, result["missing"])
                self.assertIsNone(result["environment_record_sha256"])

    def test_invalid_json_and_missing_interpreter_fail_closed(self):
        self.verified_environment()
        self.record_path.write_text("not JSON")
        self.manifest["environment_validation"]["sha256"] = hashlib.sha256(self.record_path.read_bytes()).hexdigest()
        self.save_manifest()
        self.assertIn("<environment-validation-record-invalid>", self.inventory()["missing"])
        self.verified_environment()
        self.python.unlink()
        result = self.inventory()
        self.assertFalse(result["available"])
        self.assertIn("<environment-python-unavailable>", result["missing"])

    def test_environment_validation_does_not_bypass_asset_completeness(self):
        self.verified_environment()
        weight = self.assets / "longlive_models/models/lora.pt"
        weight.write_bytes(b"truncated")
        result = self.inventory()
        self.assertFalse(result["available"])
        self.assertIn("longlive_models/models/lora.pt <manifest-size-mismatch>", result["missing"])

    def test_unvalidated_longlive_blocks_only_direction3_without_spending_or_starting_worker(self):
        config = load_config()
        config["models"]["longlive"] = self.entry
        for name in ("wan", "sana"):
            root = self.root / name
            root.mkdir()
            (root / "model_index.json").write_text("{}")
            for component in ("transformer", "vae", "text_encoder"):
                folder = root / component
                folder.mkdir()
                (folder / "config.json").write_text("{}")
                (folder / "model.safetensors").write_bytes(b"CPU fixture")
            config["models"][name] = dict(path=str(root), revision="fixed")
        output = self.root / "run"
        (output / "direction2").mkdir(parents=True)
        (output / "direction2/terminal.json").write_text('{"status":"stopped"}')
        ledger = self.root / "ledger.json"
        with patch("ditserve.directions.runner.discover_gpus", return_value=dict(
                devices=[dict(uuid=config["gpu_uuid"])])), \
             patch("ditserve.directions.runner.gpu_occupancy", return_value=dict(status="exclusive_observed")), \
             patch("ditserve.directions.runner.subprocess.Popen") as popen, \
             patch("ditserve.directions.runner.freeze_phase") as freeze:
            self.assertEqual(inspect(config, 1)["status"], "ready")
            self.assertEqual(inspect(config, 2)["status"], "ready")
            third = inspect(config, 3)
            self.assertEqual(third["status"], "blocked")
            self.assertEqual(third["missing_models"], ["longlive"])
            result = execute(config, output, ledger, 3, "correctness")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["gpu_work_started"])
        self.assertFalse(ledger.exists())
        popen.assert_not_called()
        freeze.assert_not_called()


class BudgetAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "run"
        self.ledger = self.root / "ledger.json"
        self.parent = self.root / "parent.json"
        self.parent.write_text(json.dumps(dict(used_gpu_seconds=1., limit_gpu_seconds=28800.)))
        self.config = load_config()
        self.config["python"] = sys.executable
        self.parent_patch = patch("ditserve.directions.runner.PARENT_LEDGER", self.parent)
        self.parent_patch.start()
        self.addCleanup(self.parent_patch.stop)

    def set_remaining(self, remaining):
        with Allocation(self.ledger, self.parent) as budget:
            used = budget.data["stage_limits_gpu_seconds"]["direction1"] - remaining
            budget.data["stage_used_gpu_seconds"]["direction1"] = used
            budget.data["used_gpu_seconds"] = used
            budget.save()

    def test_insufficient_budget_is_free_terminal_and_next_direction_can_enter(self):
        for remaining in (119., 0.):
            with self.subTest(remaining=remaining):
                self.set_remaining(remaining)
                before = self.ledger.read_bytes()
                with patch("ditserve.directions.runner.inspect", return_value=dict(status="ready")), \
                     patch("ditserve.directions.runner.freeze_phase") as freeze, \
                     patch("ditserve.directions.runner.subprocess.Popen") as popen:
                    result = execute(self.config, self.output, self.ledger, 1, "profile")
                self.assertEqual(result["status"], "stopped")
                self.assertEqual(result["reason_code"], "budget_insufficient")
                self.assertEqual(result["scientific_outcome"], "not_measured_budget_limited")
                self.assertFalse(result["gpu_work_started"])
                self.assertFalse(result["eligible"])
                self.assertNotIn("charge", result)
                self.assertEqual(self.ledger.read_bytes(), before)
                terminal = self.output / "direction1/terminal.json"
                self.assertEqual(json.loads(terminal.read_text()), result)
                self.assertTrue((self.output / "direction1/admission-profile.json").exists())
                self.assertFalse((self.output / "direction1/worker-profile.json").exists())
                freeze.assert_not_called()
                popen.assert_not_called()
        with patch("ditserve.directions.runner.inspect", return_value=dict(status="blocked")) as inspect:
            second = execute(self.config, self.output, self.ledger, 2, "screen")
        inspect.assert_called_once()
        self.assertEqual(second["reason"], "selected A100 is busy or unavailable")

    def test_short_explicit_timeout_is_retryable_without_terminal_or_attempt(self):
        self.set_remaining(1000.)
        before = self.ledger.read_bytes()
        with patch("ditserve.directions.runner.inspect", return_value=dict(status="ready")), \
             patch("ditserve.directions.runner.freeze_phase") as freeze, \
             patch("ditserve.directions.runner.subprocess.Popen") as popen:
            result = execute(self.config, self.output, self.ledger, 1, "profile", timeout=74.)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason_code"], "timeout_too_short")
        self.assertTrue(result["retryable"])
        self.assertFalse((self.output / "direction1/terminal.json").exists())
        self.assertEqual(self.ledger.read_bytes(), before)
        freeze.assert_not_called()
        popen.assert_not_called()
        # A corrected timeout reaches freezing in the same run directory.
        with patch("ditserve.directions.runner.inspect", return_value=dict(status="ready")), \
             patch("ditserve.directions.runner.freeze_phase", side_effect=ValueError("admitted to freezing")):
            with self.assertRaisesRegex(ValueError, "admitted to freezing"):
                execute(self.config, self.output, self.ledger, 1, "profile", timeout=75.)

    def test_exact_minimum_is_admitted(self):
        self.set_remaining(120.)
        with patch("ditserve.directions.runner.inspect", return_value=dict(status="ready")), \
             patch("ditserve.directions.runner.freeze_phase", side_effect=ValueError("admitted to freezing")):
            with self.assertRaisesRegex(ValueError, "admitted to freezing"):
                execute(self.config, self.output, self.ledger, 1, "profile")
        self.assertFalse((self.output / "direction1/terminal.json").exists())

    def test_unsettled_active_session_cannot_be_converted_to_budget_terminal(self):
        with Allocation(self.ledger, self.parent) as budget:
            budget.begin("direction1", gpu_count=1, identity={})
        before = self.ledger.read_bytes()
        with patch("ditserve.directions.runner.inspect", return_value=dict(status="ready")), \
             patch("ditserve.directions.runner.freeze_phase") as freeze, \
             patch("ditserve.directions.runner.subprocess.Popen") as popen:
            with self.assertRaises(BudgetStop):
                execute(self.config, self.output, self.ledger, 1, "profile")
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse((self.output / "direction1/terminal.json").exists())
        freeze.assert_not_called()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
