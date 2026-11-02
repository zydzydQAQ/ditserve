"""2026-11-02: output identity, bounded monitoring and queue completion."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.runtime.worker import bind_generation_identity, runtime_identity


LOCAL = Path(os.environ.get("DITSERVE_KAIROS_RUNTIME_ROOT",
    Path(__file__).resolve().parents[3] / ".local/ditserve-kairos-rewrite-runtime"))


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeEvidenceTests(unittest.TestCase):
    @unittest.skipUnless((LOCAL / "tools/generate_batch_20261102.py").is_file(), "local operator scripts")
    def test_validation_specs_are_dated_staged_and_never_overwritten(self):
        jobs = module_at("validation_specs", LOCAL / "tools/generate_batch_20261102.py")
        with tempfile.TemporaryDirectory() as directory:
            jobs.ROOT = Path(directory)
            with patch.object(jobs, "resolve_sha", return_value="0123456"), \
                 patch("sys.argv", ["generate", "0123456", "--stage", "p0", "--date", "20261102"]):
                jobs.main()
                jobs.main()
                paths = list(jobs.ROOT.rglob("*.json"))
                self.assertEqual(len(paths), 5)
                for path in paths:
                    name = json.loads(path.read_text())["name"]
                    self.assertTrue(name.startswith("d20261102-"))
                    self.assertLessEqual(len(name), 52)
                paths[0].write_text("changed")
                with self.assertRaises(ValueError):
                    jobs.main()
                self.assertEqual(paths[0].read_text(), "changed")

    def test_identity_covers_each_batch_output_without_verifying_declared_weights(self):
        result = {"results": [{"generation": {"dtype": "float16"}}, {"generation": {}}]}
        with patch("ditserve.runtime.worker.runtime_identity", return_value={"packages": {"torch": "test"}}):
            bind_generation_identity(result, {"dtype": "bfloat16", "model_manifest_sha256": "a" * 64})
        for item in result["results"]:
            self.assertEqual(item["generation"]["runtime_identity"]["packages"]["torch"], "test")
            self.assertNotEqual(item["generation"].get("model_integrity"), "manifest_verified")
        self.assertEqual(result["results"][0]["generation"]["dtype"], "float16")
        self.assertEqual(result["results"][1]["generation"]["dtype"], "bfloat16")

    def test_runtime_metadata_does_not_import_accelerator(self):
        runtime_identity.cache_clear()
        with patch("importlib.import_module", side_effect=AssertionError("runtime import")):
            identity = runtime_identity()
        self.assertIn("python", identity)
        self.assertIn("torch", identity["packages"])
        runtime_identity.cache_clear()

    @unittest.skipUnless((LOCAL / "scripts/kairos_runtime_audit.py").is_file(), "local operator scripts")
    def test_memory_monitor_retains_extrema_without_samples(self):
        audit = module_at("memory_audit", LOCAL / "scripts/kairos_runtime_audit.py")
        values = {}
        audit.add_memory_sample(values, "t1, GPU-a, 10, 90, 100, 20")
        for _ in range(1000):
            audit.add_memory_sample(values, "t2, GPU-a, 30, 70, 100, 40")
        item = values["GPU-a"]
        self.assertEqual(item["samples"], 1001)
        self.assertEqual(item["memory_used_max_mib"], 30)
        self.assertEqual(item["memory_free_min_mib"], 70)
        self.assertLess(len(json.dumps(values)), 400)
        with self.assertRaises(ValueError):
            audit.add_memory_sample(values, "t3, GPU-a, NaN, 70, 100, 40")

    @unittest.skipUnless((LOCAL / "tools/kairos_queue.py").is_file(), "local operator scripts")
    def test_successful_queue_terminates_instead_of_polling_forever(self):
        queue = module_at("kairos_queue_test", LOCAL / "tools/kairos_queue.py")
        class API:
            submitted = False
            calls = 0
            def req(self, method, path, payload=None):
                self.calls += 1
                if self.calls > 6:
                    raise AssertionError("queue did not stop after success")
                if method == "POST":
                    self.submitted = True
                    return 201, {}
                return 200, ([{"name": "d20261102-contract", "namespace": "user-jjjkairos", "state": "Succeeded"}] if self.submitted else [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.json"
            path.write_text(json.dumps(dict(name="d20261102-contract", mount_shared=True,
                workers=1, gpus_per_worker=1, env=[{"name": "DITSERVE_RUN_ID", "value": "d20261102-contract"}])) )
            with patch.object(queue, "load_api", return_value=API()), patch.object(queue, "storage_gate"), \
                 patch.object(queue, "build_gate", return_value={}), patch.object(queue.time, "sleep"), \
                 patch.object(queue, "live_capacity", return_value={"quota": 16, "account_free_gpus": 16, \
                     "unplaced_account_jobs": [], "nodes": [{"free_gpus": 8}]}), \
                 patch("sys.argv", ["queue", str(path), "--submit"]):
                queue.main()

    @unittest.skipUnless((LOCAL / "tools/kairos_queue.py").is_file(), "local operator scripts")
    def test_numeric_job_name_is_rejected_before_login(self):
        queue = module_at("kairos_queue_names", LOCAL / "tools/kairos_queue.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.json"
            path.write_text(json.dumps({"name": "20261102-invalid"}))
            with patch.object(queue, "load_api", side_effect=AssertionError("must reject before network")), \
                 patch("sys.argv", ["queue", str(path), "--submit"]):
                with self.assertRaisesRegex(ValueError, "start with a letter"):
                    queue.main()

    @unittest.skipUnless((LOCAL / "tools/kairos_queue.py").is_file(), "local operator scripts")
    def test_serial_queue_waits_for_completion_despite_spare_gpu_quota(self):
        queue = module_at("kairos_queue_serial", LOCAL / "tools/kairos_queue.py")
        class API:
            names = []
            polls = 0
            first_done = False
            def req(self, method, path, payload=None):
                if method == "POST":
                    if self.names and not self.first_done:
                        raise AssertionError("second job overlapped the first")
                    self.names.append(payload["name"])
                    return 201, {}
                self.polls += 1
                if self.polls > 8:
                    raise AssertionError("serial queue did not complete")
                self.first_done = bool(self.names) and self.polls >= 4
                return 200, [dict(name=name, state="Succeeded" if self.first_done else "Running",
                                 namespace="user-jjjkairos", total_gpus=1) for name in self.names]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("d20261102-one", "d20261102-two"):
                path = Path(directory) / (name + ".json")
                path.write_text(json.dumps(dict(name=name, mount_shared=True, workers=1, gpus_per_worker=1,
                    env=[{"name": "DITSERVE_RUN_ID", "value": name}])))
                paths.append(str(path))
            api = API()
            with patch.object(queue, "load_api", return_value=api), patch.object(queue, "storage_gate"), \
                 patch.object(queue, "build_gate", return_value={}), patch.object(queue.time, "sleep"), \
                 patch.object(queue, "live_capacity", return_value={"quota": 16, "account_free_gpus": 16, \
                     "unplaced_account_jobs": [], "nodes": [{"free_gpus": 8}]}), \
                 patch("sys.argv", ["queue", *paths, "--submit", "--max-active-jobs", "1"]):
                queue.main()
            self.assertEqual(len(api.names), 2)


if __name__ == "__main__":
    unittest.main()
