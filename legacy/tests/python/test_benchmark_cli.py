import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ditserve.cli import main
from ditserve.benchmark.common import read_json, verify_seal


class BenchmarkCliTests(unittest.TestCase):
    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["benchmark"] + [str(arg) for arg in args])

    def test_public_cpu_pipeline_replay_evaluation_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prompts.json"
            source.write_text(json.dumps([{"prompt_en": "A bird flying over scene %d." % i,
                                           "dimension": ["subject_consistency"]} for i in range(8)]))
            dataset = root / "dataset.json"
            self.assertEqual(self.invoke("prepare", "--dataset", "vbench", "--source", source,
                                         "--dev-count", 2, "--output", dataset), 0)
            self.assertEqual(self.invoke("profile", "--dataset", dataset, "--output", root / "profile"), 0)
            self.assertEqual(self.invoke("trace", "--dataset", dataset,
                "--profile", root / "profile" / "profile.json", "--output", root / "trace"), 0)
            trace = root / "trace" / "trace.json"
            self.assertEqual(self.invoke("run", "--trace", trace, "--dry-run"), 0)
            self.assertEqual(self.invoke("run", "--trace", trace, "--output", root / "attempt"), 0)
            run = verify_seal(read_json(root / "attempt" / "run.json"))
            self.assertFalse(run["measured"])
            self.assertEqual(set(run["policies"]), {"fcfs", "edf", "srtf"})
            self.assertTrue(all(len(arm["requests"]) == 6 for arm in run["policies"].values()))
            self.assertEqual(self.invoke("evaluate", "--run", root / "attempt"), 0)
            self.assertEqual(self.invoke("report", "--run", root / "attempt"), 0)
            reports = list((root / "attempt").rglob("REPORT.md"))
            self.assertTrue(reports)
            self.assertEqual(self.invoke("run", "--trace", trace, "--output", root / "attempt"), 2)
            self.assertEqual(read_json(root / "attempt" / "run.json")["sha256"], run["sha256"])

    def test_download_not_implicit_and_frozen_data_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.json"
            self.assertEqual(self.invoke("prepare", "--dataset", "vbench", "--output", destination), 2)
            self.assertFalse(destination.exists())
            destination.write_text("preserve")
            self.assertEqual(self.invoke("prepare", "--dataset", "vbench", "--download", "--output", destination), 2)
            self.assertEqual(destination.read_text(), "preserve")

    def test_quality_manifest_runs_all_generations_without_queue_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "full_info.json"
            source.write_text(json.dumps([{"prompt_en": "A scene %d." % i,
                                           "dimension": ["subject_consistency"]} for i in range(8)]))
            data, manifest = root / "dataset.json", root / "manifest.json"
            self.assertEqual(self.invoke("prepare", "--dataset", "vbench", "--source", source,
                                         "--dev-count", 2, "--output", data), 0)
            self.assertEqual(self.invoke("profile", "--dataset", data, "--output", root / "profile"), 0)
            self.assertEqual(self.invoke("evaluate", "--vbench-info", source, "--config", "smoke",
                "--shape", "short", "--dimensions", "subject_consistency", "--output", manifest), 0)
            self.assertEqual(self.invoke("trace", "--dataset", data, "--profile", root / "profile" / "profile.json",
                "--vbench-manifest", manifest, "--output", root / "trace"), 0)
            self.assertEqual(self.invoke("run", "--trace", root / "trace" / "trace.json", "--policies", "fcfs",
                "--output", root / "run"), 0)
            result = read_json(root / "run" / "run.json")
            self.assertFalse(result["serving_performance_evidence"])
            self.assertEqual(len(result["policies"]["fcfs"]["requests"]), 40)
            self.assertTrue(all(r["status"] == "completed" for r in result["policies"]["fcfs"]["requests"]))
            self.assertEqual(self.invoke("evaluate", "--export-videos", "--run", root / "run",
                "--vbench-manifest", manifest, "--videos-dir", root / "videos"), 2)


if __name__ == "__main__":
    unittest.main()
