"""Derived encoding summaries preserve launch vetoes and missing evidence."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[2] / "docker/summarize-workflow-encoding.py"
SPEC = importlib.util.spec_from_file_location("encoding_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class EncodingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "run01"
        self.root.mkdir()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def encoder(self, variant="T0", status="completed", start=150):
        return dict(schema="ditserve.workflow.encoder.v1", variant=variant, status=status,
                    started_unix_ns=start, finished_unix_ns=start + 20, frames=[17, 33], numeric=[], requests=[])

    def launch(self, variant="T0", status="completed", start=100, end=200, replay=False, identifier=None):
        identifier = identifier or variant
        module = "workflow_embedding_replay" if replay else "workflow_encoder"
        output = self.root / ("embedding-replay" if replay else "encoder-v2")
        command = ["python", "-m", "ditserve.evaluation." + module, "--out" if replay else "--output", str(output)]
        if not replay:
            command += ["--phases", variant]
        return self.write("launches/" + identifier + "/launch.json", dict(id=identifier, command=command, status=status,
                          started_unix_ns=start, finished_unix_ns=end, returncode=0 if status == "completed" else 1,
                          gpu_monitor=dict(conflicts=[], errors=[], admission="accepted")))

    def test_aggregates_per_request_and_preserves_missing_null(self):
        value = self.encoder()
        value["requests"] = [dict(kind="video", frames=17, repeat=0, inference_wall_ms=20, finite=True)]
        def phase(wall, increment, purpose="video_measurement"):
            return dict(name="text_encoder", status="completed", wall_ms=wall, process_cpu_ms=wall*8,
                        metadata=dict(purpose=purpose, request_id="T0-f17-r0", repeat=0),
                        peak_reserved_increment_bytes=increment, peak_allocated_increment_bytes=increment/2,
                        torch_after=dict(device="cuda:1", peak_allocated_bytes=100, peak_reserved_bytes=200, device_total_bytes=1000))
        value["phase_records"] = [phase(3, 10), phase(7, 30), phase(900, 999, "warmup")]
        self.write("encoder-v2/T0/result.json", value)
        self.launch()
        result = summary.summarize(self.root)
        stage, = result["stage_resources"]
        self.assertEqual(stage["wall_ms_per_request"]["mean"], 10)
        self.assertEqual(stage["process_average_cores"], 8)
        self.assertEqual(stage["peak_reserved_increment_bytes"]["mean"], 30)
        self.assertEqual(stage["request_wall_share_percent"]["mean"], 50)
        self.assertEqual(result["availability"]["T0"]["video_frames"], [17])
        self.assertIsNone(result["variants"]["T1"]["frames"]["17"]["finite_all_recorded"])
        self.assertFalse(result["availability"]["T1"]["performance"])
        json.dumps(result, allow_nan=False)

    def test_default_encoder_layouts_and_explicit_override(self):
        self.write("encoder/T0/result.json", self.encoder())
        standard = summary.summarize(self.root)
        self.assertEqual(standard["encoder_root"], str(self.root / "encoder"))
        self.assertTrue(standard["availability"]["T0"]["performance"])
        self.write("encoder-v2/T0/result.json", self.encoder(status="failed"))
        preferred = summary.summarize(self.root)
        self.assertEqual(preferred["encoder_root"], str(self.root / "encoder-v2"))
        self.assertFalse(preferred["availability"]["T0"]["performance"])
        overridden = summary.summarize(self.root, encoder_root=self.root / "encoder")
        self.assertEqual(overridden["encoder_root"], str(self.root / "encoder"))
        self.assertTrue(overridden["availability"]["T0"]["performance"])

    def test_interrupted_launch_vetoes_raw_completed_and_new_retry_is_independent(self):
        self.write("encoder-v2/T0/result.json", self.encoder())
        self.launch(status="interrupted", identifier="old")
        result = summary.summarize(self.root)
        self.assertEqual(result["variants"]["T0"]["status"], "invalidated")
        self.assertFalse(result["availability"]["T0"]["performance"])
        self.assertEqual(result["stage_resources"], [])
        self.write("encoder-v2/T0/result.json", self.encoder(start=350))
        self.launch(start=300, end=400, identifier="retry")
        self.assertTrue(summary.summarize(self.root)["availability"]["T0"]["performance"])

    def test_oom_and_running_without_result_are_not_successes(self):
        value = self.encoder("T1", "failed")
        value["error"] = dict(type="OutOfMemoryError", message="capacity")
        self.write("encoder-v2/T1/result.json", value)
        self.launch("T1", "failed")
        self.launch("T2", "running", end=None)
        self.launch("T3", "gpu_monitor_unavailable")
        self.write("operator-notes.json", dict(active_job="E2-f33-resume1", current_batch="resume requested"))
        result = summary.summarize(self.root)
        self.assertEqual(result["variants"]["T1"]["status"], "oom")
        self.assertEqual(result["variants"]["T2"]["status"], "running")
        self.assertFalse(result["availability"]["T1"]["performance"])
        self.assertEqual(result["availability"]["T2"]["embedding_cases"], [])
        self.assertEqual(result["variants"]["T3"]["status"], "unmeasured")
        self.assertEqual(result["variants"]["T3"]["launch"]["status"], "gpu_monitor_unavailable")
        self.assertEqual(result["current_queue_context"]["operator_notes"]["active_job"], "E2-f33-resume1")

    def test_replay_requires_bridge_and_valid_launcher(self):
        value = dict(schema="ditserve.workflow.embedding_replay.v1", status="completed", started_unix_ns=150,
                     finished_unix_ns=170, results=[dict(variant="T1", frames=17, status="compared", comparison={})])
        self.write("embedding-replay/run.json", value)
        self.launch(replay=True)
        self.assertFalse(summary.summarize(self.root)["replay"]["results"][0]["evidence_eligible"])
        value["results"].append(dict(variant="T0", frames=17, status="compared", baseline_bridge_exact=True))
        self.write("embedding-replay/run.json", value)
        self.assertTrue(summary.summarize(self.root)["replay"]["results"][0]["evidence_eligible"])
        self.launch(replay=True, status="interrupted")
        result = summary.summarize(self.root)["replay"]
        self.assertEqual(result["status"], "invalidated")
        self.assertFalse(result["results"][0]["evidence_eligible"])
        self.assertFalse(result["include_in_performance"])

    def test_headers_are_observations_and_atomic_output_cannot_replace_raw(self):
        header = repr(dict(descr="<f4", fortran_order=False, shape=(1, 512, 4096))).encode() + b"\n"
        path = self.root / "encoder-v2/T0/numeric/english.npz"
        path.parent.mkdir(parents=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("prompt.npy", b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
        observed = summary.artifact_headers(str(path), self.root)
        self.assertEqual(observed["arrays"]["prompt"]["shape"], [1, 512, 4096])
        self.assertFalse(observed["values_recomputed"])
        source = self.write("encoder-v2/T0/result.json", self.encoder())
        before = source.read_bytes()
        with self.assertRaises(SystemExit):
            summary.main(["--root", str(self.root), "--out", str(source)])
        self.assertEqual(before, source.read_bytes())
        output = self.root / "reviews/summary.json"
        self.assertEqual(summary.main(["--root", str(self.root), "--out", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["schema"], summary.SCHEMA)
        self.assertEqual(list(output.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
