"""CPU-only publication, exact-evidence retention and fail-closed contracts."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ditserve import results
from ditserve.evaluation.wan_profile.figure import build_analysis
from ditserve.evaluation.wan_profile.publish import _hardware_snapshot, _snapshot, build_record


def snapshot(payload, name="fixture.json"):
    raw = json.dumps(payload, allow_nan=False).encode()
    return dict(status="loaded", payload=copy.deepcopy(payload),
                source=dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))


def evidence():
    phases = []
    for mode in ("trace", "hardware"):
        ranks = []
        for frames in (17, 33, 65):
            rank = dict(mode=mode, status="ok", spec=dict(frames=frames),
                numerical_validation=dict(passed=True, reference_sha256="1" * 64),
                instrumentation_validation=dict(passed=True), output=dict(sha256="1" * 64))
            if mode == "trace":
                rank["profiler"] = dict(status="measured", timeline_status="measured", attribution_status="measured",
                    request=dict(kernel_active_pct=60.1234567890123), gpu_window=dict(kernel_active_pct=90.),
                    transformer_kernel_duration_ms=1000.,
                    transformer_kernel_categories=[dict(name=name, share_pct=value) for name, value in
                        (("self_attention", 50.), ("cross_attention", 10.), ("ffn", 30.), ("transformer_other", 10.))],
                    counts=dict(kernel_count=10001, excluded_gpu_annotation_count=123),
                    omitted_kernel_names=dict(count=52, duration_ms=0.123456789012345))
                rank["artifacts"] = [dict(path="trace-%d.json" % frames, kind="trace", bytes=12345,
                                         sha256="2" * 64, dependencies=["profile_validation"])]
            ranks.append(rank)
        phases.append(dict(id="software_trace" if mode == "trace" else "hardware_metrics", status="completed",
            qualification=dict(qualified=True, scopes=dict(performance=dict(status="passed"), numeric=dict(status="passed"))),
            final_occupancy=dict(status="exclusive_observed", foreign=[], owned=[]),
            cells=[dict(status="completed", operations=[dict(ranks=ranks)])]))
    ledger = dict(schema_version="ditserve.budget.v1", used_gpu_seconds=1234.567890123456,
                  active=None, sessions=[dict(phase="software_trace", gpu_seconds=713.8539540779311,
                                             status="completed", cleanup_verified=True)])
    study = dict(schema="ditserve.run.v1", status="completed", phases=phases, failures=[], budget=copy.deepcopy(ledger))
    ranges = []
    for frames in (17, 33, 65):
        for scope in ("request", "denoise"):
            ranges.append(dict(frames=frames, scope=scope, valid=True, status="sampled", start_ns=12345678901234567,
                metrics=[dict(id=key, measured=True, status="sampled", time_weighted_mean_pct=12.3456789012345,
                    sample_min_pct=0., sample_max_pct=97.6543210987654, sample_count=123456, coverage_fraction=.9999999,
                    source_rows=dict(rows_seen=222222, invalid_rows=0, conflicting_duplicates=0), reason_codes=[])
                for key in ("sm_active_pct", "tensor_active_pct", "dram_read_active_pct", "dram_write_active_pct")]))
    hardware = dict(schema="ditserve.wan_hardware_capture.v1", status="sampled", measured=True,
        hardware_phase_qualified=True, capture_returncode=0, metrics=dict(status="sampled", measured=True, ranges=ranges),
        retained_files=[dict(path="gpu.sqlite", bytes=987654321, sha256="3" * 64)],
        errors=[], command=["nsys", "profile", "--gpu-metrics-frequency=1000"])
    return study, hardware, ledger


def record_for(study, hardware, ledger, work):
    analysis = build_analysis({}, {}, study=study, hardware=hardware)
    analysis["figures"] = {suffix: dict(path=str(work / ("figure." + suffix)), bytes=123, sha256="4" * 64)
                           for suffix in ("png", "svg")}
    return build_record(analysis, snapshot(study), snapshot(hardware), snapshot(ledger), work=work,
                        captured="2026-10-17T06:00:00+00:00")


class WanProfilePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_excel_roundtrip_retains_complete_hardware_and_exact_software_evidence(self):
        study, hardware, ledger = evidence()
        record, pending = record_for(study, hardware, ledger, self.root)
        self.assertEqual(pending, [])
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["budget"]["settled_measurement_gpu_seconds"], ledger["used_gpu_seconds"])
        self.assertEqual(record["budget"]["publisher_gpu_seconds"], 0)
        self.assertEqual(record["measurement_evidence"]["hardware_capture"]["payload"], hardware)
        work, archive = self.root / "work", self.root / "archive"
        work.mkdir()
        (work / "run.json").write_text(json.dumps(record, ensure_ascii=False, allow_nan=False))
        self.assertEqual(results.finalize(work, archive, prune=False)["status"], "completed")
        restored = results.load(archive)["run"]
        self.assertEqual(restored["measurement_evidence"], record["measurement_evidence"])
        self.assertEqual(restored["budget"], record["budget"])
        profiles = restored["measurement_evidence"]["software_and_hardware_profiles"]
        self.assertEqual(profiles[0]["result"]["profiler"], study["phases"][0]["cells"][0]["operations"][0]["ranks"][0]["profiler"])
        report = (archive / "REPORT.md").read_text()
        top_rows = [line for line in report.splitlines() if line.startswith("| run/")]
        self.assertEqual(len(top_rows), 30)
        self.assertTrue(any("27_65f_DRAM_write_pct" in line for line in top_rows))
        self.assertTrue(any("04_settled_GPU_seconds" in line for line in top_rows))
        self.assertFalse(any("sha256" in line or "source/path" in line for line in top_rows))

    def test_failure_state_and_unknown_hardware_are_preserved_without_mutation(self):
        study, hardware, ledger = evidence()
        hardware.update(status="failed", measured=False, failure="GPU isolation lost")
        study["failures"] = [dict(phase="hardware_metrics", error="foreign process observed")]
        original = copy.deepcopy(hardware)
        record, pending = record_for(study, hardware, ledger, self.root)
        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(set(pending), {"sm", "tensor", "dram"})
        self.assertEqual(record["measurement_evidence"]["hardware_capture"]["payload"], original)
        self.assertEqual(hardware, original)
        self.assertEqual(record["metrics"]["16_17f_SM_pct"], "request unmeasured; denoise NVTX unmeasured")
        self.assertEqual(len(record["failures"]), 2)

    def test_snapshot_rejects_malformed_nonfinite_and_raw_trace_evidence(self):
        path = self.root / "summary.json"
        for contents in ('{"status":', '{"value": NaN}', '{"traceEvents": []}'):
            with self.subTest(contents=contents):
                path.write_text(contents)
                result = _snapshot(path)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["payload"])
                self.assertEqual(result["source"]["sha256"], hashlib.sha256(contents.encode()).hexdigest())

    def test_missing_summary_falls_back_to_complete_wrapper_in_run_only(self):
        study, hardware, unused = evidence()
        study["hardware_capture"] = hardware
        study["hardware_capture_source"] = dict(path="nsys/summary.json", bytes=12345, sha256="5" * 64)
        path = self.root / "missing-summary.json"
        result = _hardware_snapshot(path, snapshot(study, "run.json"))
        self.assertEqual(result["status"], "loaded")
        self.assertEqual(result["payload"], hardware)
        self.assertEqual(result["source"]["json_pointer"], "/hardware_capture")
        self.assertEqual(result["source"]["original_summary_source"], study["hardware_capture_source"])
        path.write_text("invalid-json")
        self.assertEqual(_hardware_snapshot(path, snapshot(study))["status"], "failed")
        path.unlink()
        study["hardware_capture"] = hardware["metrics"]
        self.assertEqual(_hardware_snapshot(path, snapshot(study))["status"], "failed")

    def test_active_ledger_is_not_presented_as_settled_publication(self):
        study, hardware, ledger = evidence()
        ledger["active"] = dict(phase="hardware_metrics", session_id="running")
        record, unused = record_for(study, hardware, ledger, self.root)
        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["budget"]["accounting_status"], "active_or_unsettled")
        self.assertEqual(record["budget"]["measurement_ledger_snapshot"]["active"], ledger["active"])

    def test_replacement_preserves_original_failed_capture_and_separate_run(self):
        study, hardware, ledger = evidence()
        analysis = build_analysis({}, {}, study=study, hardware=hardware)
        analysis["figures"] = {suffix: dict(path=str(self.root / ("figure." + suffix)))
                               for suffix in ("png", "svg")}
        prior = snapshot(dict(schema="ditserve.wan_hardware_capture.v1", status="failed",
                              measured=False, capture_returncode=1, failure="Wrong event order"))
        recovery = snapshot(dict(run_id="hardware02", status="incomplete", budget=ledger,
                                 phases=study["phases"][1:]))
        record, pending = build_record(analysis, snapshot(study), snapshot(hardware), snapshot(ledger),
            work=self.root, captured="2026-10-17T06:00:00Z",
            hardware_study_snapshot=recovery, prior_capture_snapshot=prior)
        self.assertEqual(pending, [])
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["measurement_evidence"]["prior_capture_attempt"], prior)
        self.assertEqual(record["measurement_evidence"]["hardware_measurement_run"], recovery)
        self.assertEqual(record["metrics"]["35_hardware_run_status"], "incomplete")
        self.assertEqual(record["budget"]["settled_measurement_gpu_seconds"], ledger["used_gpu_seconds"])


if __name__ == "__main__":
    unittest.main()
