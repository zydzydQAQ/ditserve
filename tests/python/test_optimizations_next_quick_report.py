"""CPU-only quick report denominator, provenance and Excel tests."""

import copy
from pathlib import Path
import tempfile
import unittest

from ditserve.directions.direction1 import frozen_requests
from ditserve.optimizations.next_quick_report import collect_quick, write_quick_report
from ditserve.optimizations.online import paired_gate, request_identity, request_metrics
from ditserve.research.common import atomic_json, file_digest, read_json


class QuickReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.attempt = self.root/"run01/cache_load_09/attempt00"
        self.attempt.mkdir(parents=True)
        self.amendment = self.root/"amendment.json"
        atomic_json(self.amendment, dict(user_instruction="做完0.7的用1个seed快速验证", seed=101))
        identities, rows = [], []
        for pattern in ("ordinary", "burst"):
            requests = frozen_requests(count=24, seed=101, load=.9, pattern=pattern, image_ms=100., video_ms=1000.)
            identity = request_identity(requests)
            identities.append(identity)
            for order, arms in enumerate((("baseline", "candidate"), ("candidate", "baseline"))):
                for arm in arms:
                    values = copy.deepcopy(requests)
                    for request in values:
                        request.update(arrived=True, status="completed", start_ms=request["arrival_ms"],
                            finish_ms=request["arrival_ms"]+request["service_ms"]*(.5 if arm == "candidate" else 1.))
                    rows.append(dict(trace_sha256=identity, seed=101, load=.9, pattern=pattern,
                        order_index=order, arm=arm, requests=values,
                        elapsed_ms=max(r["finish_ms"] for r in values)+1., status="completed", complete=True,
                        events=[dict(secret_large_event="must not be copied")]))
        self.native = dict(protocol_mode="single_seed_screening", status="stopped", selected_seed=101,
            load=.9, planned_replays=8, planned_requests=192, rows=rows, expected_trace_ids=identities,
            screening_complete=True, eligible=False, online_qualified=False, full_qualification_completed=False,
            amendment_binding=dict(path=str(self.amendment), sha256=file_digest(self.amendment)))
        self.path = self.attempt/"cache_load_09.json"
        self.save()

    def save(self):
        atomic_json(self.path, self.native)

    def test_complete_screening_keeps_original_denominator_and_never_qualifies(self):
        self.native["screening_comparison"] = paired_gate(self.native["rows"], self.native["expected_trace_ids"])
        self.save()
        report = collect_quick(self.attempt)
        self.assertTrue(report["screening_complete"])
        self.assertTrue(report["screening_passed"])
        self.assertEqual((report["recorded_replays"], report["recorded_requests"]), (8, 192))
        self.assertEqual((report["original_planned_replays"], report["unexecuted_original_replays"]), (24, 16))
        self.assertFalse(report["original_matrix_complete"])
        self.assertFalse(report["online_qualified"])
        self.assertEqual(report["supervision_status"], "pending_supervision")
        self.assertNotIn("secret_large_event", str(report))

    def test_failure_stays_in_denominator_and_prevents_screening_success(self):
        request = self.native["rows"][0]["requests"][0]
        request.update(status="failed", start_ms=None, finish_ms=None)
        self.native["rows"][0].update(status="failed", complete=False)
        self.save()
        report = collect_quick(self.attempt)
        self.assertEqual(report["recorded_requests"], 192)
        self.assertFalse(report["screening_complete"])
        self.assertFalse(report["screening_passed"])
        self.assertEqual(report["replays"][0]["metrics"]["failed_or_incomplete"], 1)

    def test_partial_matrix_reports_actual_count(self):
        self.native["rows"].pop()
        self.native["screening_complete"] = False
        self.save()
        report = collect_quick(self.attempt)
        self.assertEqual((report["recorded_replays"], report["recorded_requests"]), (7, 168))
        self.assertFalse(report["screening_passed"])

    def test_changed_amendment_and_duplicate_pair_are_rejected(self):
        atomic_json(self.amendment, dict(seed=102))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            collect_quick(self.attempt)
        self.native["amendment_binding"]["sha256"] = file_digest(self.amendment)
        self.native["rows"][1] = copy.deepcopy(self.native["rows"][0])
        self.save()
        with self.assertRaisesRegex(ValueError, "identity"):
            collect_quick(self.attempt)

    def test_export_verifies_excel_and_does_not_write_parent_artifacts(self):
        before = file_digest(self.path)
        report = write_quick_report(self.attempt)
        self.assertEqual(report["excel"]["status"], "verified")
        self.assertFalse(report["excel"]["raw_events"])
        self.assertEqual(file_digest(self.path), before)
        self.assertEqual(read_json(self.attempt/"quick-report.json"), report)
        self.assertTrue((self.attempt/"QUICK_REPORT.md").exists())
        for name in ("REPORT.md", "report.json", "gpu-budget.json", "results.xlsx"):
            self.assertFalse((self.attempt/name).exists())

    def test_legacy_audit_accepts_stopped_subset_without_qualifying_full_matrix(self):
        from ditserve.optimizations.next_audit import _matrix
        expected = {}
        for seed in (101, 102, 103):
            for pattern in ("ordinary", "burst"):
                requests = frozen_requests(count=24, seed=seed, load=.9, pattern=pattern,
                    image_ms=100., video_ms=1000.)
                trace = dict(seed=seed, load=.9, pattern=pattern, requests=requests)
                for order in (0, 1):
                    for arm in ("baseline", "candidate"):
                        expected[request_identity(requests), order, arm] = trace
        for row in self.native["rows"]:
            for key, kind in (("metrics", None), ("image_metrics", "image"), ("video_metrics", "video")):
                requests = [r for r in row["requests"] if kind is None or r["kind"] == kind]
                row[key] = request_metrics(requests, row["elapsed_ms"])
        result = _matrix(self.native, expected)
        self.assertEqual((result["expected_replays"], result["recorded_replays"]), (24, 8))
        self.assertFalse(result["matrix_complete"])
        self.assertFalse(result["online_qualified"])


if __name__ == "__main__":
    unittest.main()
