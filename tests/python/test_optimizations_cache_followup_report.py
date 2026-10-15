"""Small real JSON/Excel fixtures; no models, NumPy, Torch, arrays or GPU."""

import copy
import gzip
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ditserve.optimizations import cache_followup_report as reporting
from ditserve.optimizations.online import paired_gate, request_identity
from ditserve.research.common import atomic_json, digest, file_digest, read_json


class CacheFollowupReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "results/supplement/run01"
        self.ledger = self.output.parent / "gpu-budget.json"
        self.old = self.root / "historical/run01"
        self.attempt = self.output / "cache_k4/attempt00"
        self.config_sha = "a" * 64
        self.requests = []
        for variant in range(2):
            rows = [dict(request_id="r%03d" % index, kind="video" if index in (0, 12) else "image",
                         arrival_ms=float(index * 10 + variant * 100),
                         deadline_ms=float(index * 10 + variant * 100 + 1000),
                         service_ms=10., seed=101 + variant)
                    for index in range(24)]
            self.requests.append(rows)
        self.trace_ids = tuple(request_identity(rows) for rows in self.requests)
        self.patch = patch.object(reporting, "FROZEN_TRACE_IDS", self.trace_ids)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def bound(self, path, value):
        atomic_json(path, value)
        return dict(path=str(path), sha256=file_digest(path), bytes=path.stat().st_size)

    def setup_study(self):
        original = self.old / "cache_k4/attempt00"
        bindings = {}
        for name in ("supervisor", "local", "config", "protocol", "manifest"):
            value = dict(status="passed")
            if name == "local":
                value["rows"] = [dict(kind="old_local", marker="ORIGINAL_LOCAL_ROW_MUST_NOT_COPY") for _ in range(84)]
            bindings[name] = self.bound(original / (name + ".json"), value)
        bindings["ledger"] = self.bound(self.old.parent / "gpu-budget.json",
            dict(active=None, used_gpu_seconds=22000., limit_gpu_seconds=28800.))
        research = self.bound(self.root / "research/gpu-budget.json",
            dict(active=None, used_gpu_seconds=40000., limit_gpu_seconds=86400.))
        auth = self.bound(self.output.parent / "authorization.json",
            dict(approved_by_user=True, user_instruction="fixture only", authorized_gpu_seconds=10800.,
                 scope="cache_online_factorial_only"))
        self.study = dict(source_root=str(self.old), authorization=auth, authorized_gpu_seconds=10800.,
            inherited_config_sha256=self.config_sha, old_ledgers=[bindings["ledger"], research],
            estimate=dict(trace_bindings=[dict(trace_sha256=trace) for trace in self.trace_ids]),
            replay=dict(config_sha256=self.config_sha, bindings=bindings,
                local_validation=dict(original_rows=84, new_local_executions=0, qualification_pairs=20)))
        atomic_json(self.output / "study.json", self.study)
        self.budget = dict(limit_gpu_seconds=10800., used_gpu_seconds=100., active=None,
            parents=dict(optimization=bindings["ledger"], research=research), sessions=[])
        atomic_json(self.ledger, self.budget)

    def matrix(self):
        rows = []
        for trace_index, requests in enumerate(self.requests):
            for order in (0, 1):
                for arm in reporting.ARMS if order == 0 else reversed(reporting.ARMS):
                    elapsed, latency = dict(off_step=(2000., 100.), off_k4=(1900., 80.),
                        paired_step=(1700., 80.), paired_k4=(1600., 60.))[arm]
                    measured = [dict(request, arrived=True, status="completed", start_ms=request["arrival_ms"] + 1.,
                                     finish_ms=request["arrival_ms"] + latency) for request in requests]
                    rows.append(dict(trace_sha256=self.trace_ids[trace_index], order_index=order,
                        factorial_arm=arm, arm=arm, requests=measured, elapsed_ms=elapsed,
                        status="completed", complete=True,
                        events=[dict(kind="image_service", start_ms=1., end_ms=3.,
                                     secret="RAW_EVENT_MUST_NOT_COPY"),
                                dict(kind="video_finish", start_ms=2., end_ms=5.)]))
        return rows

    def online(self, rows=None, status="completed", partial=None):
        self.rows = self.matrix() if rows is None else rows
        record = dict(schema_version="ditserve.optimizations.cache_factorial.v1", status=status,
                      rows=self.rows, comparisons={})
        for name, baseline, candidate in reporting.COMPARISONS:
            members = [dict(row, arm="baseline" if row["factorial_arm"] == baseline else "candidate")
                       for row in self.rows if row["factorial_arm"] in (baseline, candidate)]
            record["comparisons"][name] = paired_gate(members, expected_trace_ids=self.trace_ids)
        if partial is not None:
            record["active_trace_path"] = str(self.attempt / "partial.json")
            atomic_json(self.attempt / "partial.json", partial)
        atomic_json(self.attempt / "cache-online.json", record)
        self.online_record = record

    def terminal(self, status="passed", returncode=0, *, qualify=True):
        snapshot = self.output / "sources/cache_k4-attempt00"
        source = snapshot / "python/fixture.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# frozen fixture\n")
        hashes = {"python/fixture.py": file_digest(source)}
        source_sha = digest(hashes)
        atomic_json(snapshot / "manifest.json", dict(source_files=hashes, source_sha256=source_sha,
            execution_root=str(snapshot), config_sha256=self.config_sha))
        charge = dict(session_id="supplement-session", stage="cache_k4", gpu_count=1,
            gpu_seconds=100., elapsed_seconds=100., status="completed" if returncode == 0 else "failed",
            identity=dict(output=str(self.attempt), run_root=str(self.output), config_sha256=self.config_sha,
                source_sha256=source_sha, authorization_source=self.study["authorization"],
                cache_replay_from=self.study["replay"]))
        self.budget["sessions"] = [charge]
        atomic_json(self.ledger, self.budget)
        self.supervisor = dict(status=status, returncode=returncode, phase="cache_k4", output=str(self.attempt),
            charge=charge, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]),
            config_sha256=self.config_sha, source_sha256=source_sha, new_local_executions=0,
            online_evaluation_complete=status == "passed", online_qualified=qualify,
            reason="fixture only")
        self.save_supervisor()

    def save_supervisor(self):
        atomic_json(self.attempt / "supervisor.json", self.supervisor)
        atomic_json(self.output / "cache_k4/terminal.json", self.supervisor)

    def complete(self):
        self.setup_study()
        self.online()
        self.terminal()

    def test_not_started_report_is_explicit_and_creates_no_ledger(self):
        result = reporting.write_report(self.output, self.ledger)
        self.assertEqual(result["status"], "not_started")
        self.assertEqual(result["decision"], "未验证")
        self.assertFalse(result["archive_ready"])
        self.assertFalse(self.ledger.exists())
        self.assertEqual(result["new_local_executions"], 0)
        self.assertEqual(reporting.archive(self.output, self.ledger)["status"], "protected")
        self.assertIn("16 次轨迹", (self.output / "REPORT.md").read_text())

    def test_complete_matrix_has_four_96_request_arms_and_four_recomputed_gates(self):
        self.complete()
        report = reporting.collect(self.output, self.ledger)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["matrix_complete"])
        self.assertTrue(report["online_evaluation_complete"])
        self.assertTrue(report["online_qualified"])
        self.assertEqual(report["current"]["raw_request_count"], 384)
        self.assertEqual(report["current"]["complete_traces"], 16)
        for arm in reporting.ARMS:
            value = report["current"]["arms"][arm]
            self.assertEqual(value["all"]["arrived"], 96)
            self.assertEqual(value["images"]["latency_ms"]["n"], 88)
            self.assertEqual(value["videos"]["arrived"], 8)
            self.assertEqual(value["videos"]["on_time_fraction"], 1.)
        for trace in self.trace_ids:
            for arm in reporting.ARMS:
                value = report["current"]["per_trace_arms"][trace][arm]
                self.assertEqual(value["recorded_requests"], 48)
                self.assertEqual(value["images"]["sample_count"], 44)
                self.assertEqual(value["videos"]["arrived"], 4)
        self.assertEqual(len(report["current"]["comparisons"]), 4)

    def test_explicit_unlimited_study_auth_and_ledger_are_bound_and_rendered(self):
        self.complete()
        path = Path(self.study["authorization"]["path"])
        auth = read_json(path)
        auth.update(budget_unlimited=True, authorized_gpu_seconds=None)
        self.study.update(budget_unlimited=True, authorized_gpu_seconds=None,
                          authorization=self.bound(path, auth))
        self.budget.update(budget_unlimited=True, limit_gpu_seconds=None)
        atomic_json(self.output / "study.json", self.study)
        self.terminal()
        report = reporting.write_report(self.output, self.ledger)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["budget"]["budget_unlimited"])
        self.assertIsNone(report["budget"]["limit_gpu_seconds"])
        self.assertTrue(report["archive_ready"])
        self.assertTrue(report["online_qualified"])
        self.assertIn("无限（明确授权，仍完整计费）", (self.output / "REPORT.md").read_text())
        self.assertNotIn("Infinity", (self.output / "report.json").read_text())

    def test_null_limit_without_matching_explicit_unlimited_auth_never_qualifies(self):
        self.complete()
        self.budget.update(budget_unlimited=True, limit_gpu_seconds=None)
        atomic_json(self.ledger, self.budget)
        report = reporting.collect(self.output, self.ledger)
        self.assertTrue(report["errors"])
        self.assertFalse(report["archive_ready"])
        self.assertFalse(report["online_qualified"])
        self.assertNotIn("无限（明确授权", reporting.render_markdown(report))
        path = Path(self.study["authorization"]["path"])
        auth = read_json(path)
        auth["authorized_gpu_seconds"] = None
        self.study.update(authorized_gpu_seconds=None, authorization=self.bound(path, auth))
        atomic_json(self.output / "study.json", self.study)
        self.terminal()
        report = reporting.collect(self.output, self.ledger)
        self.assertTrue(any("authorization" in error for error in report["errors"]))
        self.assertFalse(report["online_qualified"])

    def test_image_p95_is_request_pooling_not_mean_of_per_trace_p95(self):
        self.setup_study()
        rows = self.matrix()
        members = [row for row in rows if row["factorial_arm"] == "off_k4"]
        for row, latency in zip(members, (1., 2., 3., 100.)):
            for request in row["requests"]:
                if request["kind"] == "image":
                    request["finish_ms"] = request["arrival_ms"] + latency
        self.online(rows)
        self.terminal(qualify=False)
        report = reporting.collect(self.output, self.ledger)
        images = report["current"]["arms"]["off_k4"]["images"]
        self.assertEqual(images["latency_p95_ms"], 100.)
        self.assertEqual(images["latency_mean_ms"], 26.5)
        self.assertEqual(images["sample_count"], 88)

    def test_partial_failed_arrivals_remain_in_denominator_and_no_survivor_promotion(self):
        self.setup_study()
        row = self.matrix()[0]
        row.update(status="failed", complete=False)
        row["requests"][0].update(status="failed", finish_ms=None)
        row["requests"][1].update(arrived=False, status="pending", finish_ms=None)
        self.online([], status="incomplete", partial=row)
        self.terminal(status="incomplete", returncode=1, qualify=False)
        report = reporting.collect(self.output, self.ledger)
        metrics = report["current"]["arms"]["off_step"]
        self.assertEqual(metrics["recorded_requests"], 24)
        self.assertEqual(metrics["unrecorded_expected_requests"], 72)
        self.assertEqual(metrics["all"]["arrived"], 23)
        self.assertEqual(metrics["all"]["completed"], 22)
        self.assertEqual(metrics["all"]["failed_or_incomplete"], 1)
        self.assertEqual(metrics["videos"]["on_time_fraction"], .5)
        self.assertEqual(metrics["all"]["goodput_rps"], 22 / 2.)
        self.assertFalse(report["online_evaluation_complete"])
        self.assertEqual(report["decision"], "未验证")
        self.assertTrue(all(not gate["qualified"] for gate in report["current"]["comparisons"].values()))

    def test_terminal_partial_duplicate_is_not_counted_twice(self):
        self.complete()
        self.online(partial=copy.deepcopy(self.rows[-1]))
        report = reporting.collect(self.output, self.ledger)
        self.assertEqual(report["current"]["raw_request_count"], 384)
        self.assertEqual(len(report["current"]["rows"]), 16)

    def test_missing_arm_duplicate_row_wrong_trace_and_request_identity_reject_matrix(self):
        self.complete()
        original = self.matrix()
        variants = [original[:-1], original[:-1] + [copy.deepcopy(original[0])]]
        wrong = copy.deepcopy(original)
        wrong[0]["requests"][0]["deadline_ms"] += 1
        variants.append(wrong)
        for rows in variants:
            with self.subTest(rows=len(rows)):
                self.online(rows)
                report = reporting.collect(self.output, self.ledger)
                self.assertFalse(report["matrix_complete"])
                self.assertFalse(report["online_qualified"])
                self.assertEqual(report["decision"], "未验证")

    def test_failed_supervisor_cannot_be_overridden_by_complete_passing_worker(self):
        self.complete()
        self.terminal(status="incomplete", returncode=1, qualify=True)
        report = reporting.collect(self.output, self.ledger)
        self.assertTrue(report["matrix_complete"])
        self.assertFalse(report["online_evaluation_complete"])
        self.assertFalse(report["online_qualified"])
        self.assertTrue(report["archive_ready"])
        self.assertTrue(all(gate["descriptive_only"] for gate in report["current"]["comparisons"].values()))

    def test_active_budget_adds_unsettled_time_and_prevents_archive(self):
        self.complete()
        self.budget["active"] = dict(started_monotonic=time.monotonic() - 12., gpu_count=1,
            identity=dict(output=str(self.attempt), phase="cache_k4"))
        atomic_json(self.ledger, self.budget)
        report = reporting.collect(self.output, self.ledger)
        self.assertEqual(report["status"], "running")
        self.assertGreaterEqual(report["budget"]["active_gpu_seconds_estimate"], 12.)
        self.assertGreaterEqual(report["budget"]["total_gpu_seconds_including_active"], 112.)
        self.assertFalse(report["archive_ready"])
        self.assertFalse(report["online_qualified"])

    def test_cleanup_requires_explicit_empty_lists_and_identical_settled_charge(self):
        self.complete()
        original = copy.deepcopy(self.supervisor)
        for change in (dict(final_occupancy=dict(status="exclusive_observed", foreign=[])),
                       dict(final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[1])),
                       dict(charge=dict(original["charge"], gpu_seconds=101.))):
            self.supervisor = dict(original, **change)
            self.save_supervisor()
            report = reporting.collect(self.output, self.ledger)
            self.assertFalse(report["online_qualified"])
            self.assertFalse(report["archive_ready"])

    def test_changed_source_authorization_or_frozen_math_blocks_archive(self):
        self.complete()
        paths = [Path(self.study["replay"]["bindings"]["local"]["path"]),
                 Path(self.study["authorization"]["path"]),
                 self.output / "sources/cache_k4-attempt00/python/fixture.py"]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            report = reporting.collect(self.output, self.ledger)
            self.assertTrue(report["errors"])
            self.assertFalse(report["archive_ready"])
            self.assertFalse(report["online_qualified"])
            path.write_bytes(original)

    def test_recomputed_gate_mismatch_is_not_silently_promoted(self):
        self.complete()
        self.online_record["comparisons"]["cache_on_k4"]["passed"] = False
        atomic_json(self.attempt / "cache-online.json", self.online_record)
        report = reporting.collect(self.output, self.ledger)
        self.assertTrue(any("recomputation" in error for error in report["errors"]))
        self.assertFalse(report["online_qualified"])
        self.assertFalse(report["archive_ready"])

    def test_observer_union_clips_overlaps_without_deducting_from_service_metrics(self):
        self.complete()
        row = self.rows[0]
        row["events"] += [dict(kind="partial_snapshot_build", start_ms=1., end_ms=4.),
            dict(kind="partial_flush", start_ms=2., end_ms=6.),
            dict(kind="partial_flush", start_ms=1999., end_ms=2005.)]
        self.online(self.rows)
        report = reporting.collect(self.output, self.ledger)
        observer = report["current"]["observer_by_arm"]["off_step"]
        self.assertEqual(observer["within_trace_union_ms"], 6.)
        self.assertEqual(observer["trace_window_ms"], 8000.)
        self.assertEqual(observer["fraction"], 6. / 8000.)
        self.assertEqual(report["current"]["arms"]["off_step"]["all"]["goodput_rps"], 96 / 8.)
        self.assertEqual(report["current"]["arms"]["off_step"]["images"]["latency_mean_ms"], 100.)

    def test_loading_capacity_warmup_and_reset_scoped_memory_metadata_are_source_bound(self):
        self.complete()
        self.bound(self.attempt / "loading.json", dict(status="completed", model_load_ms=11000.,
            peak_allocated_bytes=100, scope="loading interval"))
        self.bound(self.attempt / "capacity.json", dict(passed=True, frames=81, elapsed_ms=142000.,
            peak_allocated_bytes=200))
        self.bound(self.attempt / "cache-factorial-warmup.json", dict(status="completed", elapsed_ms=114000.,
            rows=[dict(arm="paired_k4", completed_steps=32, decoded=True, elapsed_ms=113000.,
                       memory_before_release=dict(peak_allocated_bytes=300))]))
        self.bound(self.attempt / "memory.json", dict(peak_allocated_bytes=400, peak_reserved_bytes=500,
            scope="since last warmup reset; not full process"))
        report = reporting.write_report(self.output, self.ledger)
        metadata = report["current"]["stage_metadata"]
        self.assertEqual(set(metadata), {"loading", "capacity", "cache-factorial-warmup", "memory"})
        self.assertEqual(metadata["cache-factorial-warmup"]["evidence"]["rows"][0]["completed_steps"], 32)
        self.assertEqual(metadata["memory"]["source"]["sha256"], file_digest(self.attempt / "memory.json"))
        markdown = (self.output / "REPORT.md").read_text()
        self.assertIn("since last warmup reset; not full process", markdown)
        self.assertIn("日志开销未从服务指标中扣除", markdown)

    def test_real_excel_readback_excludes_old_local_rows_and_raw_events_and_never_prunes(self):
        self.complete()
        before = {path: file_digest(path) for path in self.old.parent.rglob("*.json")}
        report = reporting.write_report(self.output, self.ledger)
        archived = reporting.archive(self.output, self.ledger)
        self.assertEqual(archived["status"], "verified")
        self.assertEqual(archived["verification"]["status"], "verified")
        self.assertEqual(archived["originals_deleted"], 0)
        self.assertEqual(archived["original_local_rows_copied"], 0)
        with gzip.open(self.output / "excel/data.json.gz", "rt") as stream:
            payload = stream.read()
        self.assertNotIn("ORIGINAL_LOCAL_ROW_MUST_NOT_COPY", payload)
        self.assertNotIn("RAW_EVENT_MUST_NOT_COPY", payload)
        self.assertIn("overlapping/nested host intervals", payload)
        self.assertIn("384", payload)
        self.assertEqual(report["local_reference"]["original_rows"], 84)
        for path, expected in before.items():
            self.assertEqual(file_digest(path), expected)
        self.assertTrue((self.attempt / "cache-online.json").exists())
        with self.assertRaises(FileExistsError):
            reporting.archive(self.output, self.ledger)

    def test_report_refuses_historical_write_target(self):
        with self.assertRaises(ValueError):
            reporting.write_report(reporting.HISTORICAL_OUTPUT, reporting.HISTORICAL_LEDGER)


if __name__ == "__main__":
    unittest.main()
