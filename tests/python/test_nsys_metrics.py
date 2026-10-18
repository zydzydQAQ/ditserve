"""Synthetic fixtures use the NVIDIA Nsight Systems 2023.4 documented DDL."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation.nsys_metrics import summarize_nsys_metrics


class NsysMetricTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "metrics.sqlite"
        with sqlite3.connect(self.path) as db:
            db.executescript("""
                CREATE TABLE TARGET_INFO_GPU_METRICS (
                    typeId INTEGER NOT NULL,sourceId INTEGER NOT NULL,typeName TEXT NOT NULL,
                    metricId INTEGER NOT NULL,metricName TEXT NOT NULL);
                CREATE TABLE GPU_METRICS (
                    timestamp INTEGER,typeId INTEGER NOT NULL,metricId INTEGER NOT NULL,value INTEGER NOT NULL);
                CREATE TABLE NVTX_EVENTS (
                    start INTEGER NOT NULL,end INTEGER,eventType INTEGER NOT NULL,
                    text TEXT,textId INTEGER,globalTid INTEGER);
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT NOT NULL);
            """)
            db.execute("INSERT INTO NVTX_EVENTS VALUES (0,10000000,59,'wan_profile/request_17',NULL,42)")
            for index, name in enumerate(("SM Active", "Tensor Active", "DRAM Read Bandwidth", "DRAM Write Bandwidth")):
                db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (256,1,'GA100',?,?)", (index, name))
                db.executemany("INSERT INTO GPU_METRICS VALUES (?,256,?,?)",
                               [(timestamp * 1000000, index, timestamp * 5) for timestamp in range(11)])

    def result(self, **kwargs):
        result = summarize_nsys_metrics(self.path, **kwargs)
        json.dumps(result, allow_nan=False)
        return result

    def metric(self, result, metric_id="sm_active_pct", index=0):
        return next(m for m in result["ranges"][index]["metrics"] if m["id"] == metric_id)

    def test_known_schema_time_weighted_mean_and_provenance(self):
        result = self.result()
        self.assertEqual(result["status"], "sampled")
        metric = self.metric(result)
        self.assertEqual(metric["time_weighted_mean_pct"], 25)
        self.assertEqual(metric["coverage_fraction"], 1)
        self.assertEqual(metric["sample_count"], 11)
        self.assertEqual((metric["sample_min_pct"], metric["sample_max_pct"]), (0, 50))
        self.assertEqual(metric["provenance"][0]["metric_name"], "SM Active")
        self.assertEqual(result["ranges"][0]["frames"], 17)
        self.assertEqual(len(result["provenance"]["sha256"]), 64)
        self.assertIn("not per-kernel NCU", result["provenance"]["scope_limitation"])

    def test_actual_export_empty_type_name_throughput_units_and_integer_percentages(self):
        # Observed Nsight Systems 2023.4.4 schema: empty typeName is legal and
        # type/source IDs are large serialized IDs, not CUDA device indices.
        type_id = 281479271677952
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE TARGET_INFO_GPU_METRICS SET typeId=?,sourceId=?,typeName='',metricName=metricName || ' [Throughput %]'",
                       (type_id, type_id))
            db.execute("UPDATE GPU_METRICS SET typeId=?", (type_id,))
        result = self.result()
        self.assertEqual(result["status"], "sampled")
        self.assertEqual(result["selected_type_id"], type_id)
        self.assertEqual(result["available_streams"], [dict(type_id=type_id, source_id=type_id, type_name="")])
        for metric in result["ranges"][0]["metrics"]:
            self.assertEqual(metric["time_weighted_mean_pct"], 25.0)
            self.assertEqual(metric["provenance"][0]["type_name"], "")
            self.assertTrue(metric["provenance"][0]["type_name_empty"])
            self.assertEqual(metric["provenance"][0]["source_unit_label"], "Throughput %")
            self.assertEqual(metric["provenance"][0]["value_scale"], 1.0)

    def test_throughput_label_does_not_rescale_or_accept_invalid_counter_values(self):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE TARGET_INFO_GPU_METRICS SET typeName='',metricName=metricName || ' [Throughput %]'")
            db.execute("UPDATE GPU_METRICS SET value=100000000 WHERE metricId=0 AND timestamp=5000000")
        result = self.result(minimum_coverage=0.5)
        self.assertEqual(self.metric(result)["status"], "unknown")
        self.assertIsNone(self.metric(result)["time_weighted_mean_pct"])
        self.assertEqual(self.metric(result)["source_rows"]["invalid_rows"], 1)
        self.assertEqual(self.metric(result, "tensor_active_pct")["time_weighted_mean_pct"], 25.0)

    def test_nonpercentage_unit_suffixes_remain_unknown(self):
        for suffix in (" [Avg]", " [MHz]", " [Throughput GB/s]", " [Throughput %] Hz", " [%/s]"):
            with self.subTest(suffix=suffix):
                with sqlite3.connect(self.path) as db:
                    db.execute("UPDATE TARGET_INFO_GPU_METRICS SET metricName=? WHERE metricId=0", ("SM Active" + suffix,))
                metric = self.metric(self.result())
                self.assertEqual(metric["status"], "unknown")
                self.assertIn("metric_not_collected", metric["reason_codes"])
                self.assertIsNone(metric["time_weighted_mean_pct"])

    def test_empty_display_names_do_not_make_ambiguous_source_identifiers_safe(self):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE TARGET_INFO_GPU_METRICS SET typeName=''")
            db.execute("UPDATE TARGET_INFO_GPU_METRICS SET sourceId=2 WHERE metricId=1")
        result = self.result()
        self.assertEqual(result["status"], "unknown")
        self.assertIn("ambiguous_gpu_metric_source", result["reason_codes"])
        self.assertTrue(all(not metric["measured"] for metric in result["ranges"][0]["metrics"]))

    def test_irregular_spacing_is_time_weighted_and_range_clipped(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM GPU_METRICS WHERE metricId=0")
            db.executemany("INSERT INTO GPU_METRICS VALUES (?,256,0,?)",
                           [(0, 0), (1000000, 100), (10000000, 100)])
            db.execute("UPDATE NVTX_EVENTS SET start=500000,end=9500000")
        result = self.result(minimum_samples=2, max_gap_periods=10)
        metric = self.metric(result)
        # [0.5,1] has linearly varying 50..100, then [1,9.5] stays at 100.
        self.assertAlmostEqual(metric["covered_time_mean_estimate_pct"], (0.5 * 75 + 8.5 * 100) / 9)
        self.assertEqual(metric["status"], "unknown")  # Only one actual sample lies in the range.
        self.assertIsNone(metric["time_weighted_mean_pct"])

    def test_gaps_are_not_interpolated_or_zero_filled(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM GPU_METRICS WHERE timestamp BETWEEN 3000000 AND 7000000")
        metric = self.metric(self.result())
        self.assertEqual(metric["status"], "unknown")
        self.assertIsNone(metric["time_weighted_mean_pct"])
        self.assertEqual(metric["coverage_fraction"], 0.4)
        self.assertEqual(metric["covered_time_mean_estimate_pct"], 25)
        self.assertEqual(metric["missing_duration_ns"], 6000000)

    def test_boundary_holes_are_not_extrapolated(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM GPU_METRICS WHERE timestamp IN (0,10000000)")
        metric = self.metric(self.result())
        self.assertEqual(metric["coverage_fraction"], 0.8)
        self.assertEqual(metric["missing_duration_ns"], 2000000)
        self.assertIsNone(metric["time_weighted_mean_pct"])

    def test_type_id_and_metric_id_are_joint_keys(self):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (257,2,'Another GPU',0,'SM Active')")
            db.executemany("INSERT INTO GPU_METRICS VALUES (?,257,0,99)", [(x * 1000000,) for x in range(11)])
        ambiguous = self.result()
        self.assertEqual(ambiguous["status"], "unknown")
        self.assertIn("ambiguous_gpu_metric_stream", ambiguous["reason_codes"])
        selected = self.result(gpu_type_id=256)
        self.assertEqual(self.metric(selected)["time_weighted_mean_pct"], 25)
        second = self.result(gpu_type_id=257)
        self.assertEqual(self.metric(second)["time_weighted_mean_pct"], 99)
        self.assertEqual(second["status"], "partial")
        missing = self.result(gpu_type_id=999)
        self.assertEqual(missing["status"], "unknown")
        self.assertIn("requested_gpu_metric_stream_missing", missing["reason_codes"])

    def test_registered_nested_ranges_are_separate_and_incomplete_ranges_unknown(self):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO StringIds VALUES (7,'wan_profile/denoise_17')")
            db.execute("INSERT INTO NVTX_EVENTS VALUES (1000000,9000000,60,NULL,7,42)")
            db.execute("INSERT INTO NVTX_EVENTS VALUES (0,NULL,59,'request_33',NULL,42)")
        result = self.result(minimum_samples=5)
        self.assertEqual(len(result["ranges"]), 3)
        denoise = next(r for r in result["ranges"] if r["scope"] == "denoise")
        self.assertEqual(denoise["duration_ns"], 8000000)
        incomplete = next(r for r in result["ranges"] if r["frames"] == 33)
        self.assertEqual(incomplete["status"], "unknown")
        self.assertTrue(all(m["time_weighted_mean_pct"] is None for m in incomplete["metrics"]))

    def test_bad_samples_and_duplicate_conflicts_are_not_claimed_as_measurements(self):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO GPU_METRICS VALUES (1000000,256,0,100)")
            db.execute("UPDATE GPU_METRICS SET value=101 WHERE metricId=1 AND timestamp=5000000")
        result = self.result(minimum_coverage=0.5)
        for kind in ("sm_active_pct", "tensor_active_pct"):
            metric = self.metric(result, kind)
            self.assertIsNone(metric["time_weighted_mean_pct"])
            self.assertIn("invalid_or_conflicting_source_samples", metric["reason_codes"])

    def test_unverified_metric_names_are_not_renamed_to_known_counters(self):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE TARGET_INFO_GPU_METRICS SET metricName='Tensor Active / FP16 Active' WHERE metricId=1")
            db.execute("UPDATE TARGET_INFO_GPU_METRICS SET metricName='DRAM Bandwidth GB/s' WHERE metricId=2")
        result = self.result()
        self.assertIsNone(self.metric(result, "tensor_active_pct")["time_weighted_mean_pct"])
        self.assertIsNone(self.metric(result, "dram_read_active_pct")["time_weighted_mean_pct"])
        self.assertIn("DRAM Bandwidth GB/s", result["unrecognized_metrics"])

    def test_empty_unknown_schema_and_explicit_row_limit_preserve_unknown(self):
        metric = self.metric(self.result(maximum_samples=10))
        self.assertIn("source_sample_limit_exceeded", metric["reason_codes"])
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE GPU_METRICS")
        result = self.result()
        self.assertEqual(result["status"], "unknown")
        self.assertIn("unsupported_or_missing_sqlite_schema", result["reason_codes"])

    def test_active_sidecar_is_not_read_and_no_output_files_are_created(self):
        sidecar = Path(str(self.path) + "-wal")
        sidecar.touch()
        before = set(self.path.parent.iterdir())
        result = self.result()
        self.assertIn("sqlite_export_has_live_sidecar", result["reason_codes"])
        self.assertEqual(set(self.path.parent.iterdir()), before)

    def test_source_change_invalidates_individual_means_too(self):
        with patch("ditserve.evaluation.nsys_metrics._sha256", side_effect=["a" * 64, "b" * 64]):
            result = self.result()
        self.assertEqual(result["status"], "unknown")
        self.assertIn("sqlite_export_changed_during_read", result["reason_codes"])
        self.assertTrue(all(m["time_weighted_mean_pct"] is None for r in result["ranges"] for m in r["metrics"]))


if __name__ == "__main__":
    unittest.main()
