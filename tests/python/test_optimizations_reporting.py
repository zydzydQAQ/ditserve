"""Tiny CPU fixtures: no model loading, GPU probes or real result cleanup."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from ditserve.optimizations.reporting import archive, collect, write_report, _preview, _video_pair, _write_video, _setup_windows, _failed_image_pairs
from ditserve.optimizations.protocol import PHASES, TOTAL, LIMITS
from ditserve.research.common import atomic_json


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.ledger = self.root / "gpu-budget.json"
        atomic_json(self.ledger, dict(active=None, used_gpu_seconds=120., limit_gpu_seconds=TOTAL,
                    stage_used_gpu_seconds={phase: 0. for phase in LIMITS}, sessions=[]))

    def tearDown(self):
        self.temp.cleanup()

    def phase(self, phase="sana_compile", status="passed", **extra):
        attempt = self.output / phase / "attempt00"
        attempt.mkdir(parents=True, exist_ok=True)
        record = dict(status=status, output=str(attempt), returncode=0, charge=dict(gpu_seconds=2.),
            local_qualified=True, online_qualified=True, decision="值得继续", reason="fixture only",
            final_occupancy=dict(status="exclusive_observed"), **extra)
        atomic_json(self.output / phase / "terminal.json", record)
        return attempt

    def all_terminal(self):
        for phase in PHASES:
            self.phase(phase)

    def local(self, attempt, rows, **extra):
        atomic_json(attempt / "sana-compile.json", dict(schema_version="ditserve.optimizations.sana.v1",
            status="passed", rows=rows, local_qualified=True, summaries=dict(eager=dict(n=2, mean=12.)), **extra))

    def online(self, attempt, complete=True):
        row = dict(trace_sha256="fixture", order_index=0, arm="baseline", complete=complete,
            status="completed" if complete else "interrupted",
            metrics=dict(arrived=2, completed=1, goodput_rps=.5),
            image_metrics=dict(sample_count=1, latency_p95_ms=20.),
            video_metrics=dict(arrived=1, completed=0, on_time=0),
            requests=[dict(request_id="r000", kind="image", arrived=True, status="completed", finish_ms=20.),
                      dict(request_id="r001", kind="video", arrived=True, status="failed")],
            events=[dict(kind="video_finish", start_ms=10., end_ms=30., secret="RAW_EVENT_MUST_NOT_BE_IN_EXCEL")])
        atomic_json(attempt / "online.json", dict(status="completed" if complete else "incomplete", rows=[row]))

    def test_supervisor_failure_does_not_promote_local_or_online_partial(self):
        attempt = self.phase(status="failed")
        self.local(attempt, [dict(kind="measurement", backend="compiled", service=dict(ms=10.))])
        self.online(attempt, complete=False)
        report = write_report(self.output, self.ledger)
        phase = report["phases"]["sana_compile"]
        self.assertFalse(phase["local_qualified"])
        self.assertFalse(phase["online_qualified"])
        self.assertEqual(phase["decision"], "未验证")
        self.assertFalse(phase["online"][0]["authoritative"])
        self.assertEqual(phase["sample_counts"]["online_requests"], 2)
        self.assertEqual(phase["event_summaries"][0]["intervals"]["video_finish"]["mean"], 20)
        self.assertIn("描述性 p95", (self.output / "REPORT.md").read_text())

    def trace(self, arm="baseline", order=0, **extra):
        return dict(trace_sha256="fixture-trace", order_index=order, arm=arm,
            status="completed", complete=True,
            requests=[dict(request_id="r0", status="completed", arrived=True)],
            events=[dict(kind="partial_flush", start_ms=3., end_ms=5.)], **extra)

    def test_external_partial_has_its_actual_source_and_never_grants_qualification(self):
        attempt = self.phase()
        active = self.trace("candidate")
        atomic_json(attempt / "partial.json", active)
        atomic_json(attempt / "online.json", dict(status="completed", rows=[self.trace()],
            active_trace_path=str(attempt / "partial.json"), qualification=dict(passed=True)))
        phase = collect(self.output, self.ledger)["phases"]["sana_compile"]
        self.assertEqual(len(phase["online"]), 2)
        partial = phase["online"][1]
        self.assertEqual(partial["source"]["path"], str(attempt / "partial.json"))
        self.assertEqual(partial["source"]["sha256"], hashlib.sha256((attempt / "partial.json").read_bytes()).hexdigest())
        self.assertEqual(partial["pointer"], "")
        self.assertFalse(partial["complete"])
        self.assertFalse(partial["authoritative"])
        self.assertFalse(phase["online_qualified"])
        self.assertEqual(phase["request_samples"][1]["pointer"], "/requests/0")
        self.assertEqual(phase["request_samples"][1]["source"], partial["source"])
        self.assertEqual(phase["event_summaries"][1]["pointer"], "/events")
        self.assertEqual(phase["event_summaries"][1]["source"], partial["source"])

    def test_merged_external_partial_is_not_counted_twice_or_revoke_complete_result(self):
        attempt = self.phase()
        row = self.trace()
        atomic_json(attempt / "partial.json", row)
        atomic_json(attempt / "online.json", dict(status="completed", rows=[row],
            active_trace_path="partial.json", qualification=dict(passed=True)))
        phase = collect(self.output, self.ledger)["phases"]["sana_compile"]
        self.assertEqual(len(phase["online"]), 1)
        self.assertEqual(len(phase["request_samples"]), 1)
        self.assertEqual(phase["online"][0]["pointer"], "/rows/0")
        self.assertTrue(phase["online_qualified"])

    def test_partial_identity_keeps_different_order_or_arm(self):
        attempt = self.phase()
        for active in (self.trace("candidate"), self.trace(order=1)):
            with self.subTest(active=(active["arm"], active["order_index"])):
                atomic_json(attempt / "partial.json", active)
                atomic_json(attempt / "online.json", dict(status="running", rows=[self.trace()],
                    active_trace_path="partial.json"))
                phase = collect(self.output, self.ledger)["phases"]["sana_compile"]
                self.assertEqual(len(phase["online"]), 2)
                self.assertFalse(phase["online"][1]["complete"])

    def test_legacy_inline_partial_remains_source_bound_and_deduplicates_merged_trace(self):
        attempt = self.phase()
        for arm, count in (("candidate", 2), ("baseline", 1)):
            atomic_json(attempt / "online.json", dict(status="running", rows=[self.trace()], active_trace=self.trace(arm)))
            phase = collect(self.output, self.ledger)["phases"]["sana_compile"]
            self.assertEqual(len(phase["online"]), count)
            if count == 2:
                self.assertEqual(phase["online"][1]["pointer"], "/active_trace")
                self.assertEqual(phase["request_samples"][1]["pointer"], "/active_trace/requests/0")
                self.assertEqual(phase["online"][1]["source"]["path"], str(attempt / "online.json"))
                self.assertFalse(phase["online"][1]["complete"])

    def test_unsafe_or_invalid_external_partial_rejected_without_dropping_cumulative_rows(self):
        attempt = self.phase()
        other = self.output / "other/attempt00"
        other.mkdir(parents=True)
        atomic_json(other / "partial.json", self.trace("candidate"))
        outside = self.root / "partial.json"
        atomic_json(outside, self.trace("candidate"))
        for case in ("other_attempt", "outside_run", "symlink", "oversized", "missing_identity"):
            with self.subTest(case=case):
                partial = attempt / "partial.json"
                if partial.exists() or partial.is_symlink():
                    partial.unlink()
                reference = str(partial)
                if case == "other_attempt":
                    reference = str(other / "partial.json")
                elif case == "outside_run":
                    reference = str(outside)
                elif case == "symlink":
                    partial.symlink_to(other / "partial.json")
                elif case == "oversized":
                    atomic_json(partial, dict(self.trace("candidate"), padding="x"*2000))
                else:
                    atomic_json(partial, dict(status="completed", complete=True, requests=[]))
                atomic_json(attempt / "online.json", dict(status="completed", rows=[self.trace()],
                    active_trace_path=reference, qualification=dict(passed=True)))
                with patch("ditserve.optimizations.reporting.MAX_METADATA_BYTES", 1800):
                    phase = collect(self.output, self.ledger)["phases"]["sana_compile"]
                self.assertEqual(len(phase["online"]), 1)
                self.assertTrue(phase["errors"])
                self.assertFalse(phase["online_qualified"])

    def test_live_cache_factorial_partial_visible_before_terminal_and_uses_full_arm_identity(self):
        attempt = self.output / "cache_k4/attempt00"
        attempt.mkdir(parents=True)
        baseline = self.trace("off_k4", factorial_arm="off_k4", policy="block-4", cache=False)
        active = self.trace("paired_k4", factorial_arm="paired_k4", policy="block-4", cache=True)
        atomic_json(attempt / "partial.json", active)
        atomic_json(attempt / "cache-online.json", dict(schema_version="ditserve.optimizations.cache_factorial.v1",
            status="running", rows=[baseline], active_trace_path="partial.json"))
        phase = collect(self.output, self.ledger)["phases"]["cache_k4"]
        self.assertEqual(phase["status"], "partial")
        self.assertEqual(len(phase["online"]), 2)
        self.assertTrue(phase["online"][1]["evidence"]["cache"])
        self.assertEqual(phase["online"][1]["evidence"]["factorial_arm"], "paired_k4")
        self.assertEqual(phase["online"][1]["source"]["path"], str(attempt / "partial.json"))
        self.assertFalse(phase["online_qualified"])

    def test_persistence_sidecar_is_bounded_source_linked_descriptive_metadata(self):
        attempt = self.phase()
        path = attempt / "online-flushes.json"
        atomic_json(path, dict(schema_version="ditserve.host_persistence.v1",
            flushes=[dict(kind="cumulative", host_ms=2., status="completed"),
                     dict(kind="preparing", host_ms=4., status="failed")]))
        atomic_json(attempt / "online.json", dict(status="running", rows=[], flush_log_path=str(path)))
        phase = collect(self.output, self.ledger)["phases"]["sana_compile"]
        summary = phase["persistence_summary"]
        self.assertEqual(summary["source"]["path"], str(path))
        self.assertEqual(summary["pointer"], "/flushes")
        self.assertEqual(summary["flush_count"], 2)
        self.assertEqual(summary["host_ms"]["mean"], 3.)
        self.assertEqual(summary["status_counts"], dict(completed=1, failed=1))
        self.assertEqual(phase["local_samples"], [])
        self.assertFalse(phase["online_qualified"])

    def test_local_stats_keep_order_samples_source_hash_and_failures(self):
        attempt = self.phase()
        rows = [dict(kind="measurement", backend="eager", round=0, service=dict(ms=value),
                     quality=dict(passed=value != 14., metrics=dict(lpips_mean=.1))) for value in (10., 14.)]
        self.local(attempt, rows)
        phase = collect(self.output, self.ledger)["phases"]["sana_compile"]
        group = phase["local"][0]["measured_groups"][0]
        self.assertEqual(group["service_ms"]["n"], 2)
        self.assertEqual(group["service_ms"]["mean"], 12.)
        self.assertAlmostEqual(group["service_ms"]["p95"], 13.8)
        self.assertEqual(len(phase["quality_failures"]), 1)
        source = phase["local"][0]["source"]
        self.assertEqual(source["sha256"], hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest())

    def test_complete_negative_online_gate_is_measured_not_unverified(self):
        attempt = self.phase()
        path = self.output / "sana_compile/terminal.json"
        terminal = json.loads(path.read_text())
        terminal.update(online_qualified=False, decision="未验证")
        atomic_json(path, terminal)
        row = dict(status="completed", complete=True,
                   requests=[dict(status="completed", arrived=True)])
        atomic_json(attempt / "online.json", dict(status="completed", rows=[row],
            qualification=dict(passed=False, failures=[], pooled=dict(system_gain=False))))
        value = collect(self.output, self.ledger)["phases"]["sana_compile"]
        self.assertTrue(value["local_qualified"])
        self.assertFalse(value["online_qualified"])
        self.assertTrue(value["online_evaluation_complete"])
        self.assertEqual(value["decision"], "当前配置无收益")
        self.assertEqual(json.loads(path.read_text())["decision"], "未验证")

    def test_stage_top_level_pairs_include_quality_failures(self):
        attempt = self.phase("text_encoder")
        atomic_json(attempt / "text_encoder.json", dict(schema_version="ditserve.optimizations.stages.v1",
            status="passed", rows=[dict(input_id="p0", arm="gpu", encode_ms=5., completed=True)],
            pairs=[dict(input_id="p0", order=0, quality=dict(passed=False, metrics=dict(lpips_mean=.2)))]))
        value = collect(self.output, self.ledger)["phases"]["text_encoder"]
        self.assertEqual(len(value["local_samples"]), 2)
        self.assertEqual(value["quality_failures"][0]["pointer"], "/pairs/0/quality")

    def test_tolerated_cross_backend_numerical_difference_is_not_quality_failure(self):
        attempt = self.phase()
        self.local(attempt, [dict(kind="measurement", quality=dict(passed=True,
            metrics=dict(lpips_mean=.001), numerical=dict(passed=False, max_abs=.02)))])
        value = collect(self.output, self.ledger)["phases"]["sana_compile"]
        self.assertEqual(value["quality_failures"], [])

    def test_cache_pending_uses_producer_full_qualification_and_requires_complete_pass(self):
        attempt = self.phase("cache_k4")
        cases = [({}, True),
                 (dict(status="not_measured_budget_limited", complete=False, qualified=False), True),
                 (dict(status="failed", complete=True, qualified=False), True),
                 (dict(status="incomplete", complete=False, qualified=True), True),
                 (dict(status="passed", complete=True, qualified=True), False)]
        for full, expected in cases:
            with self.subTest(full_qualification=full):
                atomic_json(attempt / "cache-k4.json", dict(schema_version="ditserve.optimizations.wan.v1",
                    status="passed", rows=[], required_full_qualification="8dev x2orders +4heldout",
                    full_qualification=full,
                    qualification=dict(complete=True, qualified=True)))
                phase = collect(self.output, self.ledger)["phases"]["cache_k4"]
                self.assertEqual(phase["local"][0]["continuation_pending"], expected)
                self.assertFalse(phase["online_qualified"])

    def test_cache_online_budget_reason_is_visible_without_mutating_terminal_or_gates(self):
        original_reason = "finite eager cache/K4 pilot; online qualification separate"
        online_reason = "remaining allowance cannot admit complete four-arm AB/BA factorial"
        attempt = self.phase("cache_k4", online_unverified_reason=online_reason)
        path = self.output / "cache_k4/terminal.json"
        terminal = json.loads(path.read_text())
        terminal.update(reason=original_reason, decision="未验证", online_qualified=False)
        atomic_json(path, terminal)
        self.cache_evidence(attempt, [])
        originals = {p: p.read_bytes() for p in (path, self.ledger)}
        report = write_report(self.output, self.ledger)
        value = report["phases"]["cache_k4"]
        self.assertEqual(value["reported_reason"], original_reason)
        self.assertEqual(value["online_unverified_reason"], online_reason)
        self.assertIn(original_reason, value["reason"])
        self.assertIn(online_reason, value["reason"])
        self.assertIn(online_reason, (self.output / "REPORT.md").read_text())
        self.assertTrue(value["local_qualified"])
        self.assertFalse(value["online_qualified"])
        self.assertEqual(value["decision"], "未验证")
        self.assertEqual(value["terminal_source"]["sha256"], hashlib.sha256(originals[path]).hexdigest())
        for p, original in originals.items():
            self.assertEqual(p.read_bytes(), original)

    def test_completed_cache_qualification_does_not_close_active_or_budget_stopped_work(self):
        attempt = self.phase("cache_k4")
        for pending in (dict(active_unit={"kind": "online"}), dict(budget_stopped=True)):
            with self.subTest(pending=pending):
                atomic_json(attempt / "cache-k4.json", dict(schema_version="ditserve.optimizations.wan.v1",
                    status="stopped", rows=[], required_full_qualification="8dev x2orders +4heldout",
                    full_qualification=dict(status="passed", complete=True, qualified=True), **pending))
                phase = collect(self.output, self.ledger)["phases"]["cache_k4"]
                self.assertTrue(phase["local"][0]["continuation_pending"])

    def test_archive_refuses_missing_phase_or_active_ledger_without_creating_staging(self):
        self.phase()
        self.assertEqual(archive(self.output, self.ledger)["status"], "protected")
        self.all_terminal()
        data = json.loads(self.ledger.read_text())
        data["active"] = dict(started_monotonic=0., gpu_count=1)
        atomic_json(self.ledger, data)
        self.assertEqual(archive(self.output, self.ledger)["status"], "protected")
        self.assertFalse((self.output / "archive-metadata").exists())

    def test_compact_archive_contains_small_samples_not_events_or_compiler_cache(self):
        self.all_terminal()
        attempt = self.output / "sana_compile" / "attempt00"
        self.local(attempt, [dict(kind="measurement", backend="eager", service=dict(ms=12.))])
        self.online(attempt)
        compiler = attempt / "compiler-cache"
        compiler.mkdir()
        (compiler / "huge.json").write_text('{"private_cache":"COMPILER_CACHE_NOT_EXPORTED"}')
        result = archive(self.output, self.ledger, preview_limit=0)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["originals_deleted"], 0)
        self.assertFalse(result["original_prune_authorized"])
        with gzip.open(self.output / "excel/data.json.gz", "rt") as stream:
            payload = stream.read()
        self.assertNotIn("RAW_EVENT_MUST_NOT_BE_IN_EXCEL", payload)
        self.assertNotIn("COMPILER_CACHE_NOT_EXPORTED", payload)
        self.assertIn("r001", payload)
        self.assertIn("failed", payload)
        self.assertTrue((attempt / "online.json").exists())
        self.assertTrue((compiler / "huge.json").exists())

    def test_previews_are_failure_first_capped_and_cross_root_raw_is_retained(self):
        self.all_terminal()
        attempt = self.output / "sana_compile" / "attempt00"
        rows = []
        for index in range(6):
            path = (self.root if index == 5 else attempt) / ("raw%d.npy" % index)
            path.write_bytes(("raw%d" % index).encode())
            rows.append(dict(kind="measurement", input_id=str(index),
                quality=dict(passed=index != 5),
                artifact=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        self.local(attempt, rows, required_full_qualification="pending larger matrix")
        selected = []
        def preview(path, output):
            selected.append(path)
            output.write_bytes(b"tiny fixture preview")
            return dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        with patch("ditserve.optimizations.reporting._preview", side_effect=preview):
            result = archive(self.output, self.ledger)
        self.assertEqual(result["preview_count"], 4)
        self.assertEqual(selected[0].name, "raw5.npy")
        retention = json.loads(Path(result["retention_plan"]).read_text())
        outside = next(row for row in retention["original_artifacts"] if row["path"].endswith("raw5.npy"))
        self.assertIn("outside output", outside["retention_reason"])
        self.assertTrue(all(not row["deletable"] for row in retention["original_artifacts"]))
        self.assertTrue(all(Path(row["path"]).exists() for row in retention["original_artifacts"]))
        self.assertTrue(all(row["continuation_or_qualification_pending"] for row in retention["original_artifacts"]))

    def test_raw_hash_mismatch_never_previews_or_deletes(self):
        self.all_terminal()
        attempt = self.output / "sana_compile" / "attempt00"
        path = attempt / "wrong.npy"
        path.write_bytes(b"preserve this")
        self.local(attempt, [dict(artifact=dict(path=str(path), sha256="wrong"))])
        with patch("ditserve.optimizations.reporting._preview") as preview:
            result = archive(self.output, self.ledger)
        preview.assert_not_called()
        plan = json.loads(Path(result["retention_plan"]).read_text())
        self.assertFalse(plan["original_artifacts"][0]["hash_matches"])
        self.assertTrue(path.exists())

    def test_sana_failed_batch_images_keep_exact_two_reference_pairs_within_four_pngs(self):
        self.all_terminal()
        attempt = self.output / "sana_batch/attempt00"
        def artifact(name):
            path = attempt / (name + ".npy")
            path.write_bytes(name.encode())
            return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=[1,1024,1024,3])
        # The unrelated alphabetically earlier reference must not displace z.
        rows = [dict(kind="eager_reference", input_id=name, artifact=artifact("ref-" + name))
                for name in ("a", "b-unrelated", "z")]
        rows.append(dict(kind="measurement", round=0, repeat=0, batch_size=2, input_ids=["a", "z"],
            quality=[dict(input_id=name, passed=False, metrics=dict(lpips_mean=.02), artifact=artifact("batch-" + name))
                     for name in ("a", "z")]))
        atomic_json(attempt / "sana-batch.json", dict(schema_version="ditserve.optimizations.sana.v1", status="stopped", rows=rows))
        value = collect(self.output, self.ledger)["phases"]["sana_batch"]
        candidates = [item for item in value["artifacts"] if item["arm"] == "batch-2"]
        self.assertEqual([item["input_id"] for item in candidates], ["a", "z"])
        self.assertEqual([item["image_quality"]["pointer"] for item in candidates],
                         ["/rows/3/quality/0", "/rows/3/quality/1"])
        pairs = _failed_image_pairs(value["artifacts"])
        self.assertEqual([[item["input_id"] for item in pair] for pair in pairs], [["a","a"],["z","z"]])
        def render(path, output):
            output.write_bytes(b"tiny fake PNG")
            return dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        with patch("ditserve.optimizations.reporting._preview", side_effect=render) as preview:
            result = archive(self.output, self.ledger)
        self.assertEqual(result["preview_count"], 4)
        self.assertEqual({call.args[0].name for call in preview.call_args_list},
                         {"ref-a.npy", "ref-z.npy", "batch-a.npy", "batch-z.npy"})

    def test_sana_image_pairing_refuses_ambiguous_wrong_input_order_or_source(self):
        base = dict(phase="sana_compile", input_id="image", source=dict(path="same.json", sha256="a"*64),
                    shape=[1,1024,1024,3], kind="measurement", order=0, repeat=0)
        reference = dict(base, path="ref.npy", arm="eager", failed=False)
        candidate = dict(base, path="candidate.npy", arm="compiled", failed=True)
        self.assertEqual(_failed_image_pairs([reference,candidate]), [[reference,candidate]])
        variants = [dict(reference,input_id="other"), dict(reference,order=1), dict(reference,repeat=1),
                    dict(reference,source=dict(path="same.json",sha256="b"*64))]
        for wrong in variants:
            self.assertEqual(_failed_image_pairs([wrong,candidate]), [])
        self.assertEqual(_failed_image_pairs([reference,reference,candidate]), [])
        self.assertEqual(_failed_image_pairs([reference,candidate,candidate]), [])
        self.assertEqual(_failed_image_pairs([dict(reference,source={}),dict(candidate,source={})]), [])

    def test_tiny_actual_video_preview_is_a_contact_sheet(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("optional CPU preview dependencies absent")
        source = self.root / "tiny.npy"
        np.save(source, np.zeros((1, 9, 8, 8, 3), dtype=np.float32))
        result = _preview(source, self.root / "preview.png")
        self.assertEqual(len(result["frame_indices"]), 8)
        with Image.open(result["path"]) as preview:
            self.assertEqual(preview.size, (32, 16))

    def test_square_1024_preview_preserves_square_geometry_with_one_stride(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("optional CPU preview dependencies absent")
        source = self.root / "square.npy"
        np.save(source, np.zeros((1, 1024, 1024, 3), dtype=np.uint8))
        result = _preview(source, self.root / "square.png")
        self.assertEqual(result["sampling_stride"], 5)
        self.assertEqual(result["sample_frame_size"], [205, 205])
        self.assertEqual(result["frame_indices"], [0])
        self.assertEqual(result["sampled_rgb_bytes"], 205 * 205 * 3)
        with Image.open(result["path"]) as preview:
            self.assertEqual(preview.size, (205, 205))

    def test_landscape_832_480_preview_preserves_ratio_and_eight_frame_bound(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("optional CPU preview dependencies absent")
        source = self.root / "landscape.npy"
        np.save(source, np.zeros((1, 9, 480, 832, 3), dtype=np.uint8))
        result = _preview(source, self.root / "landscape.png")
        self.assertEqual(result["sampling_stride"], 3)
        self.assertEqual(result["sample_frame_size"], [278, 160])
        self.assertEqual(len(result["frame_indices"]), 8)
        self.assertEqual((result["frame_indices"][0], result["frame_indices"][-1]), (0, 8))
        self.assertEqual(result["sampled_rgb_bytes"], 8 * 278 * 160 * 3)
        self.assertLessEqual(result["sampled_rgb_bytes"], 4 * 1024 * 1024)
        # Each dimension differs from exact uniform scaling by less than a pixel.
        self.assertLess(278 - 832 / 3, 1)
        self.assertEqual(160, 480 / 3)
        with Image.open(result["path"]) as preview:
            self.assertEqual(preview.size, (4 * 278, 2 * 160))

    def test_video_pairs_use_actual_same_input_order_references_with_failure_priority(self):
        artifacts = []
        for name, failed in (("a", False), ("b", True)):
            for arm in ("eager", "compiled"):
                artifacts.append(dict(source=dict(path="source.json"), input_id=name, kind="measurement",
                    order=0, repeat=0, arm=arm, native=None, pointer="/rows/0/artifact",
                    path=name + "-" + arm + ".npy", failed=failed, shape=[1, 81, 480, 832, 3]))
        pair, missing = _video_pair(artifacts)
        self.assertEqual([row["input_id"] for row in pair], ["b", "b"])
        self.assertEqual([row["arm"] for row in pair], ["eager", "compiled"])
        self.assertEqual(missing, [])
        pair, missing = _video_pair([artifacts[0], artifacts[3]])
        self.assertEqual(pair, [])
        self.assertEqual(len(missing), 2)

    def test_video_pair_prefers_current_performance_over_historical_and_resume(self):
        old, new = self.historical_attempts()

        def artifact(path):
            return dict(path=str(path), shape=[1, 81, 480, 832, 3], sha256="a" * 64)

        def resume(attempt):
            return dict(kind="same_backend_resume", input_id="item", arm="eager",
                numerical=dict(passed=True), result=dict(artifacts=[
                    artifact(attempt / "a-continuous.npy"), artifact(attempt / "a-resumed.npy")]))

        old_rows = [resume(old)]
        current_rows = [resume(new)] + [dict(kind="performance", input_id="item", arm=arm,
            order=0, repeat=0, quality=dict(passed=True), artifact=artifact(new / ("z-" + arm + ".npy")))
            for arm in ("eager", "candidate")]
        for attempt, rows in ((old, old_rows), (new, current_rows)):
            atomic_json(attempt / "wan-compile.json", dict(schema_version="ditserve.optimizations.wan.v1",
                status="passed", rows=rows))
        artifacts = collect(self.output, self.ledger)["phases"]["wan_compile"]["artifacts"]
        historical = [row for row in artifacts if row.get("historical")]
        self.assertTrue(historical)
        self.assertTrue(all(row["failed"] and not row["quality_failed"] for row in historical))
        for candidates in (artifacts, list(reversed(artifacts))):
            pair, missing = _video_pair(candidates)
            self.assertEqual([(row["kind"], row["arm"]) for row in pair],
                             [("performance", "eager"), ("performance", "candidate")])
            self.assertTrue(all(not row.get("historical") for row in pair))
            self.assertEqual(missing, [])
        # Old report JSON has no distinct quality flag. A stopped historical
        # attempt must still not outrank the current actual candidate comparison.
        legacy = [{key: value for key, value in row.items() if key != "quality_failed"} for row in artifacts]
        self.assertEqual(_video_pair(legacy)[0][0]["kind"], "performance")
        # An explicit historical quality failure does outrank successful pairs.
        historical[1]["quality_failed"] = True
        self.assertTrue(all(row.get("historical") for row in _video_pair(artifacts)[0]))

    def test_video_pair_prefers_cache_qualification_over_native_resume_and_screen(self):
        base = dict(source=dict(path="cache.json", sha256="a" * 64), input_id="item",
                    shape=[1, 81, 480, 832, 3], failed=False, historical=False)
        artifacts = []
        for index, native in enumerate((True, False)):
            artifacts.append(dict(base, kind="original_split_equivalence", native=native,
                arm=None, pointer="/rows/%d/artifact" % index, path="a-native-%d.npy" % index))
        for index in range(2):
            artifacts.append(dict(base, kind="same_backend_resume", native=None, arm=None,
                pointer="/rows/2/result/artifacts/%d" % index, path="b-resume-%d.npy" % index))
        for kind, prefix in (("cache_screen", "c"), ("cache_qualification", "z")):
            for arm in ("off", "paired"):
                artifacts.append(dict(base, kind=kind, native=None, arm=arm, split="heldout",
                    order=0, pointer="/rows/3/artifact", path=prefix + "-" + arm + ".npy"))
        pair, missing = _video_pair(artifacts)
        self.assertEqual([(row["kind"], row["arm"]) for row in pair],
                         [("cache_qualification", "off"), ("cache_qualification", "paired")])
        self.assertEqual(missing, [])

    def cache_pair_rows(self, attempt, *, name="item", split="dev", order=0, passed=False):
        rows = [dict(kind="cache_qualification", split=split, order=order, input_id=name, arm=arm,
                     complete=True, artifact=dict(path=str(attempt / ("%s-%s-%d-%s.npy" %
                         (name, split, order, arm))), shape=[1, 81, 480, 832, 3], sha256="a"*64))
                for arm in ("off", "paired")]
        rows.append(dict(kind="cache_qualification_pair", split=split, order=order, input_id=name,
                         complete=True, quality=dict(passed=passed, metrics=dict(lpips_mean=.001 if passed else .06))))
        return rows

    def cache_evidence(self, attempt, rows, name="cache-k4.json"):
        path = attempt / name
        atomic_json(path, dict(schema_version="ditserve.optimizations.wan.v1", status="passed", rows=rows))
        return path

    def test_cache_pair_quality_binds_only_candidate_and_selects_later_failure_video(self):
        attempt = self.phase("cache_k4")
        rows = self.cache_pair_rows(attempt, name="a-pass", passed=True)
        rows += self.cache_pair_rows(attempt, name="z-fail")
        path = self.cache_evidence(attempt, rows)
        value = collect(self.output, self.ledger)["phases"]["cache_k4"]
        artifacts = value["artifacts"]
        pair, missing = _video_pair(artifacts)
        self.assertEqual([item["input_id"] for item in pair], ["z-fail", "z-fail"])
        self.assertEqual([item["arm"] for item in pair], ["off", "paired"])
        self.assertFalse(pair[0]["failed"])
        self.assertNotIn("pair_quality", pair[0])
        self.assertTrue(pair[1]["failed"])
        binding = pair[1]["pair_quality"]
        self.assertEqual(binding["source"]["path"], str(path))
        self.assertEqual(binding["source"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(binding["pointer"], "/rows/5/quality")
        self.assertEqual(binding["reference_artifact"]["path"], pair[0]["path"])
        self.assertEqual(binding["evidence"]["metrics"]["lpips_mean"], .06)
        self.assertEqual(binding["split"], "dev")
        self.assertEqual(missing, [])

    def test_cache_pair_quality_and_video_pairing_keep_split_order_and_source(self):
        attempt = self.phase("cache_k4")
        rows = []
        for split in ("dev", "heldout"):
            for order in (0, 1):
                rows += self.cache_pair_rows(attempt, split=split, order=order,
                    passed=(split, order) != ("heldout", 1))
        self.cache_evidence(attempt, rows)
        artifacts = collect(self.output, self.ledger)["phases"]["cache_k4"]["artifacts"]
        pair, missing = _video_pair(artifacts)
        self.assertEqual([(x["split"], x["order"]) for x in pair], [("heldout",1)]*2)
        self.assertEqual(sum(x["failed"] for x in artifacts), 1)
        crossed = [next(x for x in artifacts if x["split"] == "dev" and x["order"] == 0 and x["arm"] == "off"),
                   next(x for x in artifacts if x["split"] == "heldout" and x["order"] == 0 and x["arm"] == "paired")]
        self.assertEqual(_video_pair(crossed)[0], [])
        other_hash = dict(crossed[0], source=dict(crossed[0]["source"], sha256="other"), arm="paired")
        self.assertEqual(_video_pair([crossed[0], other_hash])[0], [])

    def test_cache_pair_quality_missing_or_duplicate_matches_stay_unknown(self):
        attempt = self.phase("cache_k4")
        for case in ("missing_reference", "duplicate_reference", "duplicate_candidate", "duplicate_pair",
                     "missing_split", "incomplete_pair", "unknown_quality", "other_source"):
            with self.subTest(case=case):
                rows = self.cache_pair_rows(attempt)
                if case == "missing_reference": rows.pop(0)
                elif case == "duplicate_reference": rows.insert(0, dict(rows[0]))
                elif case == "duplicate_candidate": rows.insert(0, dict(rows[1]))
                elif case == "duplicate_pair": rows.append(dict(rows[-1]))
                elif case == "missing_split": rows[-1].pop("split")
                elif case == "incomplete_pair": rows[-1]["complete"] = False
                elif case == "unknown_quality": rows[-1]["quality"].pop("passed")
                elif case == "other_source":
                    self.cache_evidence(attempt, [rows.pop()], "cache-other.json")
                self.cache_evidence(attempt, rows)
                artifacts = collect(self.output, self.ledger)["phases"]["cache_k4"]["artifacts"]
                self.assertTrue(all("pair_quality" not in item for item in artifacts))
                self.assertTrue(all(not item["failed"] for item in artifacts))
                extra = attempt / "cache-other.json"
                if extra.exists(): extra.unlink()

    def test_paired_mp4_export_is_capped_and_preserves_raw(self):
        self.all_terminal()
        attempt = self.output / "vae" / "attempt00"
        rows = []
        for name in ("a", "b"):
            for arm in ("eager", "compiled"):
                path = attempt / (name + "-" + arm + ".npy")
                path.write_bytes((name + arm).encode())
                rows.append(dict(kind="measurement", input_id=name, round=0, repeat=0, arm=arm,
                    quality=dict(passed=name != "b"), artifact=dict(path=str(path), shape=[1, 81, 480, 832, 3],
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        atomic_json(attempt / "vae.json", dict(schema_version="ditserve.optimizations.stages.v1", status="stopped", rows=rows))
        def render(path, output):
            output.write_bytes(b"bounded fake media")
            return dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), fps=16, frame_count=81)
        with patch("ditserve.optimizations.reporting._preview", side_effect=render), \
             patch("ditserve.optimizations.reporting._write_video", side_effect=render) as video:
            result = archive(self.output, self.ledger)
        self.assertEqual(video.call_count, 2)
        self.assertEqual(result["video_count"], 2)
        self.assertEqual(result["complete_video_pairs"], 1)
        self.assertEqual(result["preview_count"], 4)
        self.assertTrue(all("b-" in call.args[0].name for call in video.call_args_list))
        self.assertTrue(all(Path(row["artifact"]["path"]).exists() for row in rows))

    def test_streamed_video_writes_each_of_81_frames_and_checks_16fps_without_resampling(self):
        from ditserve.gpu import media
        selected, appended = [], []
        destination = self.root / "tiny.mp4"
        class Array:
            ndim, shape = 5, (1, 81, 480, 832, 3)
            def __getitem__(self, key):
                selected.append(key[1].start)
                return key
        class Writer:
            def __enter__(self): return self
            def __exit__(self, *args): destination.write_bytes(b"fake mp4")
            def append_data(self, frame): appended.append(frame)
        class Reader:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def get_meta_data(self): return dict(fps=16., size=(832, 480))
            def count_frames(self): return 81
        numpy = types.ModuleType("numpy")
        numpy.load = lambda *args, **kwargs: Array()
        imageio, v2 = types.ModuleType("imageio"), types.ModuleType("imageio.v2")
        imageio.v2 = v2
        writer_options = []
        def get_writer(path, **kwargs):
            writer_options.append(kwargs)
            return Writer()
        v2.get_writer, v2.get_reader = get_writer, lambda path: Reader()
        with patch.dict("sys.modules", {"numpy": numpy, "imageio": imageio, "imageio.v2": v2}), \
             patch.object(media, "video_frames", return_value=["frame"]):
            result = _write_video(self.root / "source.npy", destination)
        self.assertEqual(selected, list(range(81)))
        self.assertEqual(len(appended), 81)
        self.assertEqual(writer_options[0]["fps"], 16)
        self.assertEqual(result["frame_count"], 81)
        self.assertEqual(result["fps"], 16.)

    def test_factorial_online_qualification_is_not_misclassified_as_local(self):
        attempt = self.phase("cache_k4")
        path = attempt / "cache-online.json"
        atomic_json(path, dict(schema_version="ditserve.optimizations.cache_factorial.v1", status="completed",
            comparisons=dict(cache_on_k4=dict(passed=True)),
            rows=[dict(status="completed", complete=True, factorial_arm="paired_k4", order_index=0,
                       requests=[dict(request_id="r000", status="completed", arrived=True)], events=[])]))
        terminal_path = self.output / "cache_k4" / "terminal.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["online_path"] = str(path)
        atomic_json(terminal_path, terminal)
        report = collect(self.output, self.ledger)["phases"]["cache_k4"]
        self.assertEqual(report["local_samples"], [])
        self.assertTrue(report["online_qualified"])
        self.assertTrue(report["factorial_comparisons"]["cache_on_k4"]["passed"])

    def historical_attempts(self):
        self.all_terminal()
        old = self.output / "wan_compile/attempt00"
        failed = dict(status="failed", output=str(old), reason="KeyboardInterrupt", returncode=1,
            charge=dict(gpu_seconds=286.20375, session_id="stopped-attempt"),
            local_qualified=True, online_qualified=True,
            final_occupancy=dict(status="exclusive_observed"))
        atomic_json(old / "supervisor.json", failed)
        atomic_json(old / "baseline-audit-stop.json", dict(reason="native baseline wrapper overhead invalidates comparison",
                                                         action="stop owned process"))
        atomic_json(old / "wan-compile.json", dict(schema_version="ditserve.optimizations.wan.v1", status="failed",
            rows=[dict(kind="measurement", arm="eager", service=dict(ms=20.))],
            active_unit=dict(kind="same_backend_resume", input_id="p0", started=True)))
        new = self.output / "wan_compile/attempt01"
        new.mkdir()
        passed = dict(status="passed", output=str(new), returncode=0,
            charge=dict(gpu_seconds=50., session_id="new-attempt"), local_qualified=True, online_qualified=False,
            final_occupancy=dict(status="exclusive_observed"))
        atomic_json(new / "supervisor.json", passed)
        atomic_json(self.output / "wan_compile/terminal.json", passed)
        atomic_json(new / "wan-compile.json", dict(schema_version="ditserve.optimizations.wan.v1", status="passed",
            rows=[dict(kind="measurement", arm="eager", service=dict(ms=5.))]))
        return old, new

    def test_old_stopped_attempt_samples_audit_and_charge_survive_latest_report_and_excel(self):
        old, new = self.historical_attempts()
        original = (old / "supervisor.json").read_bytes()
        value = collect(self.output, self.ledger)["phases"]["wan_compile"]
        self.assertEqual(value["attempt_path"], str(new))
        self.assertEqual(value["local"][0]["measured_groups"][0]["service_ms"]["mean"], 5.)
        self.assertEqual(len(value["previous_attempts"]), 1)
        previous = value["previous_attempts"][0]
        self.assertFalse(previous["qualification_eligible"])
        self.assertFalse(previous["local_qualified"])
        self.assertFalse(previous["online_qualified"])
        self.assertEqual(previous["charge"]["gpu_seconds"], 286.20375)
        self.assertEqual(previous["audit_notes"][0]["source"]["sha256"],
                         hashlib.sha256((old / "baseline-audit-stop.json").read_bytes()).hexdigest())
        self.assertEqual(len(value["previous_samples"]), 2)
        self.assertTrue(all(row["historical"] and not row["qualification_eligible"] for row in value["previous_samples"]))
        write_report(self.output, self.ledger)
        markdown = (self.output / "REPORT.md").read_text()
        self.assertIn("attempt00", markdown)
        self.assertIn("286.204", markdown)
        self.assertIn("wrapper overhead", markdown)
        archive(self.output, self.ledger, preview_limit=0)
        with gzip.open(self.output / "excel/data.json.gz", "rt") as stream:
            payload = stream.read()
        self.assertIn("stopped-attempt", payload)
        self.assertIn("native baseline wrapper overhead", payload)
        self.assertIn("previous_samples.jsonl", payload)
        self.assertEqual((old / "supervisor.json").read_bytes(), original)

    def test_historical_compile_counter_audit_preserves_error_and_source_bound_interpretation(self):
        old, new = self.historical_attempts()
        supervisor = json.loads((old / "supervisor.json").read_text())
        error = "RuntimeError: compiled Wan region silently executed without a compiled graph"
        supervisor["reason"] = error
        atomic_json(old / "supervisor.json", supervisor)
        audit = old / "compile-monitor-audit.json"
        interpretation = ("Dynamo unwrapping bypassed the monitoring counter; "
                          "this is not evidence of a confirmed eager fallback.")
        atomic_json(audit, dict(failure_interpretation=interpretation,
                               status="confirmed_implementation_observation_defect"))
        originals = {path: path.read_bytes() for path in (audit, old / "supervisor.json", self.ledger)}
        report = write_report(self.output, self.ledger)
        markdown = (self.output / "REPORT.md").read_text()
        self.assertIn(error, markdown)
        self.assertIn(interpretation, markdown)
        self.assertIn("[compile-monitor-audit.json](%s)" % audit, markdown)
        self.assertIn(hashlib.sha256(originals[audit]).hexdigest(), markdown)
        prior = report["phases"]["wan_compile"]["previous_attempts"][0]
        self.assertFalse(prior["local_qualified"])
        self.assertFalse(prior["online_qualified"])
        self.assertEqual(prior["reason"], error)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_active_retry_selects_current_attempt_and_moves_old_terminal_into_history(self):
        old, new = self.historical_attempts()
        (new / "supervisor.json").unlink()
        atomic_json(self.output / "wan_compile/terminal.json", json.loads((old / "supervisor.json").read_text()))
        budget = json.loads(self.ledger.read_text())
        budget["active"] = dict(started_monotonic=0., gpu_count=1, identity=dict(phase="wan_compile", output=str(new)))
        atomic_json(self.ledger, budget)
        value = collect(self.output, self.ledger)
        phase = value["phases"]["wan_compile"]
        self.assertEqual(phase["attempt_path"], str(new))
        self.assertEqual(phase["status"], "running")
        self.assertFalse(phase["terminal_present"])
        self.assertFalse(phase["local_qualified"])
        self.assertEqual(phase["previous_attempts"][0]["status"], "failed")
        self.assertFalse(value["archive_ready"])

    def test_audit_stop_invalidates_stale_success_fields_in_same_attempt(self):
        attempt = self.phase()
        self.local(attempt, [])
        atomic_json(attempt / "baseline-audit-stop.json", dict(reason="invalid baseline"))
        value = collect(self.output, self.ledger)["phases"]["sana_compile"]
        self.assertFalse(value["supervisor_valid"])
        self.assertFalse(value["local_qualified"])
        self.assertFalse(value["online_qualified"])

    def test_online_only_audit_preserves_local_gain_quality_and_measured_metrics(self):
        attempt = self.phase()
        self.local(attempt, [dict(kind="measurement", backend="compiled", service=dict(ms=12.),
            quality=dict(passed=True, metrics=dict(lpips_mean=.003)))])
        atomic_json(attempt / "online.json", dict(status="completed", rows=[self.trace()],
            qualification=dict(passed=True, failures=[], pooled=dict(goodput_gain=.2))))
        audit = attempt / "online-measurement-audit.json"
        atomic_json(audit, dict(invalidates_online_qualification=True,
            reason="cumulative persistence changed measured online arrival timing"))
        report = write_report(self.output, self.ledger)
        value = report["phases"]["sana_compile"]
        self.assertTrue(value["supervisor_valid"])
        self.assertTrue(value["local_qualified"])
        self.assertEqual(value["local"][0]["measured_groups"][0]["service_ms"]["mean"], 12.)
        self.assertTrue(value["local_samples"][0]["evidence"]["quality"]["passed"])
        self.assertFalse(value["online_qualified"])
        self.assertFalse(value["online_evaluation_complete"])
        self.assertFalse(value["online"][0]["authoritative"])
        self.assertTrue(value["online"][0]["complete"])
        self.assertEqual(value["online_qualification"]["pooled"]["goodput_gain"], .2)
        self.assertEqual(value["decision"], "未验证")
        self.assertIn("cumulative persistence", (self.output / "REPORT.md").read_text())
        self.assertEqual(value["audit_notes"][0]["source"]["sha256"], hashlib.sha256(audit.read_bytes()).hexdigest())

    def test_online_only_audit_survives_history_without_mixing_local_means(self):
        old, new = self.historical_attempts()
        (old / "baseline-audit-stop.json").unlink()
        atomic_json(old / "online-measurement-audit.json", dict(invalidates_online_qualification=True,
            reason="old online timing invalid; old local evidence remains descriptive"))
        phase = collect(self.output, self.ledger)["phases"]["wan_compile"]
        self.assertEqual(phase["local"][0]["measured_groups"][0]["service_ms"]["mean"], 5.)
        prior = phase["previous_attempts"][0]
        self.assertTrue(prior["online_audit_invalidated"])
        self.assertEqual(prior["local"][0]["measured_groups"][0]["service_ms"]["mean"], 20.)
        self.assertEqual(prior["local"][0]["source"]["path"], str(old / "wan-compile.json"))
        self.assertFalse(prior["qualification_eligible"])
        self.assertTrue(all(row["source"]["path"].startswith(str(old)) for row in phase["previous_samples"]))

    def test_replayed_local_rows_and_compilation_keep_original_execution_origin(self):
        old, new = self.historical_attempts()
        original = old / "wan-compile.json"
        origin = dict(original_local_path=str(original), original_local_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            remeasured=False, binding=dict(source_attempt=str(old), session_id="stopped-attempt"))
        rows = [dict(kind="same_backend_resume", input_id="r%d" % i, numerical=dict(passed=True)) for i in range(8)]
        rows += [dict(kind="performance", backend="compiled", input_id="p%d" % i, service=dict(ms=10.)) for i in range(16)]
        atomic_json(new / "wan-compile.json", dict(schema_version="ditserve.optimizations.wan.v1", status="passed",
            rows=rows, local_evidence_reuse=origin,
            compilation={"compiled": dict(compile_backend_ms=7870., compilations=1)}))
        report = write_report(self.output, self.ledger)
        phase = report["phases"]["wan_compile"]
        local = phase["local"][0]
        self.assertEqual(phase["sample_counts"]["local_rows"], 24)
        self.assertEqual(phase["sample_counts"]["reused_local_rows"], 24)
        self.assertEqual(local["new_local_execution_rows"], 0)
        self.assertEqual(local["reused_local_rows"], 24)
        self.assertEqual(local["measured_groups"][0]["service_ms"]["n"], 16)
        self.assertTrue(all(row["measurement_origin"]["mode"] == "reused_not_remeasured" for row in phase["local_samples"]))
        self.assertTrue(all(not row["measurement_origin"]["executed_in_current_attempt"] for row in phase["local_samples"]))
        self.assertEqual(local["compilation_evidence"]["values"]["compiled"]["compile_backend_ms"], 7870.)
        self.assertFalse(local["compilation_evidence"]["current_process_measurement"])
        self.assertEqual(local["compilation_evidence"]["measurement_origin"]["original_source"]["path"], str(original))
        self.assertIsNone(local["current_process_cost"]["compilation_ms"])
        self.assertFalse(local["current_process_cost"]["separate_components_measured"])
        self.assertEqual(phase["charge"]["gpu_seconds"], 50.)
        self.assertEqual(report["budget"]["used_gpu_seconds"], 120.)
        markdown = (self.output / "REPORT.md").read_text()
        self.assertIn("新增局部执行 0 行", markdown)
        self.assertIn("不是本进程编译耗时", markdown)
        self.assertIn("不能合并增加样本量", markdown)
        self.assertIn("原始尝试 attempt00", markdown)
        archive(self.output, self.ledger, preview_limit=0)
        with gzip.open(self.output / "excel/data.json.gz", "rt") as stream:
            payload = stream.read()
        self.assertIn("reused_not_remeasured", payload)
        self.assertIn("current_process_measurement", payload)

    def test_live_replay_never_gains_qualification_or_reports_new_cold_compile(self):
        old, new = self.historical_attempts()
        (new / "supervisor.json").unlink()
        value = json.loads((new / "wan-compile.json").read_text())
        value["local_evidence_reuse"] = dict(remeasured=False, original_local_path=str(old / "wan-compile.json"),
            original_local_sha256="fixture-original", binding=dict(source_attempt=str(old)))
        atomic_json(new / "wan-compile.json", value)
        budget = json.loads(self.ledger.read_text())
        budget["active"] = dict(started_monotonic=0., gpu_count=1, identity=dict(phase="wan_compile", output=str(new)))
        atomic_json(self.ledger, budget)
        with patch("ditserve.optimizations.reporting.time.monotonic", return_value=12.):
            report = collect(self.output, self.ledger)
        phase = report["phases"]["wan_compile"]
        self.assertFalse(phase["local_qualified"])
        self.assertFalse(phase["online_qualified"])
        self.assertEqual(phase["status"], "running")
        self.assertEqual(report["active_gpu_seconds_estimate"], 12.)
        self.assertEqual(phase["local"][0]["new_local_execution_rows"], 0)
        self.assertIsNone(phase["local"][0]["current_process_cost"]["compilation_ms"])

    def test_setup_windows_are_mixed_host_scopes_and_not_compile_only(self):
        flushes = [dict(kind="cumulative_record", status="completed", finished_perf_counter_s=1.),
            dict(kind="prepare_partial", status="completed", started_perf_counter_s=1.003),
            dict(kind="terminal_partial", status="completed", started_perf_counter_s=4.),
            dict(kind="cumulative_record", status="completed", finished_perf_counter_s=5.),
            dict(kind="prepare_partial", status="completed", started_perf_counter_s=5.007),
            dict(kind="prepare_partial", status="completed", started_perf_counter_s=6.),
            dict(kind="cumulative_record", status="failed", finished_perf_counter_s=7.),
            dict(kind="prepare_partial", status="completed", started_perf_counter_s=8.),
            dict(kind="cumulative_record", status="completed", finished_perf_counter_s=10.),
            dict(kind="prepare_partial", status="completed", started_perf_counter_s=9.)]
        windows = _setup_windows(flushes)
        self.assertEqual(windows["host_ms"]["n"], 2)
        self.assertAlmostEqual(windows["host_ms"]["mean"], 5.)
        self.assertEqual(windows["windows"][1]["end_pointer"], "/flushes/4/started_perf_counter_s")
        self.assertIn("not compile-only", windows["scope"])
        self.assertIn("excludes model loading", windows["scope"])
        self.assertIn("already within allocated cost", windows["scope"])


if __name__ == "__main__":
    unittest.main()
