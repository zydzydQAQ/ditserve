"""Counter units and asynchronous CUDA projection must survive trace aggregation."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ditserve.evaluation.workflow_trace import (
    MetricSeries, analyze_trace, metric_definition, normalized_metric, parse_label,
)


class WorkflowCounterTests(unittest.TestCase):
    def test_percentages_are_not_rescaled_and_old_clock_overflow_is_corrected(self):
        self.assertEqual(normalized_metric(87, metric_definition("SM Active [Throughput %]")), 87)
        self.assertEqual(metric_definition("SMs Active [Throughput %]"), metric_definition("SM Active [Throughput %]"))
        self.assertEqual(normalized_metric(-1794967296, metric_definition("GPC Clock Frequency [MHz]")), 2500)
        self.assertIsNone(metric_definition("Compute Warps in Flight [Avg]"))

    def test_counter_sample_represents_preceding_interval_and_clips_window(self):
        summary = MetricSeries([(0, 0), (10, 20), (30, 40)]).summary(5, 25)
        self.assertEqual(summary["mean"], 35)
        self.assertEqual(summary["covered_ns"], 20)
        self.assertEqual(summary["p50"], 40)
        self.assertEqual(summary["p05"], 20)

    def test_sampling_gaps_and_unobserved_boundaries_are_not_filled(self):
        summary = MetricSeries([(0, 0), (10, 20), (20, 30), (200, 90), (210, 40)]).summary(-10, 220)
        self.assertEqual(summary["covered_ns"], 30)
        self.assertEqual(summary["mean"], 30)
        self.assertEqual(summary["nominal_period_ns"], 10)

    def test_filters_preserve_real_input_stage_and_separate_arms(self):
        self.assertFalse(parse_label('ditserve.workflow/text_encoder|{"purpose":"numerical_validation","variant":"T2"}')["included"])
        self.assertFalse(parse_label("ditserve.workflow/E6/baseline/r-1/dit.step.0|{}")["included"])
        self.assertTrue(parse_label('ditserve.workflow/wan_prepare_input_h2d_latent|{"purpose":"video_measurement"}')["included"])
        self.assertEqual(parse_label("E2/preencodedTrue/r2/dual/request.0/dit.step.1")["arm"], "E2/preencodedTrue/r2/dual")
        self.assertEqual(parse_label("E5/dit/r0/streams/dit.step.1")["arm"], "E5/dit/r0/streams")
        self.assertEqual(parse_label("E6/heads_pipeline/r2/dit.step.1")["arm"], "E6/heads_pipeline/r2")

    def test_encoder_and_gpu_ranges_recover_common_request_frames(self):
        metadata = dict(variant="T0", purpose="video_measurement", repeat=0, request_id="T0-f17-r0")
        encoder = parse_label("ditserve.workflow/text_encoder|" + json.dumps(metadata))
        gpu = parse_label("ditserve.workflow/dit_step|" + json.dumps(dict(metadata, frames=17, step=1)))
        self.assertEqual(encoder["arm"], gpu["arm"])
        self.assertEqual(encoder["metadata"]["frames"], 17)
        warmup = parse_label('ditserve.workflow/text_encoder|{"variant":"T0",'
                            '"purpose":"video_measurement","request_id":"T0-f33-r-1"}')
        self.assertFalse(warmup["included"])

    def test_dedicated_queue_modes_and_frame_counts_are_separate(self):
        arms = []
        for mode in ("serial", "prefetch"):
            for frames in (17, 33):
                metadata = dict(variant="T5", purpose="dedicated_queue", repeat=0,
                                request_id=f"T5-q-{mode}-f{frames}-r0-i1")
                encoder = parse_label("ditserve.workflow/text_encoder|" + json.dumps(metadata))
                gpu = parse_label("ditserve.workflow/dit_step|" + json.dumps(dict(metadata, mode=mode, frames=frames)))
                self.assertEqual(encoder["arm"], gpu["arm"])
                self.assertEqual(encoder["metadata"]["mode"], mode)
                arms.append(encoder["arm"])
        self.assertEqual(len(set(arms)), 4)

    def test_p4_peer_frame_counts_remain_distinct(self):
        arms = []
        for peer_frames in (17, 33):
            metadata = dict(variant="P4", purpose="timed", mode="streams", repeat=0,
                            frames=17, peer_frames=peer_frames, role="A")
            arms.append(parse_label("ditserve.workflow/P4/dit_step|" + json.dumps(metadata))["arm"])
        self.assertEqual(arms, ["P4/timed/streams/frames.17/peer_frames.17/r0",
                                "P4/timed/streams/frames.17/peer_frames.33/r0"])


class WorkflowProjectionTests(unittest.TestCase):
    def test_p4_intersection_excludes_ancestor_prefix_wrong_role_and_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,copyKind INTEGER,bytes INTEGER);
            ''')
            pid, tid = 17 << 24, (17 << 24) + 42
            connection.execute("INSERT INTO StringIds VALUES (1,'model_kernel')")
            correlation = 0
            for repeat, mode in enumerate(("serial", "streams")):
                offset = repeat * 1000
                metadata = dict(variant="P4", purpose="timed", mode=mode, repeat=0, frames=17, peer_frames=17)
                parent = "ditserve.workflow/P4/stage_pair|" + json.dumps(dict(metadata, role="A"))
                connection.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                                   (offset, offset + 100, parent, tid))
                # Serial A [100,120], B [120,140]: zero. Streams A kernels
                # [100,120] and [110,125] union to 25; B [115,135]: 10 ns overlap.
                windows = [("dit_step", "A", 10, 100, 120, "kernel"),
                           ("vae_decode", "B", 20, 120 if mode == "serial" else 115,
                            140 if mode == "serial" else 135, "kernel"),
                           ("prefix_b_dit_step", "A", 30, 90, 150, "kernel"),
                           ("dit_step", "B", 40, 90, 150, "kernel"),
                           ("vae_decode", "A", 50, 90, 150, "kernel"),
                           ("dit_step", "A", 60, 90, 150, "copy")]
                if mode == "streams":
                    windows.append(("dit_step", "A", 70, 110, 125, "kernel"))
                for leaf, role, host, start, end, kind in windows:
                    correlation += 1
                    label = "ditserve.workflow/P4/" + leaf + "|" + json.dumps(dict(metadata, role=role))
                    connection.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                                       (offset + host, offset + host + 3, label, tid))
                    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                                       (offset + host + 1, offset + host + 2, tid, correlation))
                    if kind == "kernel":
                        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,1)",
                                           (offset + start, offset + end, 1, pid, correlation, 1 if role == "A" else 2))
                    else:
                        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,2,1024)",
                                           (offset + start, offset + end, 1, pid, correlation, 2))
            connection.commit()
            connection.close()
            arms = {arm["name"]: arm for arm in analyze_trace(path)["arms"]}
            serial = arms["P4/timed/serial/frames.17/peer_frames.17/r0"]["devices"]["1"]
            streams = arms["P4/timed/streams/frames.17/peer_frames.17/r0"]["devices"]["1"]
            self.assertTrue(serial["dit_vae_kernel_overlap_available"])
            self.assertEqual(serial["dit_vae_kernel_overlap_ms"], 0)
            self.assertEqual(streams["dit_vae_kernel_overlap_ms"], 10e-6)
            self.assertEqual(streams["dit_a_kernel_union_ms"], 25e-6)
            self.assertEqual(streams["vae_b_kernel_union_ms"], 20e-6)

    def test_p4_missing_role_is_unavailable_and_devices_are_not_crossed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
            ''')
            pid, tid = 17 << 24, (17 << 24) + 42
            connection.execute("INSERT INTO StringIds VALUES (1,'model_kernel')")
            for index, (leaf, role, device) in enumerate((("dit_step", "A", 0), ("vae_decode", "B", 1)), 1):
                metadata = dict(variant="P4", purpose="timed", mode="streams", repeat=0,
                                frames=17, peer_frames=17, role=role)
                label = "ditserve.workflow/P4/" + leaf + "|" + json.dumps(metadata)
                connection.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                                   (index*10, index*10+3, label, tid))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                                   (index*10+1, index*10+2, tid, index))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,1)",
                                   (100, 200, device, pid, index, 1))
            connection.commit()
            connection.close()
            devices = analyze_trace(path)["arms"][0]["devices"]
            for value in devices.values():
                self.assertFalse(value["dit_vae_kernel_overlap_available"])
                self.assertIsNone(value["dit_vae_kernel_overlap_ms"])

    def test_vae_arm_prefix_does_not_attribute_sana_kernels_to_wan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
            ''')
            pid, tid = 17 << 24, (17 << 24) + 42
            connection.execute("INSERT INTO StringIds VALUES (1,'model_kernel')")
            # Every arm includes /vae/ in its prefix. Only actual VAE operation
            # leaves are Wan, including the interleaved chunked decode leaves.
            launches = [("sana_only", "sana.0", 0, 10, 20, 1),
                        ("interleaved", "vae.chunk.0", 30, 40, 50, 1),
                        ("interleaved", "sana.0", 35, 50, 60, 1),
                        ("streams", "vae", 70, 80, 100, 1),
                        ("streams", "sana.0", 75, 90, 110, 2)]
            for index, (mode, leaf, host, start, end, stream) in enumerate(launches, 1):
                label = f"ditserve.workflow/E5/vae/r0/{mode}/{leaf}|{{}}"
                connection.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                                   (host, host + 3, label, tid))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                                   (host + 1, host + 2, tid, index))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,1)",
                                   (start, end, 0, pid, index, stream))
            connection.commit()
            connection.close()
            arms = {arm["name"]: arm for arm in analyze_trace(path)["arms"]}
            for mode, expected in (("sana_only", 0), ("interleaved", 0), ("streams", 10e-6)):
                with self.subTest(mode=mode):
                    actual = arms[f"E5/vae/r0/{mode}"]["devices"]["0"]
                    self.assertEqual(actual["wan_sana_kernel_overlap_ms"], expected)

    def test_dedicated_encoder_overlap_joins_devices_without_merging_schedules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
            ''')
            pid, main_tid, encoder_tid = 17 << 24, (17 << 24) + 42, (17 << 24) + 43
            connection.execute("INSERT INTO StringIds VALUES (1,'model_kernel')")
            windows = [("serial", "text_encoder", 0, 10, 0, encoder_tid),
                       ("serial", "dit_step", 10, 20, 1, main_tid),
                       ("prefetch", "text_encoder", 30, 40, 0, encoder_tid),
                       ("prefetch", "dit_step", 35, 45, 1, main_tid)]
            for index, (mode, phase, start, end, device, tid) in enumerate(windows, 1):
                metadata = dict(variant="T5", purpose="dedicated_queue", repeat=0,
                                request_id=f"T5-q-{mode}-f17-r0-i0")
                if phase == "dit_step":
                    metadata.update(mode=mode, frames=17, step=0)
                label = "ditserve.workflow/" + phase + "|" + json.dumps(metadata)
                connection.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)", (start, end, label, tid))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)", (start, start + 1, tid, index))
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,1)",
                                   (start, end, device, pid, index, 1))
            connection.commit()
            connection.close()
            report = analyze_trace(path)
            arms = {arm["name"]: arm for arm in report["arms"]}
            self.assertEqual(len(arms), 2)
            serial = arms["T5/dedicated_queue/serial/frames.17/r0"]
            prefetch = arms["T5/dedicated_queue/prefetch/frames.17/r0"]
            self.assertEqual(serial["dual_gpu_overlap"][0]["simultaneous_compute_ms"], 0)
            self.assertEqual(prefetch["dual_gpu_overlap"][0]["simultaneous_compute_ms"], 5e-6)
            self.assertEqual(len(serial["phase_ids"]), 2)
            self.assertEqual(len(prefetch["phase_ids"]), 2)

    def test_projection_follows_launch_correlation_beyond_cpu_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,copyKind INTEGER,bytes INTEGER);
                CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INTEGER,metricId INTEGER,metricName TEXT,sourceId INTEGER,typeName TEXT);
                CREATE TABLE GPU_METRICS(timestamp INTEGER,typeId INTEGER,metricId INTEGER,value INTEGER);
            ''')
            pid, tid = 17 << 24, (17 << 24) + 42
            connection.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, "attention_kernel"), (2, "ncclDevKernel_AllToAll")])
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (0,10,?,?,NULL)", ("ditserve.workflow/E3/r0/async/dit.step.0|{}", tid))
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)", [(1, 2, tid, 1), (3, 4, tid, 2), (5, 6, tid, 3)])
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)", [(8, 20, 0, pid, 1, 1, 1), (10, 14, 0, pid, 2, 2, 2),
                                                                                                    (100, 200, 0, 99 << 24, 1, 1, 1)])
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (12,16,0,?,3,3,1,1024)", (pid,))
            connection.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (256,0,'SM Active [Throughput %]',0,'GPU 0')")
            connection.executemany("INSERT INTO GPU_METRICS VALUES (?,256,0,?)", [(0, 0), (10, 80), (20, 60)])
            connection.commit()
            connection.close()
            report = analyze_trace(path)
            projection = report["phases"][0]["devices"]["0"]["gpu_projection"]
            self.assertEqual(projection["start_ns"], 8)
            self.assertEqual(projection["end_ns"], 20)
            self.assertEqual(projection["operation_count"], 3)
            self.assertEqual(projection["communication"]["nccl"]["overlap_ns"], 4)
            self.assertEqual(projection["communication"]["h2d"]["overlap_ns"], 4)
            self.assertEqual(projection["communication"]["h2d"]["bytes"], 1024)
            self.assertEqual(report["counts"]["cuda_operations_correlated_to_workflow"], 3)
            self.assertEqual(report["counts"]["cuda_operations"], 4)
            arm = report["arms"][0]
            self.assertEqual(arm["name"], "E3/r0/async")
            self.assertEqual((arm["host_start_ns"], arm["host_end_ns"], arm["host_span_ms"]), (0, 10, 10e-6))
            # The foreign PID's 100..200 kernel must not extend this arm.
            self.assertEqual((arm["execution_start_ns"], arm["execution_end_ns"], arm["execution_span_ms"]), (0, 20, 20e-6))
            self.assertEqual(arm["devices"]["0"]["metrics"]["sm_active_percent"]["mean"], 80)
            self.assertEqual(arm["devices"]["0"]["execution_window_metrics"]["sm_active_percent"]["mean"], 70)
            self.assertEqual(report["status"], "completed")
            json.dumps(report, allow_nan=False)

    def test_async_arm_execution_envelope_includes_trailing_copy_and_cpu_only_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,copyKind INTEGER,bytes INTEGER);
            ''')
            pid, tid = 17 << 24, (17 << 24) + 42
            connection.execute("INSERT INTO StringIds VALUES (1,'model_kernel')")
            for start, end, name in [(0, 5, "E5/dit/r0/streams/dit.step.0"),
                                     (5, 10, "E5/dit/r0/streams/sana.0"),
                                     (50, 70, "E1/r0/serial/request.0/cpu_encode")]:
                connection.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
                                   (start, end, "ditserve.workflow/" + name + "|{}", tid))
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                                   [(1, 2, tid, 1), (6, 7, tid, 2), (8, 9, tid, 3)])
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,1)",
                                   [(3, 30, 0, pid, 1, 1), (8, 35, 0, pid, 2, 2)])
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (35,40,0,?,3,2,2,1024)", (pid,))
            connection.commit()
            connection.close()
            arms = {arm["name"]: arm for arm in analyze_trace(path)["arms"]}
            streams = arms["E5/dit/r0/streams"]
            self.assertEqual(streams["host_span_ms"], 10e-6)
            self.assertEqual(streams["execution_span_ms"], 40e-6)
            self.assertEqual(streams["execution_end_ns"], 40)
            self.assertEqual(streams["devices"]["0"]["compute_union_ms"], 32e-6)
            self.assertEqual(streams["devices"]["0"]["wan_sana_kernel_overlap_ms"], 22e-6)
            self.assertAlmostEqual(streams["devices"]["0"]["compute_union_ms"] / streams["execution_span_ms"], .8)
            cpu = arms["E1/r0/serial"]
            self.assertEqual((cpu["execution_start_ns"], cpu["execution_end_ns"]), (50, 70))
            self.assertEqual(cpu["execution_span_ms"], cpu["host_span_ms"])


if __name__ == "__main__":
    unittest.main()
