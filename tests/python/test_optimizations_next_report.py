"""Finite CPU metadata fixtures; never import model code or access a GPU."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.optimizations.next_protocol import PHASES
from ditserve.optimizations.online import paired_gate, request_identity
from ditserve.optimizations import next_report as report


class NextReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output, self.ledger = self.root / "new/run01", self.root / "new/gpu-budget.json"
        self.config = dict(quality=dict(image_lpips=.01), direction1=dict(prompts=["a", "b"]), seeds=[101, 102, 201, 202],
                           gpu_uuid="GPU-a100-fixture", v100_uuid="GPU-v100-fixture")
        self.workloads = dict(highload={}, diagnosis=[])
        for load in (.5, .7, .9):
            traces = []
            for seed in (101, 102, 103):
                for pattern in ("ordinary", "burst"):
                    requests = []
                    for index in range(24):
                        arrival = (index if pattern == "ordinary" else index // 4 * 4) * 10 / load
                        requests.append(dict(request_id="r%03d" % index,
                            kind="video" if index in (0, 12) else "image", arrival_ms=arrival,
                            deadline_ms=arrival+1000, service_ms=50 if index in (0, 12) else 5,
                            seed=seed+index, status="pending", start_ms=None, finish_ms=None))
                    traces.append(dict(seed=seed, load=load, pattern=pattern, requests=requests,
                                       trace_sha256=request_identity(requests)))
            self.workloads["highload"][str(load)] = traces
        for mixed in (self.workloads["highload"]["0.5"][0], self.workloads["highload"]["0.5"][3]):
            images = [r for r in mixed["requests"] if r["kind"] == "image"]
            self.workloads["diagnosis"].append(dict(parent_trace_sha256=mixed["trace_sha256"],
                seed=mixed["seed"], load=mixed["load"], pattern=mixed["pattern"], mixed=copy.deepcopy(mixed),
                image_only=dict(mixed, requests=images, trace_sha256=request_identity(images))))
        auth_path = self.root / "authorization.json"
        self.auth = dict(approved_by_user=True, scope="serving_next_validation", budget_unlimited=True,
                         authorized_gpu_seconds=None, user_instruction="就按照这个继续；取消8小时GPU小时硬上限，GPU无上限，开始在线对照")
        auth = self.bound(auth_path, self.auth)
        baseline = self.bound(self.root / "old/report.json", dict(archive_verified=True, matrix_complete=True,
            online_evaluation_complete=True, online_qualified=True))
        parents = [self.bound(self.root / ("parent%d.json" % i), dict(active=None, used_gpu_seconds=float(i))) for i in range(3)]
        self.study = dict(config_sha256=digest(self.config), authorization=auth, workloads_sha256=digest(self.workloads),
            budget_unlimited=True, authorized_gpu_seconds=None, historical_ledgers=parents,
            online_baseline_binding=baseline, cache_local_binding=dict(bindings={}), phases=list(PHASES))
        self.budget = dict(active=None, sessions=[], budget_unlimited=True, limit_gpu_seconds=None,
            authorization_binding=auth, config_sha256=digest(self.config), used_gpu_seconds=0.,
            parents={p["path"]:p for p in parents}, stage_used_gpu_seconds={p:0. for p in PHASES})
        atomic_json(self.output / "study.json", self.study)
        atomic_json(self.output / "workloads.json", self.workloads)
        atomic_json(self.ledger, self.budget)

    def bound(self, path, value):
        atomic_json(path, value)
        return dict(path=str(path), sha256=file_digest(path))

    def measured(self, trace, order, arm, factor=1.):
        requests = copy.deepcopy(trace["requests"])
        for request in requests:
            request.update(arrived=True, status="completed", start_ms=request["arrival_ms"]+1,
                finish_ms=request["arrival_ms"]+1+request["service_ms"]*factor,
                service=dict(peak_allocated_bytes=1000))
        return dict(trace_sha256=trace["trace_sha256"], parent_trace_sha256=trace.get("parent_trace_sha256", trace["trace_sha256"]),
            seed=trace["seed"], load=trace["load"], pattern=trace["pattern"], order_index=order, arm=arm,
            status="completed", complete=True, elapsed_ms=1000*factor, requests=requests,
            events=[dict(kind="video_prepare", request_id="r000", start_ms=0., end_ms=1.),
                dict(kind="video_quantum", request_id="r000", start_ms=1., end_ms=40., secret="NO_RAW_EVENT_BODY"),
                dict(kind="video_finish", request_id="r000", start_ms=40., end_ms=51.),
                dict(kind="partial_flush", start_ms=20., end_ms=22.)])

    def cache_rows(self, load=.5):
        return [self.measured(trace, order, arm, 1. if arm == "baseline" else .8)
            for trace in self.workloads["highload"][str(load)] for order in (0, 1) for arm in ("baseline", "candidate")]

    def install(self, phase, data, *, failed=False, gpu_count=1):
        attempt = self.output / phase / "attempt00"
        atomic_json(attempt / "config.json", self.config)
        atomic_json(attempt / "study.json", self.study)
        source = self.output / "sources" / (phase + "-attempt00")
        file = source / "python/fixture.py"
        file.parent.mkdir(parents=True)
        file.write_text("# frozen temporary CPU fixture\n")
        hashes = {"python/fixture.py": file_digest(file)}
        source_sha = digest(hashes)
        atomic_json(source / "manifest.json", dict(source_files=hashes, source_sha256=source_sha,
            config_sha256=digest(self.config), execution_root=str(source)))
        evidence = attempt / report.FILES.get(phase, phase + ".json")
        atomic_json(evidence, data)
        identity = dict(phase=phase, output=str(attempt), run_root=str(self.output),
            config_sha256=digest(self.config), source_sha256=source_sha,
            authorization_source=self.study["authorization"], workloads_sha256=digest(self.workloads),
            gpu_uuids=[self.config["gpu_uuid"]] + ([self.config["v100_uuid"]] if gpu_count == 2 else []),
            auxiliary_encoder=gpu_count == 2)
        charge = dict(session_id=phase, stage=phase, identity=identity, gpu_count=gpu_count,
                      status="failed" if failed else "completed", gpu_seconds=10.*gpu_count, elapsed_seconds=10.)
        self.budget["sessions"].append(charge)
        self.budget["used_gpu_seconds"] += 10.*gpu_count
        self.budget["stage_used_gpu_seconds"][phase] += 10.*gpu_count
        atomic_json(self.ledger, self.budget)
        terminal = dict(status="incomplete" if failed else data["status"], phase=phase,
            output=str(attempt), evidence_path=str(evidence), charge=charge, returncode=1 if failed else 0,
            final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]),
            config_sha256=digest(self.config), source_sha256=source_sha,
            online_qualified=data.get("online_qualified", False), reason=data.get("reason", "fixture"))
        atomic_json(attempt / "supervisor.json", terminal)
        atomic_json(self.output / phase / "terminal.json", terminal)
        return attempt, terminal

    def complete_cache(self, load=.5, failed=False):
        phase = { .5:"cache_load_05", .7:"cache_load_07", .9:"cache_load_09"}[load]
        rows = self.cache_rows(load)
        gate = paired_gate(rows, expected_trace_ids=[r["trace_sha256"] for r in self.workloads["highload"][str(load)]])
        data = dict(status="completed", rows=rows, evaluation_complete=True, comparison=gate,
                    online_qualified=gate["passed"])
        return self.install(phase, data, failed=failed)

    def complete_sana(self):
        phase, candidate = "sana_vae", "vae_compile"
        rows = []
        for order in (0, 1):
            for arm in ("eager", candidate):
                for prompt in (0, 1):
                    for seed in self.config["seeds"]:
                        for repeat in range(3):
                            row = dict(kind="measurement", order_index=order, arm=arm,
                                input_id="p%d-s%d" % (prompt, seed), repeat=repeat, repeat_exact=True,
                                split="holdout" if seed in (201, 202) else "development",
                                service=dict(invocation_wall_ms=100. if arm == "eager" else 90., ms=40. if arm == "eager" else 50.))
                            if repeat == 0:
                                row["quality"] = dict(passed=True, metrics=dict(lpips=0.))
                            rows.append(row)
        attempt_path = self.output / phase / "attempt00"
        recovery = dict(status="passed", candidate=candidate, resume_qualified=True, eligible=True,
            local_qualified=False, online_qualified=False, cache_quality_qualified=False,
            cleanup=dict(passed=True), gate=dict(passed=True, step_count=32, real_sana_insertions=3,
                checks=dict(complete_steps=True, state_checks=True, actual_candidate_insertions=True)))
        recovery_binding = self.bound(attempt_path / "sana-wan-resume.json", recovery)
        online_rows = [self.measured(pair["mixed"], order, arm, 1. if arm == "baseline" else .8)
            for pair in self.workloads["diagnosis"] for order in (0, 1) for arm in ("baseline", "candidate")]
        comparison = paired_gate(online_rows, expected_trace_ids=[p["parent_trace_sha256"] for p in self.workloads["diagnosis"]])
        atomic_json(attempt_path / "sana-online.json", dict(status="completed", rows=online_rows, comparison=comparison))
        data = dict(status="completed", candidate=candidate, rows=rows, round_speed_gains=[.1, .1],
            quality_passed=True, performance_passed=True, local_qualified=True, capacity=dict(passed=True),
            recovery=dict(recovery_binding, status="passed", passed=True, gate=recovery["gate"]),
            online_evaluation_complete=True, online_qualified=True)
        return self.install(phase, data)

    def phase(self, name):
        return next(p for p in report.collect(self.output, self.ledger)["phases"] if p["phase"] == name)

    def complete_conditional(self, components=("video_vae", "text_encoder")):
        attempt = self.output / "conditional/attempt00"
        inputs = ["p%d-s%d" % (p, seed) for p in (0, 1) for seed in (101, 102)]
        aggregate = dict(status="completed", local_qualified=False, online_qualified=False,
                         trigger_evidence=dict(vae_trigger=True, encoding_trigger=True))
        for name in components:
            rows, pairs = [], []
            for index, item in enumerate(inputs):
                if name == "video_vae":
                    for order, arms in enumerate((("eager", "compiled"), ("compiled", "eager"))):
                        for arm in arms:
                            rows.append(dict(kind="measurement", input_id=item, round=order, arm=arm, repeat=0,
                                service=dict(ms=100. if arm == "eager" else 90.), generation_completed=True,
                                quality_status="completed", quality=dict(passed=True, metrics=dict(mean_lpips=.002))))
                    pairs.append(dict(input_id=item, quality_passed=True))
                else:
                    order = ["cpu", "gpu"] if index % 2 == 0 else ["gpu", "cpu"]
                    for arm in order:
                        rows.append(dict(input_id=item, arm=arm, encode_ms=100. if arm == "cpu" else 90.,
                            generation_ms=200. if arm == "cpu" else 180., completed=True, completed_steps=32,
                            cache="paired_008", policy="block-4", wan_backend="eager", device="cpu" if arm == "cpu" else "cuda:1"))
                    pairs.append(dict(input_id=item, order=order, quality=dict(passed=True, metrics=dict(mean_lpips=.002))))
            local_name, online_name, summary_name = (("vae.json", "video-vae-online.json", "conditional-video-vae.json")
                if name == "video_vae" else ("text_encoder.json", "encoder-online.json", "conditional-text-encoder.json"))
            native = dict(status="passed", rows=rows, pairs=pairs, local_qualified=True, quality_epsilon=.05,
                minimum_local_gain=.05, online_qualified=True, online_evaluation_complete=True, online_path=str(attempt / online_name))
            if name == "video_vae":
                native["order_gains"] = [.1, .1]
            else:
                native.update(encode_gain=.1, generation_gain=.1, encoder_device="cuda:1", encoder_dtype="float32",
                    payload_device="cpu", payload_dtype="float32", co_resident_full_generation=True)
            local_binding = self.bound(attempt / local_name, native)
            online_rows = [self.measured(pair["mixed"], order, arm, 1. if arm == "baseline" else .8)
                for pair in self.workloads["diagnosis"] for order in (0, 1) for arm in ("baseline", "candidate")]
            gate = paired_gate(online_rows, expected_trace_ids=[p["parent_trace_sha256"] for p in self.workloads["diagnosis"]])
            self.bound(attempt / online_name, dict(status="completed", evaluation_complete=True, rows=online_rows, comparison=gate))
            summary = dict(status="passed", local_qualified=True, online_qualified=True, online_evaluation_complete=True,
                local_evidence=local_binding, online_path=str(attempt / online_name), dependencies_closed=False)
            if name == "video_vae":
                summary.update(decoder_cleanup_passed=True,
                    compiler=dict(graph_invocations=100, compilation=[dict(status="completed", graph_invocations=100)]))
            binding = self.bound(attempt / summary_name, summary)
            aggregate[name] = dict(summary, triggered=True, component_evidence=binding)
        return self.install("conditional", aggregate, gpu_count=2 if "text_encoder" in components else 1)

    def test_conditional_components_remain_independent_with_bound_quality_and_384_excel_requests(self):
        attempt, terminal = self.complete_conditional()
        value = report.collect(self.output, self.ledger)
        phase = next(p for p in value["phases"] if p["phase"] == "conditional")
        self.assertTrue(phase["authoritative"], phase["errors"])
        self.assertFalse(phase["online_qualified"])
        self.assertIn("组合未验证", phase["decision"])
        for name, count in (("video_vae", 16), ("text_encoder", 8)):
            component = phase["components"][name]
            self.assertTrue(component["local_qualified"])
            self.assertTrue(component["online_qualified"])
            self.assertEqual(component["local"]["measured_count"], count)
            self.assertEqual(component["online"]["recorded_requests"], 192)
            self.assertEqual(component["online"]["recorded_replays"], 8)
            self.assertAlmostEqual(component["local"]["recomputed_gains"][0], .1)
        table = report._tables(value)
        rows = [r for r in table["Requests"][1:] if r[0].startswith("conditional/")]
        self.assertEqual(len(rows), 384)
        self.assertEqual({r[0] for r in rows}, {"conditional/video_vae", "conditional/text_encoder"})
        self.assertEqual({Path(r[-3]).name for r in rows}, {"video-vae-online.json", "encoder-online.json"})
        self.assertTrue(all(r[-1].startswith("/rows/") for r in rows))
        self.assertNotIn("NO_RAW_EVENT_BODY", json.dumps(value))
        self.assertEqual(value["budget"]["settled_gpu_seconds"], 20.)

    def test_conditional_local_hash_or_summary_binding_and_wrong_devices_block_promotion(self):
        attempt, terminal = self.complete_conditional()
        local = attempt / "text_encoder.json"
        saved = local.read_bytes()
        atomic_json(local, dict(read_json(local), changed=True))
        self.assertFalse(self.phase("conditional")["authoritative"])
        local.write_bytes(saved)
        terminal["charge"]["identity"]["gpu_uuids"].reverse()
        self.budget["sessions"][-1] = terminal["charge"]
        atomic_json(self.ledger, self.budget)
        atomic_json(attempt / "supervisor.json", terminal)
        atomic_json(self.output / "conditional/terminal.json", terminal)
        phase = self.phase("conditional")
        self.assertFalse(phase["authoritative"])
        self.assertFalse(phase["components"]["text_encoder"]["online_qualified"])

    def test_conditional_partial_online_is_not_promoted_by_complete_local_or_summary_flags(self):
        attempt, _ = self.complete_conditional()
        path = attempt / "video-vae-online.json"
        value = read_json(path)
        value["rows"] = value["rows"][:-1]
        value["status"] = "incomplete"
        atomic_json(path, value)
        phase = self.phase("conditional")
        self.assertTrue(phase["authoritative"], phase["errors"])
        component = phase["components"]["video_vae"]
        self.assertTrue(component["local_qualified"])
        self.assertFalse(component["online_qualified"])
        self.assertFalse(component["evaluation_complete"])
        self.assertEqual(component["online"]["expected_requests"], 192)
        self.assertEqual(component["online"]["recorded_requests"], 168)
        self.assertFalse(phase["online_qualified"])

    def test_conditional_independent_failure_retains_rows_and_does_not_invalidate_other_component(self):
        attempt, _ = self.complete_conditional()
        native = attempt / "vae.json"
        data = read_json(native)
        data.update(local_qualified=False, online_qualified=False, online_evaluation_complete=False, status="stopped")
        data["rows"] = data["rows"][:2]
        data["pairs"] = []
        data["rows"][-1]["quality"]["passed"] = False
        data.pop("online_path")
        binding = self.bound(native, data)
        summary = dict(status="stopped", local_qualified=False, online_qualified=False,
                       online_evaluation_complete=False, local_evidence=binding, reason="vae_quality_failed")
        component_binding = self.bound(attempt / "conditional-video-vae.json", summary)
        aggregate = read_json(attempt / "conditional.json")
        aggregate["video_vae"] = dict(summary, triggered=True, component_evidence=component_binding)
        atomic_json(attempt / "conditional.json", aggregate)
        (attempt / "video-vae-online.json").unlink()
        phase = self.phase("conditional")
        self.assertTrue(phase["authoritative"], phase["errors"])
        failed = phase["components"]["video_vae"]
        self.assertEqual(failed["local"]["measured_count"], 2)
        self.assertEqual(failed["local"]["quality_failed_count"], 1)
        self.assertFalse(failed["online_qualified"])
        self.assertTrue(phase["components"]["text_encoder"]["online_qualified"])

    def test_active_conditional_multiplies_full_holding_time_by_two_and_other_phase_cannot(self):
        attempt, terminal = self.complete_conditional()
        self.budget["active"] = dict(stage="conditional", identity=terminal["charge"]["identity"],
                                     started_monotonic=100., gpu_count=2)
        atomic_json(self.ledger, self.budget)
        with patch.object(report.time, "monotonic", return_value=123.):
            result = report.collect(self.output, self.ledger)
        self.assertEqual(result["budget"]["active_gpu_seconds_estimate"], 46.)
        self.assertEqual(result["budget"]["total_gpu_seconds_including_active"], 66.)
        self.assertFalse(next(p for p in result["phases"] if p["phase"] == "conditional")["authoritative"])
        self.assertIsNone(report._device_count(terminal["charge"], "diagnosis", self.config))

    def test_conditional_component_paths_cannot_escape_attempt(self):
        attempt, _ = self.complete_conditional(("video_vae",))
        summary_path = attempt / "conditional-video-vae.json"
        summary = read_json(summary_path)
        summary["online_path"] = str(self.root / "other/video-vae-online.json")
        summary["local_evidence"] = dict(path=str(self.root / "other/vae.json"), sha256="0"*64)
        component_binding = self.bound(summary_path, summary)
        data = read_json(attempt / "conditional.json")
        data["video_vae"] = dict(summary, triggered=True, component_evidence=component_binding)
        atomic_json(attempt / "conditional.json", data)
        phase = self.phase("conditional")
        self.assertFalse(phase["authoritative"])
        self.assertTrue(any("fixed attempt path" in e for e in phase["errors"]))

    def test_conditional_cleanup_revocation_preserves_local_facts_without_promoting(self):
        attempt, _ = self.complete_conditional(("video_vae",))
        summary_path = attempt / "conditional-video-vae.json"
        summary = read_json(summary_path)
        summary.update(status="failed", decoder_cleanup_passed=False, local_qualified=False,
                       online_qualified=False, online_evaluation_complete=False, reason="decoder cleanup failed")
        binding = self.bound(summary_path, summary)
        aggregate = read_json(attempt / "conditional.json")
        aggregate["video_vae"] = dict(summary, triggered=True, component_evidence=binding)
        atomic_json(attempt / "conditional.json", aggregate)
        phase = self.phase("conditional")
        self.assertTrue(phase["authoritative"], phase["errors"])
        component = phase["components"]["video_vae"]
        self.assertTrue(component["local"]["quality_and_speed_gate"])
        self.assertFalse(component["local_qualified"])
        self.assertFalse(component["online_qualified"])
        self.assertEqual(component["reason"], "decoder cleanup failed")

    def test_conditional_zero_compiled_invocations_cannot_claim_local_success(self):
        attempt, _ = self.complete_conditional(("video_vae",))
        summary_path = attempt / "conditional-video-vae.json"
        summary = read_json(summary_path)
        summary["compiler"]["graph_invocations"] = 0
        binding = self.bound(summary_path, summary)
        aggregate = read_json(attempt / "conditional.json")
        aggregate["video_vae"] = dict(summary, triggered=True, component_evidence=binding)
        atomic_json(attempt / "conditional.json", aggregate)
        phase = self.phase("conditional")
        self.assertFalse(phase["authoritative"])
        self.assertFalse(phase["components"]["video_vae"]["online_qualified"])

    def test_complete_cache_has_576_requests_and_separate_load_pattern_seed_trace_pools(self):
        self.complete_cache(.5)
        self.complete_cache(.7)
        result = report.collect(self.output, self.ledger)
        for phase in (p for p in result["phases"] if p["phase"] in ("cache_load_05", "cache_load_07")):
            self.assertTrue(phase["authoritative"], phase["errors"])
            self.assertTrue(phase["online_qualified"])
            online = phase["online"]
            self.assertEqual((online["recorded_replays"], online["recorded_requests"]), (24, 576))
            groups = online["groups"]
            self.assertEqual(len([g for g in groups if g["group"] == "trace"]), 12)
            for group in (g for g in groups if g["group"] == "pooled"):
                self.assertEqual(group["metrics"]["recorded_requests"], 288)
                self.assertEqual(group["metrics"]["images"]["sample_count"], 264)
                self.assertEqual(group["metrics"]["videos"]["arrived"], 24)
            self.assertEqual(len([g for g in groups if g["group"] == "seed"]), 6)
            self.assertEqual(len([g for g in groups if g["group"] == "pattern"]), 4)
        self.assertNotIn("NO_RAW_EVENT_BODY", json.dumps(result))

    def test_supervisor_failure_never_qualifies_complete_worker_matrix(self):
        self.complete_cache(failed=True)
        phase = self.phase("cache_load_05")
        self.assertTrue(phase["online"]["matrix_complete"])
        self.assertFalse(phase["authoritative"])
        self.assertFalse(phase["online_qualified"])
        self.assertEqual(phase["decision"], "未验证")

    def test_partial_active_trace_retains_failed_arrivals_and_avoids_duplicate_trace(self):
        rows = self.cache_rows()
        partial = copy.deepcopy(rows[1])
        partial.update(status="failed", complete=False)
        for index, request in enumerate(partial["requests"]):
            request.update(arrived=index < 4, status="completed" if index < 2 else "failed" if index == 2 else "pending")
            if index >= 2:
                request.update(start_ms=None, finish_ms=None)
        data = dict(status="incomplete", rows=rows[:1], evaluation_complete=False)
        attempt, terminal = self.install("cache_load_05", data, failed=True)
        path = attempt / "partials/replay-001/trace.partial.json"
        atomic_json(path, partial)
        data["active_trace_path"] = str(path)
        atomic_json(attempt / "cache_load_05.json", data)
        phase = self.phase("cache_load_05")
        self.assertEqual(phase["online"]["recorded_requests"], 48)
        candidate = next(g["metrics"]["all"] for g in phase["online"]["groups"] if g["group"] == "pooled" and g["arm"] == "candidate")
        self.assertEqual((candidate["arrived"], candidate["completed"], candidate["failed_or_incomplete"]), (4, 2, 2))
        atomic_json(path, rows[0])
        self.assertEqual(self.phase("cache_load_05")["online"]["recorded_requests"], 24)

    def test_pressure_intervals_and_memory_are_observed_not_capacity_claims(self):
        row = dict(elapsed_ms=10., requests=[
            dict(kind="image", arrived=True, arrival_ms=0., start_ms=2., finish_ms=4., status="completed", service=dict(peak_allocated_bytes=200)),
            dict(kind="video", arrived=True, arrival_ms=1., start_ms=3., finish_ms=None, status="pending")],
            events=[dict(kind="video_prepare", request_id="v", start_ms=2., end_ms=3.),
                    dict(kind="video_quantum", request_id="v", start_ms=3., end_ms=5.)])
        value = report._pressure(row)
        self.assertEqual(value["pending"]["maximum"], 2)
        self.assertAlmostEqual(value["pending"]["time_weighted_mean"], .4)
        self.assertEqual(value["inflight"]["maximum"], 2)
        self.assertAlmostEqual(value["inflight"]["time_weighted_mean"], .9)
        self.assertEqual(value["unfinished_arrived_at_end"], 1)
        self.assertEqual(value["image_request_peak_allocated_bytes"]["maximum"], 200)
        self.assertIn("not a paused-state capacity", value["video_state_scope"])

    def test_completed_local_requires_full_wall_speed_quality_capacity_and_independent_recovery(self):
        self.complete_sana()
        phase = self.phase("sana_vae")
        self.assertTrue(phase["authoritative"], phase["errors"])
        self.assertTrue(phase["online_qualified"], phase["online"]["comparison"])
        local = phase["local"]
        self.assertTrue(local["full_local_matrix"])
        self.assertTrue(local["quality_and_speed_gate"])
        self.assertTrue(local["recovery"]["passed"])
        self.assertEqual((local["measured_count"], local["quality_checked"]), (96, 32))
        self.assertAlmostEqual(local["recomputed_round_speed_gains"][0], .1)
        candidate = next(r for r in local["rows"] if r["arm"] == "vae_compile")
        self.assertEqual((candidate["service_ms"], candidate["pipeline_ms"]), (90., 50.))

    def test_recovery_source_cleanup_and_complete_step_gate_fail_closed(self):
        attempt, _ = self.complete_sana()
        path = attempt / "sana-wan-resume.json"
        native_path = attempt / "sana-candidate.json"
        original, native = read_json(path), read_json(native_path)
        for kind in ("missing", "hash", "cleanup", "steps", "insertions", "qualification"):
            with self.subTest(kind=kind):
                changed, record = copy.deepcopy(original), copy.deepcopy(native)
                if kind == "missing":
                    path.unlink()
                elif kind == "hash":
                    changed["extra"] = "tampered"
                    atomic_json(path, changed)
                else:
                    if kind == "cleanup":
                        changed["cleanup"]["passed"] = False
                    if kind == "steps":
                        changed["gate"]["step_count"] = 31
                    if kind == "insertions":
                        changed["gate"]["real_sana_insertions"] = 2
                    if kind == "qualification":
                        changed["resume_qualified"] = False
                    atomic_json(path, changed)
                    record["recovery"].update(sha256=file_digest(path), gate=changed["gate"])
                atomic_json(native_path, record)
                phase = self.phase("sana_vae")
                self.assertFalse(phase["local"]["recovery"]["passed"])
                self.assertFalse(phase["authoritative"])
                self.assertFalse(phase["online_qualified"])
                atomic_json(path, original)
                atomic_json(native_path, native)

    def test_pipeline_speed_cannot_substitute_for_full_invocation_or_missing_quality(self):
        attempt, _ = self.complete_sana()
        path = attempt / "sana-candidate.json"
        original = read_json(path)
        for kind in ("pipeline_only_gain", "missing_quality", "wrong_frozen_input", "recorded_gain"):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(original)
                if kind == "pipeline_only_gain":
                    for row in changed["rows"]:
                        if row["arm"] == "vae_compile":
                            row["service"].update(invocation_wall_ms=110., ms=1.)
                elif kind == "missing_quality":
                    changed["rows"][0].pop("quality")
                elif kind == "wrong_frozen_input":
                    for row in changed["rows"]:
                        if row["input_id"] == "p0-s101":
                            row["input_id"] = "unregistered-input"
                else:
                    changed["round_speed_gains"] = [.3, .3]
                atomic_json(path, changed)
                phase = self.phase("sana_vae")
                self.assertFalse(phase["local"]["quality_and_speed_gate"])
                self.assertFalse(phase["online_qualified"])
        atomic_json(path, original)

    def test_explicit_reader_bound_does_not_silently_drop_oversized_evidence(self):
        path = self.root / "bounded.json"
        atomic_json(path, {"payload": "a"*100})
        errors = []
        with patch.object(report, "MAX_EVIDENCE_BYTES", 10):
            value, binding = report._Reader().read(path, errors)
        self.assertEqual(value, {})
        self.assertIsNone(binding)
        self.assertIn("reporting bound", errors[0])
        self.assertEqual(report.MAX_EVIDENCE_BYTES, 128*1024*1024)

    def test_finalization_dispatches_verified_archive_view_and_never_rewrites(self):
        frozen = report.collect(self.output, self.ledger)
        frozen.update(archive_only=True, raw_re_evaluable=False, qualification_recomputed=False)
        atomic_json(self.output / "next-archive/finalization.json", dict(status="completed"))
        loader = Mock(return_value=frozen)
        with patch.dict(sys.modules, {"ditserve.optimizations.next_retention": SimpleNamespace(load_archive_report=loader)}):
            self.assertEqual(report.collect(self.output, self.ledger), frozen)
            loader.assert_called_once_with(self.output, self.ledger)
            self.assertIn("归档只读", report.render(frozen))
            with self.assertRaisesRegex(ValueError, "immutable"):
                report.write_report(self.output, self.ledger)
        self.assertFalse((self.output / "results.xlsx").exists())

    def test_incomplete_or_invalid_finalization_cannot_fall_back_to_raw(self):
        atomic_json(self.output / "next-archive/finalization.json", dict(status="in_progress"))
        loader = Mock(side_effect=ValueError("archive integrity failed"))
        with patch.dict(sys.modules, {"ditserve.optimizations.next_retention": SimpleNamespace(load_archive_report=loader)}):
            with self.assertRaisesRegex(ValueError, "archive integrity"):
                report.collect(self.output, self.ledger)

    def test_diagnosis_compares_same_image_metrics_without_goodput_gate(self):
        rows = []
        for pair in self.workloads["diagnosis"]:
            for order in (0, 1):
                for arm in ("image_only", "mixed"):
                    rows.append(self.measured(dict(pair[arm], parent_trace_sha256=pair["parent_trace_sha256"]), order, arm))
        self.install("diagnosis", dict(status="completed", rows=rows, evaluation_complete=True))
        phase = self.phase("diagnosis")
        self.assertTrue(phase["evaluation_complete"])
        self.assertFalse(phase["online_qualified"])
        self.assertEqual(phase["online"]["recorded_requests"], 184)
        for group in phase["online"]["groups"]:
            self.assertNotIn("goodput_rps", group["metrics"]["images"])
        self.assertIsNone(phase["online"]["comparison"])

    def test_local_quality_failure_remains_local_and_native_rows_survive_failed_envelope(self):
        data = dict(status="stopped", quality_passed=False, rows=[dict(kind="measurement", arm="vae_compile",
            input_id="p0-s101", split="development", repeat=0, order_index=0, repeat_exact=True,
            service=dict(ms=1.), quality=dict(passed=False, metrics=dict(lpips=.02)))])
        attempt, _ = self.install("sana_vae", data)
        phase = self.phase("sana_vae")
        self.assertTrue(phase["authoritative"])
        self.assertEqual(phase["local"]["quality_failed_count"], 1)
        self.assertIn("局部质量", phase["decision"])
        self.assertFalse(phase["online_qualified"])
        terminal = read_json(attempt / "supervisor.json")
        terminal.update(status="failed", returncode=1)
        atomic_json(attempt / "supervisor.json", terminal)
        atomic_json(self.output / "sana_vae/terminal.json", terminal)
        phase = self.phase("sana_vae")
        self.assertEqual(phase["local"]["measured_count"], 1)
        self.assertEqual(phase["decision"], "未验证")

    def test_numerics_failure_is_retained_without_any_performance_qualification(self):
        rows = [dict(unit_index=i, arm="compiled_single", status="failed" if i == 11 else "completed",
            input_ids=["p0"], error="compiler failed" if i == 11 else None, execution=dict(ms=99999),
            comparisons=[] if i == 11 else [dict(input_id="p0", status="completed",
                trajectory=dict(first_exact_drift="step0.transformer", first_strict_drift=None, all_exact=False, all_strict=True),
                output=dict(passed=True, metrics=dict(lpips=.001)))]) for i in range(12)]
        self.install("sana_numerics", dict(status="passed", rows=rows, diagnostic_complete=True))
        phase = self.phase("sana_numerics")
        self.assertTrue(phase["authoritative"])
        self.assertEqual(phase["numerics"]["failed_units"], 1)
        self.assertFalse(phase["numerics"]["performance_claims"])
        self.assertFalse(phase["online_qualified"])
        self.assertNotIn("99999", json.dumps(phase))

    def test_active_ledger_is_unsettled_and_failed_sources_block_promotion(self):
        attempt, terminal = self.complete_cache()
        self.budget["active"] = dict(stage="cache_load_05", identity=terminal["charge"]["identity"],
                                     started_monotonic=time.monotonic()-3, gpu_count=1)
        atomic_json(self.ledger, self.budget)
        result = report.collect(self.output, self.ledger)
        self.assertEqual(result["status"], "running")
        self.assertTrue(result["budget"]["active_unsettled"])
        self.assertGreaterEqual(result["budget"]["active_gpu_seconds_estimate"], 3)
        self.assertFalse(next(p for p in result["phases"] if p["phase"] == "cache_load_05")["online_qualified"])
        self.budget["active"] = None
        atomic_json(self.ledger, self.budget)
        source = self.output / "sources/cache_load_05-attempt00/python/fixture.py"
        source.write_text("changed")
        self.assertFalse(self.phase("cache_load_05")["authoritative"])

    def test_changed_authorization_cost_and_recorded_gate_fail_closed(self):
        attempt, _ = self.complete_cache()
        auth = Path(self.study["authorization"]["path"])
        old = auth.read_bytes()
        atomic_json(auth, dict(self.auth, approved_by_user=False))
        self.assertFalse(self.phase("cache_load_05")["authoritative"])
        auth.write_bytes(old)
        self.budget["used_gpu_seconds"] += 1
        atomic_json(self.ledger, self.budget)
        self.assertFalse(self.phase("cache_load_05")["authoritative"])
        self.budget["used_gpu_seconds"] -= 1
        atomic_json(self.ledger, self.budget)
        data = read_json(attempt / "cache_load_05.json")
        data["comparison"]["passed"] = False
        atomic_json(attempt / "cache_load_05.json", data)
        self.assertFalse(self.phase("cache_load_05")["authoritative"])

    def test_missing_native_evidence_is_unverified_not_empty_success(self):
        attempt, _ = self.complete_cache()
        (attempt / "cache_load_05.json").unlink()
        phase = self.phase("cache_load_05")
        self.assertTrue(phase["errors"])
        self.assertIsNone(phase["evidence"])
        self.assertFalse(phase["authoritative"])

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "existing reporting environment provides openpyxl")
    def test_excel_readback_is_exact_literal_and_does_not_include_event_bodies_or_delete_sources(self):
        self.complete_cache()
        before = {str(p):p.read_bytes() for p in self.output.rglob("*.json")}
        result = report.write_report(self.output, self.ledger)
        self.assertEqual(result["excel"]["status"], "verified")
        self.assertEqual(result["requests"], 576)
        self.assertEqual(result["originals_deleted"], 0)
        self.assertEqual(result["excel"]["rows_by_sheet"]["Requests"], 576)
        from openpyxl import load_workbook
        book = load_workbook(result["excel"]["path"], read_only=True)
        values = [str(cell.value) for sheet in book for row in sheet for cell in row]
        book.close()
        self.assertNotIn("NO_RAW_EVENT_BODY", " ".join(values))
        for path, raw in before.items():
            self.assertEqual(Path(path).read_bytes(), raw)
        with patch("ditserve.compact_results.export", side_effect=AssertionError("no compaction")), \
             patch("ditserve.compact_results.prune", side_effect=AssertionError("no cleanup")):
            report.write_report(self.output, self.ledger)

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "existing reporting environment provides openpyxl")
    def test_empty_report_does_not_create_authorization_or_ledger(self):
        other = self.root / "empty"
        result = report.write_report(other / "run01", other / "gpu-budget.json")
        self.assertEqual(result["status"], "not_started")
        self.assertFalse((other / "gpu-budget.json").exists())
        self.assertEqual(result["requests"], 0)


if __name__ == "__main__":
    unittest.main()
