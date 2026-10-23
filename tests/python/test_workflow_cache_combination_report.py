"""CPU receipt producers and mutation tests for the combined-cache study."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_cache_combination_report as audit
from ditserve.evaluation import workflow_cache_combination_study as study
from ditserve.evaluation.workflow_next_study import PROMPTS
from ditserve.evaluation.workflow_serving_study import digest
from ditserve.models.sana.backend import SanaBackend
from test_workflow_image_cache_audit import FakeAdapter, receipt
from test_workflow_occupancy_report import fixture as occupancy_fixture
from test_workflow_encoding_optimization_report import launch as raw_launch


def fixture(repeat=0):
    original, job, old_config = occupancy_fixture()
    rows = copy.deepcopy(old_config["phases"][0]["workload"])
    for index, row in enumerate(r for r in rows if r["kind"] == "image"):
        row["spec"]["prompt"] = PROMPTS[index % 4]
    config = study.native_config(old_config["phases"][0]["profile"], rows, repeat=repeat)
    old = {p["id"]: p for p in original["phases"]}
    observed = []
    with patch("ditserve.models.sana.runtime.SanaAdapter", FakeAdapter):
        for phase in config["phases"]:
            name = phase["id"]
            actual = old[name]
            cell = actual["cells"][0]
            planned = {r["request_id"]: r for r in phase["workload"]}
            for row in cell["requests"]:
                if row["kind"] == "image":
                    row["spec"] = dict(planned[row["request_id"]]["spec"], request_id=row["request_id"])
            cell["guard_telemetry"]["occupancy_backend"] = "nvml"
            for event in cell["guard_telemetry"]["events"]:
                event["occupancy_backend"] = "nvml"
            for slot in (0, 1):
                worker = f"sana{slot}"
                backend = SanaBackend(dict(phase["worker_overrides"][worker], model_path="/fake"))
                warm = cell["warmup"][slot + 2]
                warm["prompt_cache"] = receipt(backend, f"warmup-{slot + 2}", warm["spec"]["prompt"])
                for row in sorted((r for r in cell["requests"] if r["kind"] == "image"
                                   and worker in r["worker_pids"]), key=lambda r: r["start_ms"]):
                    packet = receipt(backend, row["request_id"], row["spec"]["prompt"], row["spec"]["seed"])
                    row["outputs"][0]["prompt_cache"] = packet
                    row["model_ms"] = packet["service_ms"]
                backend.close()
            observed.append(actual)
    result = dict(original, config=copy.deepcopy(config), phases=observed)
    for worker in result["config"]["workers"].values():
        worker["gpu_uuid"] = f"GPU-{worker['gpu']}"
    job.update(id=study.job_id(study.SEEDS[repeat]), configuration_sha256=digest(config),
        metadata=dict(role="cache_combination_online", repeat=repeat, seed=study.SEEDS[repeat],
                      stage="confirmation", monitor_cpu_affinity=[58]))
    return result, job, config


def assessment_rows(services=(8., 6., -2.), tails=(2., 1., -1.)):
    return [dict(seed=seed, stage="confirmation", qualified=True,
        performance_assessment=dict(safety_passed=True, mean_service_reduction_percent=service,
                                    image_p95_reduction_percent=tail))
        for seed, service, tail in zip(study.SEEDS, services, tails)]


class CacheCombinationReportTests(unittest.TestCase):
    def test_real_receipt_replay_allows_route_dependent_hits_and_keeps_timing_scopes(self):
        for repeat in (0, 1, 2):
            result, job, config = fixture(repeat)
            checked = audit.audit_result(result, job, config)
            self.assertTrue(checked["passed"], checked["errors"])
            self.assertEqual(checked["exact"]["reference/candidate"]["requests"], 24)
            self.assertEqual(sum(w["measured"]["hits"] for w in checked["arms"]["candidate"]["sana"].values()), 17)
            service = checked["service_pairing"]["subsets"]
            self.assertEqual(service["all_candidate_hit"]["requests"], 17)
            self.assertGreater(service["all"]["worker_model_mean_ms"]["improvement_percent"], 5)
            self.assertEqual(service["all"]["host_dispatch_mean_ms"]["improvement_percent"], 0)
            self.assertIn("excluding later harvest", checked["service_pairing"]["timing_scope"])

    def test_changed_controls_or_fused_cache_are_rejected_even_with_new_digest(self):
        for change in ("guard", "refresh", "fused", "wan_cache", "quantum", "capacity"):
            result, job, config = fixture()
            phase = config["phases"][-1]
            if change == "guard":
                phase["occupancy_backend"] = "nvidia_smi"
            elif change == "refresh":
                phase["dispatch_refresh"] = True
            elif change == "fused":
                phase["worker_overrides"]["sana0"]["options"]["prompt_cache_sync_mode"] = "fused"
            elif change == "wan_cache":
                phase["worker_overrides"]["wan0"]["options"] = {}
            elif change == "quantum":
                phase["options"]["quantum_ms"] = 100
            else:
                phase["worker_overrides"]["sana0"]["options"]["prompt_cache_bytes"] *= 2
            result["config"]["phases"] = copy.deepcopy(config["phases"])
            job["configuration_sha256"] = digest(config)
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_warmup_routing_cache_owner_and_model_receipt_remain_bound(self):
        for change in ("warm_pid", "warm_prompt", "warm_receipt", "route", "owner", "outcome", "service", "counters"):
            result, job, config = fixture()
            cell = result["phases"][-1]["cells"][0]
            row = next(r for r in cell["requests"] if r["kind"] == "image")
            packet = row["outputs"][0]["prompt_cache"]
            if change == "warm_pid":
                cell["warmup"][2]["worker_pid"] += 1
            elif change == "warm_prompt":
                cell["warmup"][2]["spec"]["prompt"] = "different"
            elif change == "warm_receipt":
                cell["warmup"][2].pop("prompt_cache")
            elif change == "route":
                row["dispatch_timeline"][0]["group"] = [1 - row["dispatch_timeline"][0]["group"][0]]
            elif change == "owner":
                packet["adapter_id"] = "e" * 32
            elif change == "outcome":
                packet["outcome"] = "miss" if packet["outcome"] == "hit" else "hit"
            elif change == "service":
                row["model_ms"] -= 1
            else:
                packet["counters"]["hits"] += 1
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_wan_ticket_pid_and_negative_cache_are_still_checked(self):
        for change in ("ticket", "pid", "cache", "negative"):
            result, job, config = fixture()
            row = next(r for r in result["phases"][-1]["cells"][0]["requests"] if r["kind"] == "video")
            if change == "ticket":
                row["encoding_events"][0]["ticket"] = "other"
            elif change == "pid":
                row["encoding_events"][0]["pid"] += 1
            elif change == "cache":
                row.pop("negative_prompt_cache")
            else:
                row["negative_prompt_cache"]["negative_prompt_sha256"] = "0" * 64
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_guard_full_work_hashes_and_model_windows_fail_closed(self):
        for change in ("owners", "uuid", "guard_backend", "blocks", "decode", "latent", "short_hash", "model_window"):
            result, job, config = fixture()
            cell = result["phases"][-1]["cells"][0]
            video = next(r for r in cell["requests"] if r["kind"] == "video")
            if change == "owners":
                cell["guard_telemetry"]["events"][-1].update(owned_count=0, owned_processes=[])
            elif change == "uuid":
                result["config"]["workers"]["sana0"]["gpu_uuid"] = "GPU-wrong"
            elif change == "guard_backend":
                cell["guard_telemetry"]["events"][-1]["occupancy_backend"] = "nvidia_smi"
            elif change == "blocks":
                video["dispatch_timeline"][0]["executed_blocks"] -= 1
            elif change == "decode":
                video["dispatch_timeline"].pop()
            elif change == "latent":
                video["outputs"][0]["latent"]["sha256"] = "0" * 64
            elif change == "short_hash":
                for phase in result["phases"]:
                    phase["cells"][0]["requests"][0]["outputs"][0]["output"]["sha256"] = "short"
            else:
                row = next(r for r in cell["requests"] if r["kind"] == "image")
                event = row["dispatch_timeline"][0]
                event["rpc_completed_ms"] = event["start_ms"]
                event["harvest_delay_ms"] = event["harvested_ms"] - event["start_ms"]
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)
        self.assertFalse(audit.audit_result({}, fixture()[1], fixture()[2])["passed"])

    def test_screen_uses_worker_model_service_and_three_independent_safety_bounds(self):
        result, job, config = fixture()
        gate = audit.screening_gate(dict(qualified=True, capacity=True, result=result))
        self.assertTrue(gate["passed"], gate)
        checked = audit.audit_result(result, job, config)
        arms = {name: audit._metrics(cell, checked["arms"][name]) for name, cell in audit.report.native_arms(result).items()}
        changed = copy.deepcopy(arms)
        changed["candidate"]["image_service"]["worker_model_ms"]["mean"] = changed["reference"]["image_service"]["worker_model_ms"]["mean"] * .951
        self.assertFalse(audit.performance_assessment(changed)["passed"])
        for kind, name, ratio in (("image", "latency_p95_ms", 1.051), ("video", "latency_p95_ms", 1.051), ("total", "throughput_per_second", .969)):
            changed = copy.deepcopy(arms)
            changed["candidate"][kind][name] = changed["reference"][kind][name] * ratio
            self.assertFalse(audit.performance_assessment(changed)["passed"])
        self.assertFalse(audit.screening_gate(dict(qualified=False, capacity=True))["passed"])

    def test_confirmation_uses_paired_seed_median_not_every_seed_screen_threshold(self):
        rows = assessment_rows()
        result = audit.confirmation_assessment(rows)
        self.assertTrue(result["promotable"])
        self.assertEqual(result["mean_service_reduction_median_percent"], 6.)
        self.assertEqual(result["service_improving_seeds"], 2)
        self.assertFalse(result["image_tail_benefit"])
        rows.append(dict(seed=4500, stage="screening", qualified=True,
                         performance_assessment=dict(mean_service_reduction_percent=99.)))
        self.assertEqual(audit.confirmation_assessment(rows), result)
        self.assertFalse(audit.confirmation_assessment(assessment_rows((100., -1., -2.)))["promotable"])

    def test_tail_claim_is_separate_and_all_seed_safety_is_required(self):
        rows = assessment_rows(tails=(8., 6., -2.))
        self.assertTrue(audit.confirmation_assessment(rows)["image_tail_benefit"])
        rows[-1]["performance_assessment"]["safety_passed"] = False
        result = audit.confirmation_assessment(rows)
        self.assertTrue(result["qualified"])
        self.assertFalse(result["promotable"])
        self.assertFalse(result["image_tail_benefit"])
        self.assertFalse(audit.confirmation_assessment(rows[:2])["qualified"])
        rows[-1]["qualified"] = False
        self.assertFalse(audit.confirmation_assessment(rows)["qualified"])

    def test_unstarted_confirmations_do_not_become_numerical_failures(self):
        jobs = [dict(id=study.job_id(seed), decision="cache_combination_screening_not_promising",
                     metadata=dict(role=audit.ROLE, seed=seed, stage="confirmation")) for seed in study.SEEDS]
        with patch.object(audit.report, "evidence", return_value=dict(launch={}, result={}, qualified=False, capacity=False)), \
                patch.object(audit, "audit_result", side_effect=AssertionError("unstarted")):
            out = audit.summarize(dict(jobs=jobs))
        self.assertFalse(out["qualified"])
        self.assertFalse(out["promotable"])
        self.assertIsNone(out["paired_changes_median"])
        self.assertTrue(all(r["status"] == "not_started" and r["audit"] is None for r in out["per_seed"]))

    def test_summary_keeps_side_effects_and_rechecks_prerequisites(self):
        result, job, config = fixture()
        ev = dict(launch=dict(status="completed"), result=result, qualified=True, capacity=True)
        jobs = [dict(job, id=study.job_id(seed), input="fixture", requires=["numeric"],
            metadata=dict(role=audit.ROLE, repeat=0, stage="confirmation", seed=seed)) for seed in study.SEEDS]
        with patch.object(audit.report, "evidence", return_value=ev), patch.object(audit.report, "current_job", return_value={}), \
                patch.object(audit, "read", return_value=config), patch.object(audit.encoding, "raw_capacity", return_value=dict(passed=True)):
            out = audit.summarize(dict(jobs=jobs))
            self.assertTrue(out["qualified"])
            self.assertTrue(out["promotable"])
            self.assertFalse(out["default_change"])
            self.assertFalse(out["image_tail_benefit"])
            paired = out["paired_changes_median"]["reference/candidate"]
            self.assertIn("maximum_latency_ms", paired["image"])
            self.assertIn("cold_probe_ms", paired["guard"])
            self.assertIn("slo_attainment", paired["image"])
            with patch.object(audit.report, "current_job", side_effect=ValueError("missing")):
                self.assertFalse(audit.summarize(dict(jobs=jobs))["qualified"])

    def test_central_evidence_dispatch_checks_affinity_raw_capacity_and_dependencies(self):
        result, job, config = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job.update(input=str(root / "input.json"), output=str(root / "output"), requires=["numeric"])
            gate = dict(id="numeric", output=str(root / "gate"), devices=job["devices"])
            for current, payload in ((job, result), (gate, dict(status="completed"))):
                output = Path(current["output"]) / "run.json"
                output.parent.mkdir()
                raw = json.dumps(payload).encode()
                output.write_bytes(raw)
                launch = raw_launch(job)
                launch.update(status="completed", output_sha256=hashlib.sha256(raw).hexdigest(), observer_cpu_affinity=[58])
                path = root / "launches" / current["id"] / "launch.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(launch))
            Path(job["input"]).write_text(json.dumps(config))
            context = dict(root=str(root), jobs=[job, gate])
            self.assertTrue(audit.report.evidence(context, job)["qualified"])
            path = root / "launches" / job["id"] / "launch.json"
            original = json.loads(path.read_text())
            for change in ("affinity", "raw"):
                launch = copy.deepcopy(original)
                if change == "affinity":
                    launch["observer_cpu_affinity"] = [0]
                else:
                    launch["gpu_monitor"]["samples"][0]["uuid"] = "GPU-wrong"
                path.write_text(json.dumps(launch))
                self.assertFalse(audit.report.evidence(context, job)["qualified"], change)
            path.write_text(json.dumps(original))
            self.assertFalse(audit.report.evidence(dict(root=str(root), jobs=[job]), job)["qualified"])

    def test_build_report_includes_combination_summary_without_changing_old_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = dict(id=study.job_id(4500), family="extension", metadata=dict(role=audit.ROLE))
            (root / "study.json").write_text(json.dumps(dict(root=str(root), jobs=[job], selections={})))
            ev = dict(launch={}, result={}, guarded=False, qualified=False, capacity=False, path=root / "missing.json")
            sentinel = dict(schema=audit.SCHEMA, qualified=False, promotable=False)
            with patch.object(audit.report, "evidence", return_value=ev), patch.object(audit, "summarize", return_value=sentinel) as summarize:
                result = audit.report.build_report(root)
            self.assertEqual(result["metrics"]["cache_combination"], sentinel)
            summarize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
