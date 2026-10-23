"""Reject changed controls, cache receipts, missing timing, and mismatched outputs."""

import copy
import unittest

from ditserve.evaluation import workflow_dispatch_refresh_report as audit
from ditserve.evaluation import workflow_dispatch_refresh_study as study
from ditserve.evaluation.workflow_serving_study import digest
from test_workflow_encoding_optimization_report import fixture as encoding_fixture


def fixture():
    original, job, old_config = encoding_fixture("wan_negative_cache", "mixed_negative")
    third, _, _ = encoding_fixture("wan_negative_cache", "mixed_negative", repeat=1)
    config = study.native_config(old_config["phases"][0]["profile"], old_config["phases"][0]["workload"])
    phases = {p["id"]: p for p in original["phases"]}
    extra = next(p for p in third["phases"] if p["id"] == "enabled")
    observed = [phases["disabled"], phases["enabled"], extra]
    for phase, actual in zip(config["phases"], observed):
        actual["id"] = phase["id"]
        cell = actual["cells"][0]
        cell["dispatch_refresh"] = phase["dispatch_refresh"]
        for row in cell["requests"]:
            row["admission_ms"] = row["arrival_ms"]
            for event in row["dispatch_timeline"]:
                event.update(rpc_completed_ms=event["finish_ms"] - .25, harvest_delay_ms=.125,
                             harvested_ms=event["finish_ms"]-.125,
                             timing_scope="host_dispatch_task")
            if row["kind"] == "video":
                row["operation_counts"].update(encode_start=1, encode_poll=2)
                row["encoder_rpc"] = dict(calls=3, total_ms=3., max_ms=1.,
                    timing_scope="host_encoder_round_trip", failed_calls=0,
                    start=dict(start_ms=row["arrival_ms"], finish_ms=row["arrival_ms"]+1, failed=False),
                    ready_poll=dict(start_ms=row["arrival_ms"]+2, finish_ms=row["arrival_ms"]+3, failed=False))
                row.update(encode_submit_ms=row["arrival_ms"]+1, encode_ready_ms=row["arrival_ms"]+3)
    result = dict(original, config=copy.deepcopy(config), phases=observed)
    job.update(configuration_sha256=digest(config), metadata=dict(role="dispatch_refresh_online", repeat=0))
    return result, job, config


class DispatchRefreshReportTests(unittest.TestCase):
    def test_all_three_arms_and_pairings_are_required(self):
        result, job, config = fixture()
        checked = audit.audit_result(result, job, config)
        self.assertTrue(checked["passed"], checked["errors"])
        self.assertEqual(len(checked["exact"]), 3)
        result["phases"].pop()
        self.assertFalse(audit.audit_result(result, job, config)["passed"])

    def test_changed_cache_quantum_or_refresh_is_not_a_controlled_experiment(self):
        for key in ("dispatch_refresh", "quantum_ms", "negative_prompt_cache_entries"):
            result, job, config = fixture()
            phase = config["phases"][-1]
            if key == "dispatch_refresh":
                phase[key] = False
            elif key == "quantum_ms":
                phase["options"][key] = 100
            else:
                phase["worker_overrides"]["wan0"]["options"][key] = 8
            result["config"] = copy.deepcopy(config)
            job["configuration_sha256"] = digest(config)
            self.assertFalse(audit.audit_result(result, job, config)["passed"], key)

    def test_missing_or_inconsistent_timing_fails_closed(self):
        for change in ("missing", "future", "delay", "admission", "actual_mode"):
            result, job, config = fixture()
            cell = result["phases"][-1]["cells"][0]
            row = cell["requests"][0]
            event = row["dispatch_timeline"][0]
            if change == "missing":
                event.pop("rpc_completed_ms")
            elif change == "future":
                event["rpc_completed_ms"] = event["finish_ms"] + 1
            elif change == "delay":
                event["harvest_delay_ms"] = 200
            elif change == "admission":
                row["admission_ms"] = row["start_ms"] + 1
            else:
                cell["dispatch_refresh"] = False
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_candidate_cache_receipt_and_final_hash_are_audited(self):
        for change in ("cache", "output"):
            result, job, config = fixture()
            row = next(r for r in result["phases"][-1]["cells"][0]["requests"] if r["kind"] == "video")
            if change == "cache":
                row["negative_prompt_cache"]["sequence"] += 1
            else:
                row["outputs"][0]["latent"]["sha256"] = "0" * 64
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_elapsed_model_and_rpc_aggregate_must_match_observed_windows(self):
        for change in ("elapsed", "model", "aggregate"):
            result, job, config = fixture()
            cell = result["phases"][-1]["cells"][0]
            if change == "elapsed":
                cell["elapsed_ms"] = 1.
            elif change == "model":
                row = next(r for r in cell["requests"] if r["kind"] == "image")
                event = row["dispatch_timeline"][0]
                event["rpc_completed_ms"] = event["start_ms"]
                event["harvest_delay_ms"] = event["harvested_ms"] - event["start_ms"]
            else:
                row = next(r for r in cell["requests"] if r["kind"] == "video")
                row["encoder_rpc"].update(total_ms=0., max_ms=0.)
            self.assertFalse(audit.audit_result(result, job, config)["passed"], change)

    def test_failed_qualification_cannot_pass_screen(self):
        self.assertFalse(audit.screening_gate(dict(qualified=False, capacity=True))["passed"])


if __name__ == "__main__":
    unittest.main()
