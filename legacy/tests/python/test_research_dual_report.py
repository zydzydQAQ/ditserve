import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest
from ditserve.research.dual_matrix import seal
from ditserve.research.dual_report import budget_summary, evaluate_hypotheses, make_figures, report
from ditserve.research.retention import apply_plan, build_plan, verify_plan
from test_research_dual_matrix import trajectory


def quality(loss=.01, *, candidate=None, reference=None):
    result = dict(status="ok", finite=True, range_valid=True, lpips_per_frame=[loss] * 17,
                  frame_count=17, lpips_mean=loss)
    if candidate is not None and reference is not None:
        result.update(candidate_output_sha256=candidate["output_sha256"], reference_output_sha256=reference["output_sha256"],
                      candidate_identity_sha256=digest(candidate["identity"]), reference_identity_sha256=digest(reference["identity"]))
    return result


def config():
    return dict(dev_seeds=[101], holdout_seeds=[201], prompts=[dict(id="kettle", text="k", split="dev"),
                                                            dict(id="tram", text="t", split="holdout")])


def correctness_rows():
    return [dict(type="correctness", case_id="local-" + role, status="passed", performance_valid=True,
                 step_equivalent=True) for role in ("a100", "v100")] + [
                 dict(type="correctness", case_id="migration", status="passed", performance_valid=True, exact_omit_equal=True)]


def quality_rows():
    rows = correctness_rows()
    for split, prompt, text, seed in (("dev", "kettle", "k", 101), ("holdout", "tram", "t", 201)):
        item = dict(split=split, prompt_id=prompt, prompt=text, seed=seed)
        candidate = trajectory(item, "v100", "t008", "cold_migrate", context="ctx")
        common = trajectory(item, "a100", "off", "stay", context="ctx")
        path = trajectory(item, "v100", "off", "full_migrate", context="ctx")
        rows.append(dict(type="migration_quality", status="completed", performance_valid=True, action="cold_migrate",
            context_sha256="ctx", split=split, prompt_id=prompt, seed=seed, candidate=candidate,
            common_reference=common, path_reference=path, path_reference_quality=quality(candidate=path, reference=common),
            canonical_quality=quality(candidate=candidate, reference=common), path_quality=quality(candidate=candidate, reference=path)))
    return rows


def pair_rows(condition="idle", gains=(.1, .1)):
    result = []
    for index, gain in enumerate(gains):
        result.append(dict(type="mechanism", status="completed", case_id=condition + str(index),
                           performance_valid=True, quality_passed=True, all_requests_decoded=True,
                           condition=condition, queue_coverage=True, context_sha256="ctx", order=["stay", "full_migrate", "cold_migrate"] if not index
                           else ["cold_migrate", "full_migrate", "stay"],
                           costs_ms=dict(stay=100, full_migrate=100 * (1 - gain), cold_migrate=100 * (1 - gain))))
    return result


class DualReportTests(unittest.TestCase):
    def test_budget_keeps_live_and_settled_separate(self):
        value = budget_summary(dict(used_gpu_seconds=100, limit_gpu_seconds=28800,
                                    active=dict(started_at=10, gpu_count=2)), now=15)
        self.assertEqual(value["settled_gpu_seconds"], 100)
        self.assertEqual(value["active_estimated_gpu_seconds"], 10)
        self.assertEqual(value["estimated_remaining_gpu_seconds"], 28690)

    def test_unmeasured_is_insufficient_not_negative(self):
        values = evaluate_hypotheses([], config())
        self.assertEqual(values["migration"]["status"], "insufficient_evidence")
        self.assertEqual(values["memory"]["status"], "insufficient_evidence")
        self.assertEqual(values["cache"]["a100"]["status"], "insufficient_evidence")
        self.assertEqual(values["sp"]["status"], "deferred")

    def test_two_orders_quality_and_full_cost_gate(self):
        rows = quality_rows() + pair_rows()
        values = evaluate_hypotheses(rows, config())
        self.assertEqual(values["migration"]["status"], "supported")
        self.assertEqual(values["cache_joint_control"]["status"], "insufficient_evidence")
        rows[-1]["performance_valid"] = False
        self.assertEqual(evaluate_hypotheses(rows, config())["migration"]["status"], "insufficient_evidence")
        rows = quality_rows() + pair_rows(gains=(.1, .02)) + pair_rows("busy", gains=(.03, .02))
        self.assertEqual(evaluate_hypotheses(rows, config())["migration"]["status"], "falsified")

    def test_busy_pair_requires_explicit_queue_coverage_in_both_orders(self):
        for missing_value in (None, False, 1):
            rows = quality_rows() + pair_rows("busy")
            if missing_value is None:
                rows[-1].pop("queue_coverage")
            else:
                rows[-1]["queue_coverage"] = missing_value
            result = evaluate_hypotheses(rows, config())["migration"]
            self.assertEqual(result["status"], "insufficient_evidence")
            busy = next(c for c in result["conditions"] if c["condition"] == "busy")
            self.assertEqual(busy["status"], "insufficient_evidence")
            self.assertIn("real queue wait", busy["reason"])
        self.assertEqual(evaluate_hypotheses(quality_rows() + pair_rows("busy"), config())["migration"]["status"], "supported")

    def test_failed_full_trajectory_quality_blocks_performance(self):
        rows = quality_rows() + pair_rows()
        rows[4]["path_quality"] = quality(.08)
        self.assertEqual(evaluate_hypotheses(rows, config())["migration"]["status"], "falsified")

    def test_capacity_requires_real_admission_and_all_quality(self):
        keep = dict(type="capacity", case_id="keep", status="completed", performance_valid=True,
                    paused_count=1, action="keep", urgent_capacity_failed=True)
        migrated = dict(type="capacity", case_id="release", status="completed", performance_valid=True,
                    paused_count=1, action="offload", urgent_capacity_failed=False, quality_passed=True,
                    artificially_filled_memory=False, context_sha256="ctx",
                    original_requests=[dict(status="completed", quality=quality())],
                    urgent=dict(status="completed", quality=quality()))
        baseline = dict(type="calibration", device_role="a100", performance_valid=True,
                        context_sha256="ctx", reference=dict(status="completed"))
        self.assertEqual(evaluate_hypotheses([keep, migrated], config())["memory"]["status"], "insufficient_evidence")
        self.assertEqual(evaluate_hypotheses(correctness_rows() + [keep, migrated, baseline], config())["memory"]["status"], "supported")
        migrated["artificially_filled_memory"] = True
        self.assertEqual(evaluate_hypotheses(correctness_rows() + [keep, migrated, baseline], config())["memory"]["status"], "insufficient_evidence")

    def test_report_accepts_declared_prune_but_rejects_unexplained_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            root.mkdir()
            ledger = root / "gpu-budget.json"
            atomic_json(ledger, dict(active=None, used_gpu_seconds=200, limit_gpu_seconds=28800))
            frozen = dict(sha256="p", source_sha256="s", ledger=str(ledger), config=config())
            atomic_json(root / "protocol.json", frozen)
            video = root / "video.npy"
            video.write_bytes(b"historical raw video")
            video_hash = file_digest(video)
            row = seal(dict(type="probe", case_id="cost", unit_id="u", protocol_sha256="p", source_sha256="s",
                            run=dict(output_path="video.npy", output_sha256=video_hash)))
            (root / "records.jsonl").write_text(json.dumps(row) + "\n")
            (root / "units").mkdir()
            atomic_json(root / "units/u.json", seal(dict(status="completed", performance_valid=True,
                        protocol_sha256="p", source_sha256="s", records=[row["sha256"]])))
            case = dict(series="s", run=str(root), ended_at=1, case_id="cost", kind="probe", split="unknown",
                        policy=None, selected_policy=None, quality=None, failed=False, verified=True, evaluated=True,
                        artifacts=[dict(path=str(video), sha256=video_hash)], reference_sha256s=[])
            plan = build_plan(root, Path(temp) / "key", catalog=dict(cases=[case], sources=[]))
            apply_plan(plan, verify_plan(plan))
            with patch("ditserve.research.dual_report.verify_protocol", return_value=frozen), \
                    patch("ditserve.research.dual_report.make_figures", return_value=dict(status="unavailable", paths=[])):
                result = report(root)
                self.assertEqual(result["evidence_availability"]["retained_metrics_only"], 1)
                self.assertTrue((root / "report.md").is_file())
                (root / "retention-index.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    report(root)

    def test_four_figure_groups_label_missing_data(self):
        with tempfile.TemporaryDirectory() as temp:
            figures = make_figures(Path(temp), [])
            if figures["status"] == "unavailable":
                self.skipTest(figures["reason"])
            self.assertEqual(len(figures["paths"]), 4)
            self.assertTrue(all(not f["measured"] for f in figures["paths"]))
            for figure in figures["paths"]:
                self.assertIn("Insufficient completed-unit evidence", (Path(temp) / figure["path"]).read_text())

    def test_report_refuses_active_session_before_archive_or_array_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "gpu-budget.json"
            atomic_json(ledger, dict(active=dict(pid=123)))
            atomic_json(root / "protocol.json", dict(ledger=str(ledger)))
            with patch("ditserve.research.dual_report.verify_protocol", side_effect=AssertionError("no hashing")):
                with self.assertRaisesRegex(RuntimeError, "unsettled"):
                    report(root)


if __name__ == "__main__":
    unittest.main()
