"""Canonical Functional Domain Suite 5: Archive Domain (`test_domain_archive.py`).

Consolidates and verifies:
1. Compact results archive (`ditserve.results-summary.v1`), `finalize`, `verify`,
   `load`, and Markdown report generation (`summary.json` + `YYYYMMDD-<run_id>.md`,
   dependency closure rules, crash recovery, immutability & tamper detection,
   log scalar summaries, mechanism sample aggregation, atomic batch histograms).
2. Tier-1 Lossless Runtime Qualification Gate (`evaluate_tier1_lossless_qualification`,
   `apply_tier1_qualification_gate`, `qualified_goodput_rps` population on real
   lossless GPU runs vs `None` on mock/failed/uncertified-approx runs).
3. Calibration bundle provenance (`exact_uuid` vs `homogeneous_topology_signature`
   preserving `origin_gpu_uuids` without falsifying `gpu_uuids`, rejecting
   non-homogeneous allocations, and `kairos_formal_benchmark.sh` `--slo-scale 1.25`
   tight-SLO contract).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

import ditserve
from ditserve import results
from ditserve.evaluation.common import canonical_hash, seal, verify_seal
import formal_benchmark

_RESULTS_ARCHIVE_SPEC = importlib.util.spec_from_file_location(
    "ditserve.results_archive",
    ROOT / "python" / "ditserve" / "results_archive.py",
)
assert _RESULTS_ARCHIVE_SPEC is not None and _RESULTS_ARCHIVE_SPEC.loader is not None
py_results_archive = importlib.util.module_from_spec(_RESULTS_ARCHIVE_SPEC)
sys.modules["ditserve.results_archive"] = py_results_archive
setattr(ditserve, "results_archive", py_results_archive)
_RESULTS_ARCHIVE_SPEC.loader.exec_module(py_results_archive)


# ==============================================================================
# 1. Compact Results Archive (`ditserve.results-summary.v1`), Finalize & Verify
# ==============================================================================


class TestCompactResultsArchiveContracts:
    """Verifies compact publication (`ditserve.results-summary.v1`), verification, and cleanup boundaries."""

    @staticmethod
    def _base_run() -> dict:
        return {
            "schema": "ditserve.run.v1",
            "experiment": "serving",
            "run_id": "policy-study",
            "status": "completed",
            "config": {
                "workers": {
                    "video": {
                        "backend": "wan",
                        "config": {"model_path": "wan21", "revision": "frozen-revision", "dtype": "float16"},
                    }
                }
            },
            "source": {"core_sha256": "a" * 64, "files": {"large-source-manifest": "b" * 64}},
            "metrics": {"requests": 2, "completed": 1, "failed": 1, "qualified_goodput_rps": None},
            "phases": [
                {
                    "id": "cluster",
                    "status": "completed",
                    "attempt": 0,
                    "qualification": {"scopes": {"performance": {"status": "failed"}}},
                    "cells": [
                        {
                            "cell": 0,
                            "policy": "ditserve",
                            "status": "completed",
                            "gpus": 8,
                            "metrics": {"requests": 2, "latency_p95_ms": 1.2345678901234567},
                            "doorbell": {"rings": 3},
                            "by_kind": {"image": {"attainment": 0.5}},
                            "requests": [{"request_id": "r1", "events": ["huge" * 1000]}],
                        }
                    ],
                }
            ],
            "requests": [
                {"request_id": "r1", "status": "failed", "error": "out of memory"},
                {"request_id": "r2", "events": ["huge" * 1000] * 30},
            ],
            "budget": {
                "active": None,
                "used_gpu_seconds": 5.0,
                "sessions": [{"ended_at": 1793577600, "identity": {"gpu_uuids": ["GPU-1"]}}],
            },
            "failures": [],
            "artifacts": [],
            "retention": {"pending_dependencies": []},
        }

    @staticmethod
    def _add_raw(work: pathlib.Path, run: dict, name: str, dependencies=None, kind: str = "raw") -> pathlib.Path:
        path = work / name
        path.write_bytes(b"raw evidence")
        run["artifacts"].append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": kind,
                "dependencies": dependencies or [],
            }
        )
        return path

    def test_default_summary_publication_produces_two_files_and_strips_raw_arrays(
        self, tmp_path: pathlib.Path
    ) -> None:
        work, output = tmp_path / "work", tmp_path / "results"
        work.mkdir()
        run = self._base_run()
        (work / "run.json").write_text(json.dumps(run), encoding="utf-8")
        original = (work / "run.json").read_bytes()

        with patch.object(results, "_excel", side_effect=AssertionError("no Excel dependency")):
            outcome = results.finalize(work, output)
            assert outcome["status"] == "completed"
            saved = results.load(output)
            assert results.verify(output)["format"] == results.SUMMARY_SCHEMA

        assert {p.name for p in output.iterdir()} == {"summary.json", "20261102-policy-study.md"}
        assert (work / "run.json").read_bytes() == original
        assert (output / "summary.json").stat().st_size < len(original) // 10
        assert "requests" not in saved["run"]
        assert "config" not in saved["run"]
        assert "files" not in saved["run"]["source"]
        assert saved["run"]["config_content_sha256"] == results._digest(run["config"])
        assert saved["run"]["models"][0]["config"]["revision"] == "frozen-revision"
        cell = saved["run"]["phases"][0]["cells"][0]
        assert "requests" not in cell
        assert cell["metrics"]["latency_p95_ms"] == 1.2345678901234567
        assert cell["doorbell"]["rings"] == 3
        assert saved["run"]["failures"][0]["error"] == "out of memory"
        assert saved["run"]["summary_limits"]["raw_re_evaluable"] is False
        assert saved["run"]["metrics"]["qualified_goodput_rps"] is None
        report = pathlib.Path(outcome["report"]).read_text(encoding="utf-8")
        assert "performance=failed" in report
        assert "不能用于续跑或原始重评" in report

    def test_dependency_closure_and_crash_recovery_preserve_cleanup_safety(
        self, tmp_path: pathlib.Path
    ) -> None:
        work, output = tmp_path / "work", tmp_path / "results"
        work.mkdir()
        run = self._base_run()
        closed = self._add_raw(work, run, "closed.npy")
        pending = self._add_raw(work, run, "pending.npy", ["quality-evaluation"])
        (work / "run.json").write_text(json.dumps(run), encoding="utf-8")

        assert results.finalize(work, output)["status"] == "deferred"
        assert not closed.exists()
        assert pending.exists()

        # Closing dependency explicitly allows cleanup completion
        run["artifacts"][1]["dependencies"] = [{"closed": True, "reason": "quality evaluation saved"}]
        (work / "run.json").write_text(json.dumps(run), encoding="utf-8")
        assert results.finalize(work, output)["status"] == "completed"
        assert not pending.exists()
        saved = results.load(output)
        assert len(saved["finalization"]["deleted"]) == 2

        # Crash after unlink recovers from summary-bound intent
        work2, output2 = tmp_path / "work2", tmp_path / "results2"
        work2.mkdir()
        run2 = self._base_run()
        raw2 = self._add_raw(work2, run2, "sample.npy")
        (work2 / "run.json").write_text(json.dumps(run2), encoding="utf-8")
        original_save = results._save

        def _crash_after_unlink(w, journal):
            if journal.get("deleted"):
                raise KeyboardInterrupt("crash after unlink")
            original_save(w, journal)

        with patch.object(results, "_save", _crash_after_unlink), pytest.raises(KeyboardInterrupt):
            results.finalize(work2, output2)
        assert not raw2.exists()
        assert results.finalize(work2, output2)["status"] == "completed"
        assert (
            results.load(output2)["finalization"]["deleted"][0]["status"]
            == "absent_after_recorded_delete_intent"
        )

    def test_immutability_tamper_detection_and_active_run_protection(self, tmp_path: pathlib.Path) -> None:
        work, output = tmp_path / "work", tmp_path / "results"
        work.mkdir()
        run = self._base_run()
        run["status"] = "running"
        (work / "run.json").write_text(json.dumps(run), encoding="utf-8")
        assert results.finalize(work, output)["status"] == "protected"
        assert not output.exists()

        run["status"] = "completed"
        (work / "run.json").write_text(json.dumps(run), encoding="utf-8")
        assert results.finalize(work, output)["status"] == "completed"
        before = {p.name: p.read_bytes() for p in output.iterdir()}

        # Tampering with summary.json or Markdown report is caught by verify()
        summary_path = output / "summary.json"
        val = json.loads(summary_path.read_text(encoding="utf-8"))
        val["run"]["metrics"]["completed"] = 100
        summary_path.write_text(json.dumps(val), encoding="utf-8")
        with pytest.raises(ValueError, match="checksum"):
            results.verify(output)

        summary_path.write_bytes(before["summary.json"])
        md_path = output / "20261102-policy-study.md"
        md_path.write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="report differs"):
            results.verify(output)

    def test_mechanism_samples_atomic_batch_and_resume_distributions(self, tmp_path: pathlib.Path) -> None:
        work, output = tmp_path / "work", tmp_path / "results"
        work.mkdir()
        run = self._base_run()
        cell = run["phases"][0]["cells"][0]
        cell.update(
            elapsed_ms=2000,
            gpus=8,
            image_batch=8,
            guard_telemetry={"status": "exclusive_observed", "dropped_events": 0, "events": [{"large": "log"}]},
            requests=[
                {
                    "kind": "video",
                    "request_id": "v1",
                    "status": "completed",
                    "finish_ms": 10,
                    "deadline_ms": 20,
                    "transfers": 2,
                    "resumes": 2,
                    "resume_extra_ms": 6.0,
                    "preemption_reason": "block_preempt",
                    "preemption_stop_samples": [{"stop_ms": 1}, {"stop_ms": 5}],
                },
                {"kind": "image", "request_id": "a", "batch_size": 2, "batch_request_ids": ["a", "b"]},
                {"kind": "image", "request_id": "b", "batch_size": 2, "batch_request_ids": ["b", "a"]},
                {"kind": "image", "request_id": "c", "batch_size": 1, "batch_request_ids": ["c"]},
            ],
        )
        (work / "run.json").write_text(json.dumps(run), encoding="utf-8")
        assert results.finalize(work, output)["status"] == "completed"
        saved_cell = results.load(output)["run"]["phases"][0]["cells"][0]
        assert saved_cell["guard_telemetry"] == {"status": "exclusive_observed", "dropped_events": 0}
        exec_sum = saved_cell["execution_summary"]
        assert exec_sum["observed_transfers"] == 2
        assert exec_sum["preempting_request_count"] == 1
        assert exec_sum["preemption_stop_ms"]["p50"] == 3
        assert exec_sum["atomic_batch"]["batch_size_histogram"] == {"1": 1, "2": 1}
        assert exec_sum["atomic_batch"]["verified_batches"] == 2
        assert exec_sum["resumes"]["total"] == 2


# ==============================================================================
# 2. Tier-1 Lossless Runtime Qualification Gate (`ditserve.results_archive`)
# ==============================================================================


class TestTier1LosslessQualificationGate:
    """Verifies automatic Tier-1 lossless qualification gate on real GPU runs vs mock/failed/approx."""

    @staticmethod
    def _make_real_gpu_serving_run(
        *,
        mock_execution: bool = False,
        failed: int = 0,
        approx_fold_steps: int = 0,
    ) -> dict:
        return {
            "schema": "ditserve.run.v1",
            "experiment": "20261106-formal-wan22-test-anchor-main-21a1ca3-r1",
            "run_id": "20261106-formal-wan22-test-anchor-main-21a1ca3-r1",
            "status": "completed",
            "measurement_scope": "supervised gpu experiment",
            "mock_execution": mock_execution,
            "failed": failed,
            "gpu_uuids": [f"GPU-00000000-0000-0000-0000-00000000000{idx}" for idx in range(8)],
            "metrics": {
                "requests": 100,
                "completed": 100 - failed,
                "failed": failed,
                "ontime": 93 if not failed else 92,
                "slo_attainment": 0.93,
                "on_time_goodput_rps": 0.0917273,
                "goodput_rps": 0.0977234,
                "qualified_goodput_rps": None,
                "qualification_status": "unverified",
                "model_ms": 845120.0,
            },
            "config": {
                "study": "formal-wan22-test-anchor-main",
                "mock_execution": mock_execution,
                "workers": {
                    "video": {"backend": "mock" if mock_execution else "wan22"},
                    "image": {"backend": "mock" if mock_execution else "sd35"},
                },
                "phases": [
                    {
                        "name": "formal-mixed",
                        "type": "cluster",
                        "policy": "ditserve",
                        "options": {
                            "cfg_fold_steps": approx_fold_steps,
                            "cfg_fold": approx_fold_steps > 0,
                        },
                    }
                ],
            },
            "phases": [
                {
                    "name": "formal-mixed",
                    "kind": "cluster",
                    "status": "completed" if failed == 0 else "failed",
                    "qualification": {
                        "qualified": False,
                        "status": "unverified",
                        "numeric": "unverified",
                        "semantic": "unverified",
                        "performance": "unverified",
                    },
                    "metrics": {
                        "requests": 100,
                        "completed": 100 - failed,
                        "failed": failed,
                        "ontime": 93,
                        "slo_attainment": 0.93,
                        "on_time_goodput_rps": 0.0917273,
                        "goodput_rps": 0.0977234,
                        "qualified_goodput_rps": None,
                        "qualification_status": "unverified",
                    },
                    "cells": [
                        {
                            "id": "cell-ditserve",
                            "policy": "ditserve",
                            "status": "completed" if failed == 0 else "failed",
                            "elapsed_ms": 959365.0,
                            "options": {"cfg_fold": approx_fold_steps > 0},
                            "metrics": {
                                "requests": 100,
                                "completed": 100 - failed,
                                "failed": failed,
                                "ontime": 93,
                                "slo_attainment": 0.93,
                                "on_time_goodput_rps": 0.0917273,
                                "goodput_rps": 0.0977234,
                                "qualified_goodput_rps": None,
                                "qualification_status": "unverified",
                            },
                            "by_kind": {
                                "image": {
                                    "requests": 50,
                                    "completed": 50,
                                    "failed": 0,
                                    "ontime": 49,
                                    "slo_attainment": 0.98,
                                    "on_time_goodput_rps": 0.051075,
                                    "qualified_goodput_rps": None,
                                },
                                "video": {
                                    "requests": 50,
                                    "completed": 50 - failed,
                                    "failed": failed,
                                    "ontime": 44,
                                    "slo_attainment": 0.88,
                                    "on_time_goodput_rps": 0.0458636,
                                    "qualified_goodput_rps": None,
                                },
                            },
                        },
                        {
                            "id": "cell-sbp-esp",
                            "policy": "sbp-esp",
                            "status": "completed",
                            "elapsed_ms": 996473.0,
                            "options": {},
                            "metrics": {
                                "requests": 100,
                                "completed": 100,
                                "failed": 0,
                                "ontime": 90,
                                "slo_attainment": 0.90,
                                "on_time_goodput_rps": 0.0912309,
                                "goodput_rps": 0.100354,
                                "qualified_goodput_rps": None,
                                "qualification_status": "unverified",
                            },
                        },
                    ],
                }
            ],
        }

    def test_tier1_lossless_qualification_gate_closure(self) -> None:
        # Case 1: Real GPU lossless run -> automatically verified & qualified_goodput_rps populated
        real_run = self._make_real_gpu_serving_run(mock_execution=False, failed=0, approx_fold_steps=0)
        summary = py_results_archive._summary_run(real_run)

        assert summary["qualification"]["qualified"] is True
        assert summary["qualification"]["numeric"] == "verified_lossless_exact"
        assert summary["qualification"]["semantic"] == "verified_bitwise_invariant"
        assert summary["qualification"]["performance"] == "verified_real_gpu"
        assert (
            summary["qualification_status"]
            == "numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu"
        )
        assert summary["qualified_goodput_rps"] == pytest.approx(0.0917273)

        cells = summary["phases"][0]["cells"]
        assert cells[0]["policy"] == "ditserve"
        assert cells[0]["metrics"]["qualified_goodput_rps"] == pytest.approx(0.0917273)
        assert cells[0]["by_kind"]["image"]["qualified_goodput_rps"] == pytest.approx(0.051075)
        assert cells[0]["by_kind"]["video"]["qualified_goodput_rps"] == pytest.approx(0.0458636)
        assert cells[1]["policy"] == "sbp-esp"
        assert cells[1]["metrics"]["qualified_goodput_rps"] == pytest.approx(0.0912309)

        md_text = py_results_archive._summary_report(summary)
        assert "numeric=verified_lossless_exact" in md_text
        assert "0.0917" in md_text

        # Case 2: Mock execution -> remains unverified and qualified_goodput_rps is None
        mock_run = self._make_real_gpu_serving_run(mock_execution=True, failed=0, approx_fold_steps=0)
        mock_gated = py_results_archive.apply_tier1_qualification_gate(mock_run)
        assert mock_gated["qualification"]["qualified"] is False
        assert mock_gated["qualification_status"] == "unverified"
        mock_summary = py_results_archive._summary_run(mock_run)
        assert mock_summary["phases"][0]["qualification"]["qualified"] is False
        assert mock_summary["phases"][0]["cells"][0]["metrics"]["qualified_goodput_rps"] is None
        assert "未验证" in py_results_archive._summary_report(mock_summary)

        # Case 3: Failed requests > 0 -> remains unverified and qualified_goodput_rps is None
        failed_run = self._make_real_gpu_serving_run(mock_execution=False, failed=1, approx_fold_steps=0)
        failed_gated = py_results_archive.apply_tier1_qualification_gate(failed_run)
        assert failed_gated["qualification"]["qualified"] is False
        failed_summary = py_results_archive._summary_run(failed_run)
        assert failed_summary["phases"][0]["qualification"]["qualified"] is False
        assert failed_summary["phases"][0]["cells"][0]["metrics"]["qualified_goodput_rps"] is None

        # Case 4: Approximate flag without external quality certificate -> remains unverified
        approx_run = self._make_real_gpu_serving_run(mock_execution=False, failed=0, approx_fold_steps=10)
        approx_gated = py_results_archive.apply_tier1_qualification_gate(approx_run)
        assert approx_gated["qualification"]["qualified"] is False
        approx_summary = py_results_archive._summary_run(approx_run)
        assert approx_summary["phases"][0]["qualification"]["qualified"] is False
        assert approx_summary["phases"][0]["cells"][0]["metrics"]["qualified_goodput_rps"] is None


# ==============================================================================
# 3. Calibration Bundle Provenance & Kairos Formal Benchmark Tight-SLO Contract
# ==============================================================================


class TestCalibrationBundleProvenanceAndBenchmarkScript:
    """Verifies `exact_uuid` vs `homogeneous_topology_signature` provenance and tight-SLO script contract."""

    @staticmethod
    def _make_sealed_calibration_bundle(
        proto: dict,
        model: str,
        manifests: dict,
        origin_uuids: list[str],
    ) -> dict:
        ids = [
            sid
            for sid, shape in proto["shapes"].items()
            if shape["scope"] == "formal" and shape["model"] in (model, "sd35")
        ]
        classes = {}
        for sid in ids:
            classes[sid] = {
                "spec": formal_benchmark.shape_spec(proto, sid),
                "source": {
                    "gpu_uuids": list(origin_uuids),
                    "fit": {"gpu_uuids": list(origin_uuids)},
                },
            }
        raw = {
            "schema": formal_benchmark.BUNDLE_SCHEMA,
            "protocol_id": proto.get("protocol_id") or proto["id"],
            "protocol_sha256": canonical_hash(proto),
            "model": model,
            "model_manifests": manifests,
            "gpu_uuids": list(origin_uuids),
            "classes": classes,
        }
        return seal(raw)

    def test_kairos_formal_benchmark_script_provenance_and_tight_slo(self) -> None:
        script_path = ROOT / "benchmarks" / "kairos_formal_benchmark.sh"
        text = script_path.read_text(encoding="utf-8")

        assert "for exact_uuid in (True, False):" not in text
        assert "homogeneous_topology_signature" in text
        assert "origin_gpu_uuids" in text
        assert "calibration_match_mode" in text
        assert "--allow-cross-uuid" in text.lower()

        assert "--slo-scale 1.25" in text
        assert 'flags="${flags/--slo-scale 1.0/--slo-scale 1.25}"' in text
        protocol = json.loads((ROOT / "configs" / "benchmarks" / "formal-mixed-v2.json").read_text(encoding="utf-8"))
        assert protocol["id"] == "formal-mixed-v2"

        proc = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr

    def test_calibration_bundle_exact_vs_homogeneous_signature_validation(self, tmp_path: pathlib.Path) -> None:
        proto_path = ROOT / "configs" / "benchmarks" / "formal-mixed-v2.json"
        proto = formal_benchmark.protocol(proto_path)
        manifests = {"wan22": "/shared/data/DITSERVE/models/wan22", "sd35": "/shared/data/DITSERVE/models/sd35"}
        origin_uuids = [f"GPU-00000000-0000-0000-0000-00000000010{i}" for i in range(8)]
        alloc2_uuids = [f"GPU-00000000-0000-0000-0000-00000000020{i}" for i in range(8)]

        cand_bundle = self._make_sealed_calibration_bundle(proto, "wan22", manifests, origin_uuids)
        cand_path = tmp_path / "candidate-bundle.json"
        cand_path.write_text(json.dumps(cand_bundle, indent=2), encoding="utf-8")
        verify_seal(cand_bundle)

        script_text = (ROOT / "benchmarks" / "kairos_formal_benchmark.sh").read_text(encoding="utf-8")
        start_marker = (
            'try_reuse_calibration_bundle() {\n  python3 - "$benchmarks" "$protocol" "$model" '
            '"$DITSERVE_MODEL_MANIFESTS" "$RUN_DIR/bundle.json" "$DITSERVE_REUSE_BUNDLES" '
            '"${RUN_DIR}/runtime-facts.json" <<\'PY\'\n'
        )
        end_marker = "\nPY\n}"
        start_idx = script_text.index(start_marker) + len(start_marker)
        end_idx = script_text.index(end_marker, start_idx)
        py_snippet = script_text[start_idx:end_idx]

        ns: dict = {}
        func_defs = py_snippet.split("\np = protocol(")[0]
        exec(func_defs, ns)

        # 1. Exact UUID match -> calibration_match_mode == "exact_uuid"
        runtime_facts_8a100 = tmp_path / "runtime-facts-8a100.json"
        runtime_facts_8a100.write_text(
            json.dumps({"gpus": [{"name": "NVIDIA A100-PCIE-40GB"} for _ in range(8)]}),
            encoding="utf-8",
        )
        out_exact = tmp_path / "bundle-exact.json"
        rc_exact = ns["try_reuse_calibration_bundle"](
            proto, "wan22", manifests, out_exact, [cand_path], origin_uuids, runtime_facts_8a100
        )
        assert rc_exact == 0
        reused_exact = json.loads(out_exact.read_text(encoding="utf-8"))
        verify_seal(reused_exact)
        assert reused_exact["calibration_match_mode"] == "exact_uuid"
        assert reused_exact["origin_gpu_uuids"] == origin_uuids
        assert reused_exact["active_gpu_uuids"] == origin_uuids
        assert reused_exact["gpu_uuids"] == origin_uuids

        # 2. Cross-allocation on homogeneous 8x NVIDIA A100-PCIE-40GB -> preserves origin_gpu_uuids
        out_homog = tmp_path / "bundle-homog.json"
        rc_homog = ns["try_reuse_calibration_bundle"](
            proto, "wan22", manifests, out_homog, [cand_path], alloc2_uuids, runtime_facts_8a100
        )
        assert rc_homog == 0
        reused_homog = json.loads(out_homog.read_text(encoding="utf-8"))
        verify_seal(reused_homog)
        assert reused_homog["calibration_match_mode"] == "homogeneous_topology_signature"
        assert reused_homog["origin_gpu_uuids"] == origin_uuids
        assert reused_homog["active_gpu_uuids"] == alloc2_uuids
        assert reused_homog["gpu_uuids"] == origin_uuids
        for cls_entry in reused_homog["classes"].values():
            assert cls_entry["source"]["origin_gpu_uuids"] == origin_uuids
            assert cls_entry["source"]["active_gpu_uuids"] == alloc2_uuids
            assert cls_entry["source"]["calibration_match_mode"] == "homogeneous_topology_signature"

        # 3. Non-homogeneous or non-8-GPU allocation must be rejected when UUIDs differ
        runtime_facts_other = tmp_path / "runtime-facts-other.json"
        runtime_facts_other.write_text(
            json.dumps({"gpus": [{"name": "NVIDIA V100-SXM2-32GB"} for _ in range(8)]}),
            encoding="utf-8",
        )
        out_reject = tmp_path / "bundle-reject.json"
        rc_reject = ns["try_reuse_calibration_bundle"](
            proto, "wan22", manifests, out_reject, [cand_path], alloc2_uuids, runtime_facts_other
        )
        assert rc_reject == 1
        assert not out_reject.exists()
