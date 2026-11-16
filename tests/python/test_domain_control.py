"""Canonical Functional Domain Suite 1: Control Domain (`test_domain_control.py`).

Consolidates and verifies:
1. C++ controller policy catalog & traits (`ditserve`, `ditserve-approx`, `xdit`,
   `tetriserve`, `sbp-esp`, `edf-bestfit`, factorized baselines `tetriserve-cfg2`,
   `sbp-esp-cfg2`, `edf-bestfit-cfg2`, and single-factor ablations `abl-no-block`,
   `abl-no-prune`, `abl-no-mindeg`, `abl-first-fit`, `abl-no-decode-shrink`,
   `abl-no-img-priority`).
2. Tight-SLO (`1.0x-1.25x`) doorbell grace & proactive (`>65%` slack) block
   preemption separation between `ditserve`, `abl-no-block`, and `block-cfg2`,
   plus `abl-no-decode-shrink` and `abl-no-img-priority` behavioral separation.
3. 3-parameter identifiable cost model (`cost_model_variants.py`: SVD condition
   number kappa < 100 vs 4-param kappa > 10^15, 0% ranking regret, LOFO isolation,
   `fit_native_profile.py` parity) & `topology_phase_diagram.py` (PCIe 4.0 / 5.0 /
   NVLink crossover L*, 5 video shapes).
4. Budget ledger (`ditserve.budget.v1`), study runner (`ditserve.study.v1`),
   and CLI dispatch contracts.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_DIR = pathlib.Path(os.environ.get("DITSERVE_BUILD_DIR", ROOT / ".cache/build"))
DITSERVE_BIN = pathlib.Path(os.environ.get("DITSERVE_BIN", BUILD_DIR / "ditserve"))
SIMULATION_TESTS_BIN = pathlib.Path(
    os.environ.get("DITSERVE_SIMULATION_TESTS", BUILD_DIR / "benchmarks/simulation-tests")
)

import cost_model_variants as cmv
from ditserve import api, results
from ditserve.runtime import PACKAGE_ROOT
from ditserve.runtime.client import CoreBridge, CoreError
import fit_native_profile as fnp
import topology_phase_diagram as tpd


def _pcie_profile() -> dict:
    forms = {
        "sp1": {"step_ms": 1147.0},
        "cfg2": {"step_ms": 583.4},
        "sp2": {"step_ms": 1196.9},
        "cfg2sp2": {"step_ms": 790.6},
        "sp4": {"step_ms": 1091.6},
        "cfg2sp4": {"step_ms": 567.1},
    }
    return {
        "provenance": "measured",
        "classes": {
            "video": {
                "kind": "video",
                "steps": 10,
                "blocks_per_step": 40,
                "prepare_ms": 20.0,
                "encode_ms": 0.0,
                "step_compute_ms": 1147.0,
                "comm_mb_per_step": 10351.0,
                "collectives_per_step": 240,
                "finish_ms": 1840.0,
                "resume_ms": 15.0,
                "forms": forms,
            },
            "image": {
                "kind": "image",
                "service_ms": 2500.0,
            },
        },
        "links": {
            "2": {"gbps": 3.75, "latency_ms": 0.5},
            "4": {"gbps": 3.75, "latency_ms": 0.5},
            "8": {"gbps": 3.75, "latency_ms": 0.5},
        },
        "sp_efficiency": 1.0,
        "resize_ms": 15.0,
        "quantum_overhead_ms": 1.5,
    }


def _rpc(command: str, payload: dict) -> dict:
    proc = subprocess.run(
        [str(DITSERVE_BIN), "rpc", command],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _rpc_simulate(payload: dict) -> dict:
    return _rpc("simulate", payload)


# ==============================================================================
# 1. C++ Controller Policy Catalog, Traits & Ablation / Preemption Separation
# ==============================================================================


class TestControllerPolicyCatalogAndSeparation:
    """Verifies the C++ policy catalog, factorized baselines, ablations, and tight-SLO preemption."""

    def test_policy_catalog_registers_all_canonical_baselines_and_ablations(self) -> None:
        proc = subprocess.run(
            [str(SIMULATION_TESTS_BIN), "--policy-catalog"],
            text=True,
            capture_output=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        by_id = {row["id"]: row for row in data["catalog"]}

        required_policies = (
            "ditserve",
            "ditserve-approx",
            "xdit-sp1",
            "xdit-cfg2",
            "tetriserve",
            "sbp-esp",
            "edf-bestfit",
            "tetriserve-cfg2",
            "sbp-esp-cfg2",
            "edf-bestfit-cfg2",
            "abl-no-block",
            "abl-no-prune",
            "abl-no-mindeg",
            "abl-first-fit",
            "abl-no-decode-shrink",
            "abl-no-img-priority",
            "abl-no-cb",
            "stair-base",
            "stair-prune",
            "stair-pairfit",
        )
        for pid in required_policies:
            assert pid in by_id, f"missing policy {pid} in policy_catalog()"

        # DitServe vs DitServe-Approx
        assert by_id["ditserve"]["blocks"] is True
        assert by_id["ditserve"]["elastic"] is True
        assert by_id["ditserve"]["prune_dominated"] is True
        assert by_id["ditserve"]["min_video_degree"] == 2
        assert by_id["ditserve"]["image_placement"] == "pair_fit"
        assert by_id["ditserve"]["feasible_img_priority"] is True
        assert by_id["ditserve"]["hold_diffuse"] is False
        assert by_id["ditserve"]["continuous_batch"] is True
        assert by_id["ditserve"]["sequence_packing"] is True
        assert by_id["ditserve"]["bubble_free"] is True
        assert by_id["ditserve-approx"]["blocks"] is True
        assert by_id["ditserve-approx"]["elastic"] is True
        assert by_id["ditserve-approx"]["approximate"] is True
        assert by_id["ditserve-approx"]["continuous_batch"] is True
        assert by_id["ditserve-approx"]["sequence_packing"] is True
        assert by_id["ditserve-approx"]["bubble_free"] is True

        # Ablation w/o continuous batching and external baselines keep continuous_batch=False
        assert by_id["abl-no-cb"]["continuous_batch"] is False
        assert by_id["abl-no-cb"]["sequence_packing"] is False
        assert by_id["abl-no-cb"]["bubble_free"] is False
        assert by_id["abl-no-cb"]["blocks"] is True
        assert by_id["abl-no-cb"]["prune_dominated"] is True
        for baseline_id in (
            "xdit-sp1",
            "tetriserve",
            "sbp-esp",
            "edf-bestfit",
            "edf-cfg2",
            "block-cfg2",
        ):
            assert by_id[baseline_id]["continuous_batch"] is False
            assert by_id[baseline_id]["sequence_packing"] is False
            assert by_id[baseline_id]["bubble_free"] is False

        # Factorized CFG2 baselines vs SP-only counterparts
        assert by_id["tetriserve-cfg2"]["allow_cfg_forms"] is True
        assert by_id["tetriserve-cfg2"]["rounds"] is True
        assert by_id["tetriserve-cfg2"]["deadline_dp"] is True
        assert by_id["tetriserve"]["allow_cfg_forms"] is False

        assert by_id["sbp-esp-cfg2"]["allow_cfg_forms"] is True
        assert by_id["sbp-esp-cfg2"]["deadline_dp"] is True
        assert by_id["sbp-esp-cfg2"]["image_batching"] is True
        assert by_id["sbp-esp"]["allow_cfg_forms"] is False

        assert by_id["edf-bestfit-cfg2"]["allow_cfg_forms"] is True
        assert by_id["edf-bestfit-cfg2"]["best_fit"] is True
        assert by_id["edf-bestfit-cfg2"]["demote_late"] is False
        assert by_id["edf-bestfit"]["allow_cfg_forms"] is False

        # Single-factor ablations flip exactly one mechanism relative to ditserve
        assert by_id["abl-no-block"]["blocks"] is False
        assert by_id["abl-no-block"]["prune_dominated"] is True
        assert by_id["abl-no-block"]["min_video_degree"] == 2
        assert by_id["abl-no-block"]["image_placement"] == "pair_fit"

        assert by_id["abl-no-prune"]["blocks"] is True
        assert by_id["abl-no-prune"]["prune_dominated"] is False
        assert by_id["abl-no-prune"]["min_video_degree"] == 2

        assert by_id["abl-no-mindeg"]["blocks"] is True
        assert by_id["abl-no-mindeg"]["prune_dominated"] is True
        assert by_id["abl-no-mindeg"]["min_video_degree"] == 1

        assert by_id["abl-first-fit"]["blocks"] is True
        assert by_id["abl-first-fit"]["image_placement"] == "first"

        assert by_id["abl-no-decode-shrink"]["hold_diffuse"] is True
        assert by_id["abl-no-decode-shrink"]["blocks"] is True
        assert by_id["abl-no-decode-shrink"]["feasible_img_priority"] is True

        assert by_id["abl-no-img-priority"]["feasible_img_priority"] is False
        assert by_id["abl-no-img-priority"]["blocks"] is True
        assert by_id["abl-no-img-priority"]["hold_diffuse"] is False

        # Cumulative 5-rung design staircase forms a strict boolean feature lattice
        stair_rungs = ("stair-base", "stair-prune", "stair-pairfit", "abl-no-cb", "ditserve")
        stair_flags = (
            "prune_dominated",
            "feasible_img_priority",
            "blocks",
            "continuous_batch",
            "sequence_packing",
            "bubble_free",
        )
        for idx in range(len(stair_rungs) - 1):
            cur = by_id[stair_rungs[idx]]
            nxt = by_id[stair_rungs[idx + 1]]
            for flag in stair_flags:
                assert int(bool(cur[flag])) <= int(bool(nxt[flag])), f"{flag} regressed from {stair_rungs[idx]} to {stair_rungs[idx+1]}"
            assert int(cur["min_video_degree"]) <= int(nxt["min_video_degree"])
        assert by_id["stair-base"]["allow_cfg_forms"] is False
        assert by_id["stair-base"]["image_placement"] == "first"
        assert by_id["stair-prune"]["allow_cfg_forms"] is True
        assert by_id["stair-prune"]["image_placement"] == "first"
        assert by_id["stair-pairfit"]["image_placement"] == "pair_fit"
        assert by_id["stair-pairfit"]["blocks"] is False

    def test_cfg2_unlocked_baselines_use_cfg_forms_on_pcie(self) -> None:
        prof = _pcie_profile()
        workload = [
            {"request_id": "v0", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 9500.0},
            {"request_id": "v1", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 9500.0},
        ]
        options = {"forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4"]}

        for sp_policy, cfg_policy in (
            ("tetriserve", "tetriserve-cfg2"),
            ("sbp-esp", "sbp-esp-cfg2"),
            ("edf-bestfit", "edf-bestfit-cfg2"),
        ):
            sp_res = _rpc_simulate(
                {"workload": workload, "profile": prof, "policy": sp_policy, "gpus": 4, "options": options}
            )
            cfg_res = _rpc_simulate(
                {"workload": workload, "profile": prof, "policy": cfg_policy, "gpus": 4, "options": options}
            )
            sp_forms = {f for r in sp_res["requests"] for f in r["forms"]}
            cfg_forms = {f for r in cfg_res["requests"] for f in r["forms"]}
            assert all(f.startswith("sp") for f in sp_forms), f"{sp_policy} used non-SP forms: {sp_forms}"
            assert any(f.startswith("cfg2") for f in cfg_forms), f"{cfg_policy} did not use CFG2 forms: {cfg_forms}"
            assert cfg_res["makespan_ms"] < sp_res["makespan_ms"]

    def test_abl_no_decode_shrink_holds_full_group_during_vae_decode(self) -> None:
        prof = _pcie_profile()
        workload = [
            {"request_id": "v0", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 20000.0},
            {"request_id": "i0", "kind": "image", "class": "image", "arrival_ms": 6000.0, "deadline_ms": 8600.0},
        ]
        options = {"forms": ["sp1", "cfg2"], "decode_chunk_frames": 0, "chunked_decode": False}

        dit_res = _rpc_simulate(
            {"workload": workload, "profile": prof, "policy": "ditserve", "gpus": 2, "options": options}
        )
        no_shrink_res = _rpc_simulate(
            {"workload": workload, "profile": prof, "policy": "abl-no-decode-shrink", "gpus": 2, "options": options}
        )

        dit_v0 = next(r for r in dit_res["requests"] if r["request_id"] == "v0")
        no_shrink_v0 = next(r for r in no_shrink_res["requests"] if r["request_id"] == "v0")
        assert dit_v0["degrees"] == [2, 1]
        assert no_shrink_v0["degrees"] == [2]
        assert dit_res["by_kind"]["image"]["ontime"] == 1
        assert no_shrink_res["by_kind"]["image"]["ontime"] == 0

    def test_abl_no_img_priority_disables_cross_modal_priority_boost(self) -> None:
        prof = _pcie_profile()
        workload = [
            {"request_id": "v1", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 18000.0},
            {"request_id": "i0", "kind": "image", "class": "image", "arrival_ms": 0.0, "deadline_ms": 25000.0},
        ]
        options = {"forms": ["cfg2"]}

        dit_res = _rpc_simulate(
            {"workload": workload, "profile": prof, "policy": "ditserve", "gpus": 2, "options": options}
        )
        no_prio_res = _rpc_simulate(
            {"workload": workload, "profile": prof, "policy": "abl-no-img-priority", "gpus": 2, "options": options}
        )

        dit_i0 = next(r for r in dit_res["requests"] if r["request_id"] == "i0")
        no_prio_i0 = next(r for r in no_prio_res["requests"] if r["request_id"] == "i0")
        assert dit_i0["start_ms"] == pytest.approx(0.0, abs=1e-3)
        assert no_prio_i0["start_ms"] > 0.0

    def test_tight_slo_preemption_and_proactive_slack_separation(self) -> None:
        prof = _pcie_profile()
        options = {"forms": ["cfg2"], "quantum_blocks": 4}

        # 1. Tight 1.0x SLO doorbell grace preemption separation (`ditserve` vs `abl-no-block`)
        tight_1p0_workload = [
            {"request_id": "v0", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 15000.0},
            {"request_id": "i_tight", "kind": "image", "class": "image", "arrival_ms": 100.0, "deadline_ms": 2600.0},
        ]
        dit_1p0 = _rpc_simulate(
            {"workload": tight_1p0_workload, "profile": prof, "policy": "ditserve", "gpus": 2, "options": options}
        )
        no_block_1p0 = _rpc_simulate(
            {"workload": tight_1p0_workload, "profile": prof, "policy": "abl-no-block", "gpus": 2, "options": options}
        )

        assert dit_1p0["block_preemptions"] > 0
        assert no_block_1p0["block_preemptions"] == 0
        dit_img = next(r for r in dit_1p0["requests"] if r["request_id"] == "i_tight")
        no_block_img = next(r for r in no_block_1p0["requests"] if r["request_id"] == "i_tight")
        assert dit_img["finish_ms"] < 2800.0
        assert (dit_img["finish_ms"] - dit_img["arrival_ms"]) < (
            no_block_img["finish_ms"] - no_block_img["arrival_ms"]
        ) * 0.5

        # 2. Proactive >65% slack block preemption separation (`ditserve` vs `block-cfg2`)
        proactive_workload = [
            {"request_id": "v0", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 15000.0},
            {"request_id": "i_proactive", "kind": "image", "class": "image", "arrival_ms": 100.0, "deadline_ms": 3250.0},
        ]
        dit_proactive = _rpc_simulate(
            {"workload": proactive_workload, "profile": prof, "policy": "ditserve", "gpus": 2, "options": options}
        )
        block_cfg2_proactive = _rpc_simulate(
            {"workload": proactive_workload, "profile": prof, "policy": "block-cfg2", "gpus": 2, "options": options}
        )
        assert dit_proactive["block_preemptions"] > 0
        assert block_cfg2_proactive["block_preemptions"] == 0

    def test_cross_res_sequence_packing_and_ditserve_control_optimizations(self) -> None:
        # Verify the C++ simulation-tests suite (including cross-resolution sequence_packing,
        # multi-video feasibility guard, idle-GPU widening, sunk-work deadband, laxity tie-break,
        # and same-degree migration guard) passes completely, and verify simulation behavior via RPC.
        proc = subprocess.run(
            [str(SIMULATION_TESTS_BIN)],
            text=True,
            capture_output=True,
            check=True,
        )
        assert "cluster checks passed" in proc.stdout

        # Simulate multi-video feasibility protection + idle-GPU widening on 8 GPUs
        prof = _pcie_profile()
        prof["classes"]["video"]["forms"] = {
            "sp1": {"step_ms": 4000.0},
            "cfg2": {"step_ms": 2000.0},
            "sp2": {"step_ms": 2400.0},
            "cfg2sp2": {"step_ms": 1250.0},
            "sp4": {"step_ms": 1600.0},
        }
        workload = [
            {"request_id": "v0", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 35000.0},
            {"request_id": "v1", "kind": "video", "class": "video", "arrival_ms": 0.0, "deadline_ms": 40000.0},
        ]
        res = _rpc_simulate(
            {
                "workload": workload,
                "profile": prof,
                "policy": "ditserve",
                "gpus": 8,
                "options": {"forms": ["sp1", "cfg2", "sp2", "cfg2sp2", "sp4"]},
            }
        )
        v0 = next(r for r in res["requests"] if r["request_id"] == "v0")
        assert "cfg2sp2" in v0["forms"]
        assert res["by_kind"]["video"]["ontime"] == 2

    def test_policy_level_continuous_batching_ownership_and_baseline_isolation(self) -> None:
        prof = _pcie_profile()
        prof["classes"]["image"] = {
            "kind": "image",
            "service_ms": 1778.08,
            "steps": 4,
            "prepare_ms": 45.0,
            "finish_ms": 135.0,
            "batch_step_ms": {"1": 400.0, "2": 720.0, "3": 1020.0, "4": 1350.0},
        }
        workload = [
            {"request_id": "i0", "kind": "image", "class": "image", "arrival_ms": 0.0, "deadline_ms": 12000.0},
            {"request_id": "i1", "kind": "image", "class": "image", "arrival_ms": 0.0, "deadline_ms": 12000.0},
        ]
        cb_options = {"continuous_batch": 4, "sequence_packing": True, "bubble_free": True, "forms": ["cfg2"]}

        dit_res = _rpc_simulate(
            {"workload": workload, "profile": prof, "policy": "ditserve", "gpus": 2, "options": cb_options}
        )
        for req in dit_res["requests"]:
            assert "continuous_batch" in req["action_ms"]
            assert "atomic" not in req["action_ms"]
            assert req["quanta"] == 4

        for non_cb_policy in (
            "abl-no-cb",
            "xdit",
            "tetriserve",
            "sbp-esp",
            "edf-bestfit",
            "edf-cfg2",
            "block-cfg2",
        ):
            res = _rpc_simulate(
                {"workload": workload, "profile": prof, "policy": non_cb_policy, "gpus": 2, "options": cb_options}
            )
            for req in res["requests"]:
                assert "atomic" in req["action_ms"], f"{non_cb_policy} did not execute atomic quantum"
                assert "continuous_batch" not in req["action_ms"], (
                    f"{non_cb_policy} leaked continuous_batch quantum"
                )
                assert req["quanta"] == 1

    def test_work_conserving_idle_gpu_placement_and_cross_class_slo_guard(self) -> None:
        # 1. Idle-GPU spreading: 2 free GPUs + 2 unstarted images spread across GPU 0 and GPU 1 at B=1 each
        prof = _pcie_profile()
        prof["classes"]["image"] = {
            "kind": "image",
            "service_ms": 1778.08,
            "steps": 4,
            "prepare_ms": 45.0,
            "finish_ms": 135.0,
            "batch_step_ms": {"1": 400.0, "2": 720.0, "3": 1020.0, "4": 1350.0},
        }
        workload_2img = [
            {"request_id": "i0", "kind": "image", "class": "image", "arrival_ms": 0.0, "deadline_ms": 10000.0},
            {"request_id": "i1", "kind": "image", "class": "image", "arrival_ms": 0.0, "deadline_ms": 10000.0},
        ]
        cb_opts = {"continuous_batch": 4, "sequence_packing": True, "bubble_free": True, "forms": ["sp1", "cfg2"]}

        spread_2gpu = _rpc_simulate(
            {"workload": workload_2img, "profile": prof, "policy": "ditserve", "gpus": 2, "options": cb_opts}
        )
        # B=1 over 4 steps on GPU: 4 * (400 + 1.5) = 1606.0 ms GPU makespan, with finish_ms=1741.0 ms per request
        assert spread_2gpu["makespan_ms"] == pytest.approx(4 * 401.5, abs=1.0)
        for req in spread_2gpu["requests"]:
            assert req["finish_ms"] == pytest.approx(4 * 401.5 + 135.0, abs=1.0)
        assert spread_2gpu["batches"] == 8
        assert spread_2gpu["batched_images"] == 8  # 8 solo rounds (B=1 on GPU 0 and B=1 on GPU 1)

        # When 1 GPU is available (contended), the 2 images co-batch at B=2 and beat 2 sequential B=1 images
        cobatch_1gpu = _rpc_simulate(
            {"workload": workload_2img, "profile": prof, "policy": "ditserve", "gpus": 1, "options": cb_opts}
        )
        assert cobatch_1gpu["batches"] == 4
        assert cobatch_1gpu["batched_images"] == 8  # 4 co-batched B=2 rounds
        assert cobatch_1gpu["makespan_ms"] == pytest.approx(4 * 721.5, abs=1.0)
        for req in cobatch_1gpu["requests"]:
            assert req["finish_ms"] == pytest.approx(4 * 721.5 + 135.0, abs=1.0)
        assert cobatch_1gpu["makespan_ms"] < 2 * spread_2gpu["makespan_ms"]

        # 2. Cross-class SLO guard: tight-SLO sd35-512 image refuses to co-batch with slow sd35-1440 image
        prof_multi = _pcie_profile()
        prof_multi["classes"]["sd35-512"] = {
            "kind": "image",
            "service_ms": 1778.08,
            "steps": 28,
            "prepare_ms": 45.0,
            "finish_ms": 135.0,
            "batch_step_ms": {"1": 57.074, "2": 104.903, "3": 152.331, "4": 203.584},
        }
        prof_multi["classes"]["sd35-1440"] = {
            "kind": "image",
            "service_ms": 18235.67,
            "steps": 28,
            "prepare_ms": 45.0,
            "finish_ms": 135.0,
            "batch_step_ms": {"1": 644.845, "2": 1251.0, "3": 1827.49, "4": 2442.67},
        }
        mixed_img_workload = [
            {
                "request_id": "img_512_tight",
                "kind": "image",
                "class": "sd35-512",
                "arrival_ms": 0.0,
                "deadline_ms": 2220.0,
                "spec": {"height": 512, "width": 512, "steps": 28},
            },
            {
                "request_id": "img_1440_relaxed",
                "kind": "image",
                "class": "sd35-1440",
                "arrival_ms": 0.0,
                "deadline_ms": 30000.0,
                "spec": {"height": 1440, "width": 1440, "steps": 28},
            },
        ]
        guard_res = _rpc_simulate(
            {"workload": mixed_img_workload, "profile": prof_multi, "policy": "ditserve", "gpus": 1, "options": cb_opts}
        )
        r512 = next(r for r in guard_res["requests"] if r["request_id"] == "img_512_tight")
        r1440 = next(r for r in guard_res["requests"] if r["request_id"] == "img_1440_relaxed")
        assert r512["finish_ms"] <= 2220.0
        assert r1440["finish_ms"] <= 30000.0
        assert guard_res["by_kind"]["image"]["ontime"] == 2

        # 3. Prepared-leader idle-GPU fair cap & co-resident continuity:
        #    When i0 is mid-generation on GPU 0 and i1 arrives while GPU 1 is idle, i1 runs solo on GPU 1
        #    instead of late-joining GPU 0 into a slower B=2 batch.
        staggered_2gpu_workload = [
            {
                "request_id": "img_mid_gpu0",
                "kind": "image",
                "class": "sd35-512",
                "arrival_ms": 0.0,
                "deadline_ms": 2667.0,
                "spec": {"height": 512, "width": 512, "steps": 28},
            },
            {
                "request_id": "img_late_arrival",
                "kind": "image",
                "class": "sd35-512",
                "arrival_ms": 800.0,
                "deadline_ms": 800.0 + 2000.0,
                "spec": {"height": 512, "width": 512, "steps": 28},
            },
        ]
        staggered_res = _rpc_simulate(
            {"workload": staggered_2gpu_workload, "profile": prof_multi, "policy": "ditserve", "gpus": 2, "options": cb_opts}
        )
        assert staggered_res["by_kind"]["image"]["ontime"] == 2
        assert staggered_res["batches"] == 56  # 28 solo steps on GPU 0 + 28 solo steps on GPU 1 (zero B=2 stealing)


# ==============================================================================
# 2. 3-Parameter Identifiable Cost Model & Topology Phase Diagram
# ==============================================================================


class TestIdentifiableCostModelAndPhaseDiagram:
    """Verifies the 3-parameter identifiable cost model and topology phase diagram."""

    ARCH = {
        "tokens": 32760,
        "dim": 1536,
        "layers": 30,
        "comm_mb_per_step": 24153.2928,
        "collectives_per_step": 240,
        "cfg_exchange_mb": 4.19328,
    }
    FORMS = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 6), (1, 8)]

    @classmethod
    def _policy_of(cls, cfg: int, sp: int) -> str:
        if cfg == 2:
            return "static_cfg2" if sp == 1 else f"static_cfg2sp{sp}"
        return f"static_sp{sp}"

    @classmethod
    def _synthetic_rows(cls, model: str, params: dict, t1: float = 4000.0, forms=None) -> list[dict]:
        use_forms = forms or cls.FORMS
        return [
            {
                "policy": cls._policy_of(c, s),
                "cfg": c,
                "sp": s,
                "t1": t1,
                "arch": cls.ARCH,
                "dataset": "synthetic",
                "step_ms": cmv.model_predict(model, params, t1, cls.ARCH, c, s),
            }
            for c, s in use_forms
        ]

    def test_3param_identifiability_condition_number_and_4param_rank_deficiency(self) -> None:
        # 1. 3-parameter bus and baseline_cont models recover exact synthetic parameters and are well-conditioned
        bus_truth = {"link_gbps": 8.0, "bus_gbps": 30.0, "latency": 0.8}
        bus_rows = self._synthetic_rows("bus", bus_truth)
        bus_params, bus_loss = cmv.fit_model("bus", bus_rows, objective="rel_mse")
        assert bus_loss < 1e-8
        assert bus_params["link_gbps"] == pytest.approx(8.0, abs=0.2)
        assert bus_params["bus_gbps"] == pytest.approx(30.0, abs=1.0)
        assert bus_params["latency"] == pytest.approx(0.8, abs=0.05)

        bus_diag = cmv.identifiability("bus", bus_truth, bus_rows)
        assert bus_diag["n_params"] == 3
        assert bus_diag["status"] == "ok"
        assert bus_diag["condition_number"] < 100.0

        # 2. 4-parameter amdahl_sp on 4 informative points is unidentifiable
        amdahl_truth = {"bandwidth_inf": 5.0, "gamma": 0.0, "latency": 1.5, "serial": 0.12}
        few_rows = self._synthetic_rows("amdahl_sp", amdahl_truth, forms=[(1, 1), (2, 1), (1, 2), (2, 2), (1, 4)])
        diag_4p = cmv.identifiability("amdahl_sp", amdahl_truth, few_rows)
        assert diag_4p["informative_points"] == 4
        assert diag_4p["status"] == "unidentifiable"

        # 3. Collinear 4-parameter linear design matrix has SVD condition number > 10^15
        rows_4p = []
        for cfg, sp in ((1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (2, 4)):
            scale = 1.0 / (cfg * sp)
            comm_ind = 1.0 if sp > 1 else 0.0
            cfg_sync = (cfg - 1.0) * (1.0 if sp > 1 else 0.0)
            rows_4p.append([scale, comm_ind, cfg_sync, comm_ind - cfg_sync])
        cond_4p = float(np.linalg.cond(np.asarray(rows_4p, dtype=float)))
        assert cond_4p > 1e15

    def test_zero_ranking_regret_lofo_and_fit_native_profile_parity(self) -> None:
        # 1. Exact agreement between cmv.baseline_predict and fit_native_profile.predict
        for cfg, sp in self.FORMS:
            p_cmv = cmv.baseline_predict(4000.0, self.ARCH, cfg, sp, 6.0, 1.0, 0.2)
            p_fnp = fnp.predict(4000.0, self.ARCH, cfg, sp, 6.0, 1.0, 0.2)
            assert p_cmv == pytest.approx(p_fnp, rel=1e-12)

        # 2. Grid fit matches fit_native_profile.fit_links and achieves 0% ranking regret
        truth = {"bandwidth_inf": 6.0, "gamma": 1.0, "latency": 0.2}
        rows = self._synthetic_rows("baseline_grid", truth)
        videos = [{"policy": r["policy"], "form": (r["cfg"], r["sp"]), "step_ms": r["step_ms"]} for r in rows]
        step_compute, expected_links, _ = fnp.fit_links(videos, self.ARCH)
        fit_rows = [r for r in rows if r["policy"] in cmv.FIT_POLICIES]
        params, _ = cmv.fit_model("baseline_grid", fit_rows)
        assert (params["bandwidth_inf"], params["gamma"], params["latency"]) == tuple(expected_links)

        errors = cmv.relative_errors("baseline_grid", params, rows)
        decisions = cmv.decision_check(errors)
        assert len(decisions) >= 3
        assert all(d["agree"] for d in decisions)
        assert all(d["regret_pct"] == 0.0 for d in decisions)

        # 3. LOFO isolation: corrupting a held-out form never changes its LOFO prediction
        cont_rows = self._synthetic_rows("baseline_cont", {"bandwidth_inf": 4.0, "gamma": 0.5, "latency": 0.7})
        base_folds = {
            f["policy"]: f for f in cmv.leave_one_form_out("baseline_cont", cont_rows, starts=3, max_iter=300)
        }
        corrupted = [
            dict(r, step_ms=r["step_ms"] * 10.0) if r["policy"] == "static_sp6" else r for r in cont_rows
        ]
        corrupted_folds = {
            f["policy"]: f for f in cmv.leave_one_form_out("baseline_cont", corrupted, starts=3, max_iter=300)
        }
        assert corrupted_folds["static_sp6"]["predicted_step_ms"] == pytest.approx(
            base_folds["static_sp6"]["predicted_step_ms"], abs=1e-9
        )

    def test_fit_native_profile_build_profile_recovers_held_out_forms(self) -> None:
        policies = list(fnp.FIT_POLICIES) + [
            "static_sp3",
            "static_cfg2sp3",
            "static_sp6",
            "static_cfg2sp4",
            "static_sp8",
        ]
        spec = {"frames": 49, "steps": 20, "height": 512, "width": 896, "guidance_scale": 5.0}
        model = {
            "transformer_config": {
                "patch_size": [1, 2, 2],
                "num_attention_heads": 24,
                "attention_head_dim": 128,
                "num_layers": 30,
                "out_channels": 48,
            },
            "vae_config": {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
        }
        facts = {"models": {"Wan2.2-TI2V-5B": model}}
        arch = fnp.architecture(model, spec)
        assert arch["tokens"] == 5824

        requests = []
        for index, policy in enumerate(policies):
            cfg, sp = fnp.form_of(policy)
            step_ms = fnp.predict(500.0, arch, cfg, sp, 12.0, 1.0, 0.05)
            requests.append(
                {
                    "cell": index,
                    "kind": "video",
                    "status": "completed",
                    "model_ms": 20 * step_ms + 100.0,
                    "outputs": [{"model_ms": 100.0}],
                    "spec": spec,
                    "encode_model_ms": 20.0,
                    "advance_samples": [
                        {"steady": False, "model_ms": 999999.0},
                        {"steady": True, "model_ms": step_ms, "group": list(range(cfg * sp))},
                        {"steady": True, "model_ms": step_ms, "group": list(range(cfg * sp))},
                    ],
                    "transfer_export_ms": 2.0,
                    "dispatch_gap_ms": 4.0,
                    "timed_quanta": 2,
                }
            )
        requests.append({"cell": 0, "kind": "image", "status": "completed", "model_ms": 250.0})
        record = {
            "experiment": "kairos-wan22-native-calibration",
            "run_id": "fixture",
            "status": "completed",
            "phases": [{"cells": [{"config": {"policy": policy}} for policy in policies]}],
            "requests": requests,
            "budget": {"sessions": [{"identity": {"gpu_uuids": [f"GPU-{i}" for i in range(8)]}}]},
        }
        profile = fnp.build_profile(record, facts, "Wan2.2-TI2V-5B")
        assert profile["source"]["held_out_max_relative_error_pct"] < 1e-8
        assert profile["classes"]["sd35-preflight"]["service_ms"] == 250.0
        assert profile["classes"]["wan22-preflight"]["step_compute_ms"] == 500.0
        assert profile["source"]["execution_cost_mode"] == "placement_bound_form_table_with_analytical_fallback"

    def test_calibration_bundle_finish_ms_consistency_and_multires_batch_step_ms(self) -> None:
        # 1. Verify default_sd35_batch_step_ms across 512, 768, 1024, 1440
        for res, svc_ms in ((512, 1778.081), (768, 2200.0), (1024, 2856.256), (1440, 4892.341)):
            table = fnp.default_sd35_batch_step_ms(svc_ms, steps=28, resolution=res)
            assert {"1", "2", "3", "4"}.issubset(set(table.keys()))
            assert table["1"] < table["2"] < table["3"] < table["4"]
            # Sublinear scaling: B=4 step latency is strictly less than 4 * B=1
            assert table["4"] < 4.0 * table["1"]

        # 2. Verify build_profile does not mistake request completion timestamp req["finish_ms"]
        # for 2D-VAE decode duration.
        policies = list(fnp.FIT_POLICIES)
        spec = {"frames": 49, "steps": 20, "height": 512, "width": 896, "guidance_scale": 5.0}
        model = {
            "transformer_config": {
                "patch_size": [1, 2, 2],
                "num_attention_heads": 24,
                "attention_head_dim": 128,
                "num_layers": 30,
                "out_channels": 48,
            },
            "vae_config": {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
        }
        facts = {"models": {"Wan2.2-TI2V-5B": model}}
        arch = fnp.architecture(model, spec)
        requests = []
        for index, policy in enumerate(policies):
            cfg, sp = fnp.form_of(policy)
            step_ms = fnp.predict(500.0, arch, cfg, sp, 12.0, 1.0, 0.05)
            requests.append(
                {
                    "cell": index,
                    "kind": "video",
                    "status": "completed",
                    "model_ms": 20 * step_ms + 100.0,
                    "outputs": [{"model_ms": 100.0}],
                    "spec": spec,
                    "encode_model_ms": 20.0,
                    "advance_samples": [
                        {"steady": False, "model_ms": 999999.0},
                        {"steady": True, "model_ms": step_ms, "group": list(range(cfg * sp))},
                        {"steady": True, "model_ms": step_ms, "group": list(range(cfg * sp))},
                    ],
                }
            )
        # Image request with wall-clock completion timestamp finish_ms=2840.694 > model_ms=1778.081
        requests.append(
            {
                "cell": 0,
                "kind": "image",
                "status": "completed",
                "model_ms": 1778.0806655064225,
                "finish_ms": 2840.694,
                "spec": {"height": 512, "width": 512, "steps": 28},
            }
        )
        record = {
            "experiment": "kairos-wan22-native-calibration",
            "run_id": "fixture-finish-ms",
            "status": "completed",
            "phases": [{"cells": [{"config": {"policy": policy}} for policy in policies]}],
            "requests": requests,
            "budget": {"sessions": [{"identity": {"gpu_uuids": [f"GPU-{i}" for i in range(8)]}}]},
        }
        profile = fnp.build_profile(record, facts, "Wan2.2-TI2V-5B")
        img_cost = profile["classes"]["sd35-preflight"]
        assert img_cost["finish_ms"] < 200.0
        assert (
            img_cost["prepare_ms"] + img_cost["steps"] * img_cost["batch_step_ms"]["1"] + img_cost["finish_ms"]
            == pytest.approx(img_cost["service_ms"], abs=0.1)
        )

        # 3. Verify local calibration bundles b1 and b2 if present on disk
        import formal_benchmark as fb

        bundle_paths = [
            pathlib.Path("/mnt/e/000_code/zyd-2026/.local/formal-v2-cb-refresh-jobs/calibration-bundles/d20261106-formal-v2-wan22-b1-21a1ca3-r1/bundle.json"),
            pathlib.Path("/mnt/e/000_code/zyd-2026/.local/formal-v2-cb-refresh-jobs/calibration-bundles/d20261106-formal-v2-wan22-b2-21a1ca3-r1/bundle.json"),
        ]
        for bpath in bundle_paths:
            if bpath.exists():
                loaded = fb.load_bundle(bpath)
                for sid in ("sd35-512", "sd35-1024", "sd35-1440"):
                    c = loaded["classes"][sid]["cost"]
                    total_b1 = c["prepare_ms"] + c["steps"] * c["batch_step_ms"]["1"] + c["finish_ms"]
                    assert total_b1 == pytest.approx(c["service_ms"], abs=0.1)

    def test_topology_phase_diagram_crossovers_and_5_canonical_shapes(self) -> None:
        report = tpd.build_phase_diagram_summary()

        assert set(report["interconnect_tiers"].keys()) == {"pcie4_phb", "pcie5", "nvlink"}
        assert set(report["canonical_shapes"].keys()) == {
            "wan21-dev",
            "wan21-480p",
            "wan21-720p",
            "wan22-small",
            "wan22-720p",
        }
        assert report["canonical_shapes"]["wan21-dev"]["tokens"] == 14040
        assert report["canonical_shapes"]["wan21-480p"]["tokens"] == 32760
        assert report["canonical_shapes"]["wan21-720p"]["tokens"] == 75600
        assert report["canonical_shapes"]["wan22-small"]["tokens"] == 12150
        assert report["canonical_shapes"]["wan22-720p"]["tokens"] == 43200

        kf = report["key_findings"]
        assert kf["pcie4_phb_busbw_range_gbps"] == [7.85, 5.06]
        assert kf["pcie5_busbw_range_gbps"] == [15.7, 10.1]
        assert kf["nvlink_busbw_range_gbps"] == [150.0, 300.0]
        assert kf["cfg2_intra_layer_collectives"] == 0
        assert kf["cfg2_exchange_mb_range"] == [1.80, 10.48]
        assert kf["ulysses_sp_collectives_per_step"] == 240
        assert kf["ulysses_sp_traffic_gb_range"] == [10.35, 55.74]

        l_star_pcie4 = kf["pcie4_crossover_tokens_l_star"]
        l_star_pcie5 = kf["pcie5_crossover_tokens_l_star"]
        l_star_nvlink = kf["nvlink_crossover_tokens_l_star"]
        assert 35000.0 <= l_star_pcie4 <= 40500.0
        assert l_star_pcie4 > l_star_pcie5 > l_star_nvlink

        pcie4_grid = report["phase_diagram_by_tier"]["pcie4_phb"]
        for below_shape in ("wan22-small", "wan21-dev", "wan21-480p"):
            assert pcie4_grid[below_shape]["above_crossover"] is False
            assert pcie4_grid[below_shape]["optimal_regime"] == "cfg2_dominant"
        for above_shape in ("wan22-720p", "wan21-720p"):
            assert pcie4_grid[above_shape]["above_crossover"] is True
            assert pcie4_grid[above_shape]["optimal_regime"] == "cfg2xsp_hybrid_dominant"

        speedup_lo, speedup_hi = kf["hybrid_cfg2xsp_speedup_range_above_crossover"]
        assert 2.25 <= speedup_lo <= 2.30
        assert 4.05 <= speedup_hi <= 4.10

    def test_docs_contain_formal_mixed_v2_and_phase_diagram_contracts(self) -> None:
        for rel_path in ("docs/experiments.md", "docs/motivation.md", "docs/motivation-无损.md"):
            doc_text = (ROOT / rel_path).read_text(encoding="utf-8")
            for required_token in (
                "formal-mixed-v2",
                "tetriserve-cfg2",
                "sbp-esp-cfg2",
                "edf-bestfit-cfg2",
                "topology_phase_diagram.py",
            ):
                assert required_token in doc_text, f"Missing {required_token!r} in {rel_path}"
        exp_text = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")
        for ledger_token in (
            "modality_mix_summary.json",
            "rate_scale_summary.json",
            "slo_scale_summary.json",
            "gpu_scale_summary.json",
            "ablation_summary.json",
        ):
            assert ledger_token in exp_text, f"Missing {ledger_token!r} in docs/experiments.md"


# ==============================================================================
# 3. Budget Ledger (`ditserve.budget.v1`), Study (`ditserve.study.v1`), & CLI
# ==============================================================================


class _FixtureBridge(CoreBridge):
    def __init__(self, package_root: pathlib.Path):
        super().__init__()
        self.package_root = package_root

    def call(self, command: str, payload: dict) -> dict:
        return super().call(command, dict(payload, package_root=str(self.package_root)))


class TestBudgetLedgerStudyRunnerAndCliContracts:
    """Verifies `ditserve.budget.v1`, `ditserve.study.v1`, and native C++ CLI dispatch."""

    @staticmethod
    def _make_package_root(tmp_path: pathlib.Path) -> pathlib.Path:
        import shutil

        pkg_root = tmp_path / "pkg"
        for package in ("ditserve", "xfuser"):
            source = PACKAGE_ROOT / package
            for path in source.rglob("*.py"):
                target = pkg_root / package / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        return pkg_root

    @staticmethod
    def _request(request_id: str = "r0", **changes) -> dict:
        val = {
            "request_id": request_id,
            "arrival_ms": 0,
            "service_ms": 20,
            "slo_ms": 10000,
            "spec": {"steps": 2, "prompt": "a ceramic kettle", "seed": 7},
        }
        val.update(changes)
        return val

    def test_study_runner_paired_matrix_budget_ledger_and_resume_identity(self, tmp_path: pathlib.Path) -> None:
        pkg_root = self._make_package_root(tmp_path)
        bridge = _FixtureBridge(pkg_root)

        phase = {
            "id": "online",
            "kind": "online",
            "worker": "model",
            "policy": "edf",
            "workload": [self._request("a"), self._request("b")],
            "matrix": {"/policy": ["edf", "srtf"]},
            "paired": True,
        }
        config = {
            "schema": "ditserve.study.v1",
            "experiment": "native",
            "workers": {"model": {"backend": "mock", "config": {"options": {}}}},
            "budget": {"limit_gpu_seconds": None, "watchdog_seconds": 10},
            "phases": [phase],
        }
        ledger_path = tmp_path / "run-budget.json"
        value = api.run(
            config,
            run_id="run",
            work_dir=tmp_path / "run",
            results_root=tmp_path / "results",
            ledger=ledger_path,
            core=bridge,
        )
        assert value["status"] == "completed"
        assert value["schema"] == "ditserve.run.v1"
        ledger_on_disk = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger_on_disk["active"] is None
        assert ledger_on_disk["used_gpu_seconds"] == 0
        assert value["budget"]["active"] is None
        assert value["budget"]["used_gpu_seconds"] == 0
        assert all(item["cleanup_verified"] for item in value["budget"]["sessions"])

        cells = value["phases"][0]["cells"]
        assert [(c["cell"], c["order"]) for c in cells] == [
            (0, "forward"),
            (1, "forward"),
            (1, "reverse"),
            (0, "reverse"),
        ]
        assert value["metrics"]["requests"] == 8
        assert value["metrics"]["completed"] == 8

        # Resume is idempotent on completed study and enforces frozen config / ledger identity
        before_bytes = ledger_path.read_bytes()
        resumed = api.run(
            config,
            run_id="run",
            work_dir=tmp_path / "run",
            results_root=tmp_path / "results",
            ledger=ledger_path,
            core=bridge,
            resume=True,
        )
        assert resumed["status"] == "completed"
        assert resumed["phases"] == value["phases"]
        assert ledger_path.read_bytes() == before_bytes

        with pytest.raises(CoreError, match="run_id differs"):
            api.run(
                config,
                run_id="changed-id",
                work_dir=tmp_path / "run",
                results_root=tmp_path / "results",
                ledger=ledger_path,
                core=bridge,
                resume=True,
            )

    def test_study_runner_phase_gates_hot_cold_eviction_and_session_think_time(self, tmp_path: pathlib.Path) -> None:
        pkg_root = self._make_package_root(tmp_path)
        bridge = _FixtureBridge(pkg_root)

        # 1. Phase gates block downstream phases when quality gate fails
        first = {
            "id": "profile",
            "kind": "profile",
            "worker": "model",
            "policy": "edf",
            "workload": [self._request()],
            "gates": [{"metric": "/cells/0/metrics/completed", "op": "ge", "value": 1}],
        }
        second = {
            "id": "quality",
            "kind": "online",
            "worker": "model",
            "policy": "edf",
            "workload": [self._request()],
            "requires": [{"phase": "profile", "requires": "qualified"}],
            "gates": [{"metric": "/cells/0/metrics/quality_score", "op": "ge", "value": 0.9}],
        }
        third = {
            "id": "after_quality",
            "kind": "online",
            "worker": "model",
            "policy": "edf",
            "workload": [self._request()],
            "requires": [{"phase": "quality", "requires": "qualified"}],
        }
        cfg_gates = {
            "schema": "ditserve.study.v1",
            "experiment": "native",
            "workers": {"model": {"backend": "mock", "config": {"options": {}}}},
            "budget": {"limit_gpu_seconds": None, "watchdog_seconds": 10},
            "phases": [first, second, third],
        }
        gated_val = api.run(
            cfg_gates,
            run_id="gated",
            work_dir=tmp_path / "gated",
            results_root=tmp_path / "results",
            ledger=tmp_path / "gated-budget.json",
            core=bridge,
        )
        assert gated_val["status"] == "incomplete"
        assert gated_val["blocked_phase"] == "after_quality"
        assert [p["id"] for p in gated_val["phases"]] == ["profile", "quality"]

        # 2. Two-stage session causal think time
        stages = [
            {"worker": "text", "spec": {"steps": 2, "prompt": "blue cup", "seed": 8}, "cost_ms": 10},
            {
                "worker": "image",
                "spec": {"steps": 2, "prompt": "overwritten", "seed": 9},
                "prompt_from_previous": True,
                "cost_ms": 30,
            },
        ]
        rows = [
            self._request("first", session_id="chat", stages=stages),
            self._request("second", session_id="chat", stages=stages, after="first", think_ms=17),
        ]
        cfg_sess = {
            "schema": "ditserve.study.v1",
            "experiment": "native",
            "workers": {"text": {"backend": "mock"}, "image": {"backend": "mock"}},
            "budget": {"limit_gpu_seconds": None, "watchdog_seconds": 10},
            "phases": [{"id": "session", "kind": "session", "policy": "stage_edf", "worker": "text", "workload": rows}],
        }
        sess_val = api.run(
            cfg_sess,
            run_id="sess",
            work_dir=tmp_path / "sess",
            results_root=tmp_path / "results",
            ledger=tmp_path / "sess-budget.json",
            core=bridge,
        )
        assert sess_val["status"] == "completed"
        req1, req2 = sess_val["requests"]
        assert req2["arrival_ms"] == pytest.approx(req1["finish_ms"] + 17.0)
        assert req2["start_ms"] >= req2["arrival_ms"]

    def test_cli_dispatch_help_rpc_and_simulate_summary_publication(self, tmp_path: pathlib.Path) -> None:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        env.pop("PYTHONPATH", None)

        def invoke(args: list[str], payload: dict | None = None) -> dict:
            cmd = [
                str(DITSERVE_BIN),
                "--root",
                str(tmp_path),
                "--package-root",
                str(ROOT / "workers"),
                "--python",
                sys.executable,
            ] + args
            proc = subprocess.run(
                cmd,
                cwd=tmp_path,
                env=env,
                input=json.dumps(payload) if payload is not None else None,
                text=True,
                capture_output=True,
                check=True,
            )
            return json.loads(proc.stdout)

        help_proc = subprocess.run([str(DITSERVE_BIN), "--help"], capture_output=True, text=True, check=True)
        assert "simulate" in help_proc.stdout
        assert "results" in help_proc.stdout

        generated = invoke(
            ["rpc", "generate"],
            payload={
                "generator": {
                    "count": 2,
                    "seed": 7,
                    "pattern": "poisson",
                    "rate_rps": 2,
                    "classes": [
                        {"class": "video", "kind": "video", "service_ms": 3400},
                        {"class": "image", "kind": "image", "service_ms": 40},
                    ],
                }
            },
        )
        assert len(generated) == 2

        matrix_cells = invoke(
            ["rpc", "control"],
            payload={"op": "matrix", "phase": {"id": "fixture", "matrix": {"/policy": ["edf", "fcfs"]}}},
        )
        assert len(matrix_cells) == 2

        sim_published = invoke(["--json", "simulate", "--count", "2", "--seed", "9", "--run-id", "sim"])
        assert sim_published["status"] == "completed"
        dest = tmp_path / "results/synthetic/sim"
        verified = invoke(["results", "verify", str(dest)])
        assert verified["status"] == "verified"
        assert verified["format"] == "ditserve.results-summary.v1"

    def test_formal_benchmark_serving_5point_slo_scale_sweep(self) -> None:
        from unittest.mock import patch

        import formal_benchmark
        import native_serving
        from ditserve.evaluation.common import canonical_hash, seal

        model = "wan22"
        uuids = [f"GPU-00000000-0000-0000-0000-00000000030{i}" for i in range(8)]
        proto = formal_benchmark.protocol(ROOT / "configs" / "benchmarks" / "formal-mixed-v2.json")
        assert canonical_hash(proto) == "1531fe434654134d508d59fc88073a68585ef43a400a9113c0b300f0a08734de"

        manifests = {
            key: {
                "path": f"/shared/data/DITSERVE/models/{key}/manifest.json",
                "sha256": str(index + 1) * 64,
            }
            for index, key in enumerate((model, "sd35"))
        }
        classes = {}
        for sid, shape in proto["shapes"].items():
            if shape["scope"] != "formal" or shape["model"] not in (model, "sd35"):
                continue
            if shape["kind"] == "video":
                cost = {
                    "kind": "video",
                    "steps": shape["steps"],
                    "prepare_ms": 0.0,
                    "step_compute_ms": 100.0 * (shape["height"] / 480.0),
                    "finish_ms": 50.0,
                    "encode_ms": 10.0,
                    "links": {str(degree): {"gbps": 5.0, "latency_ms": 0.1} for degree in (2, 3, 4, 6, 8)},
                }
            else:
                cost = {"kind": "image", "service_ms": 250.0 * (shape["height"] / 512.0) ** 2}
            classes[sid] = {
                "spec": formal_benchmark.shape_spec(proto, sid),
                "cost": cost,
                "source": {
                    "gpu_uuids": list(uuids),
                    "resize_ms": 10.0,
                    "quantum_overhead_ms": 1.0,
                },
            }
        bundle = seal(
            {
                "schema": formal_benchmark.BUNDLE_SCHEMA,
                "protocol_sha256": canonical_hash(proto),
                "model": model,
                "model_manifests": manifests,
                "gpu_uuids": list(uuids),
                "classes": classes,
            }
        )
        pools = {
            kind: {
                "kind": kind,
                "sha256": str(index + 3) * 64,
                "samples": [
                    {"sample_id": f"{kind}-{sample}", "prompt": f"{kind} prompt {sample}"}
                    for sample in range(50)
                ],
            }
            for index, kind in enumerate(("image", "video"))
        }

        scales = (1.0, 1.25, 1.5, 1.75, 2.0)
        scale_tags = ("1p0", "1p25", "1p5", "1p75", "2p0")
        observed_experiments = []
        with patch.object(formal_benchmark, "split_pools", return_value=pools):
            for slo_scale, stag in zip(scales, scale_tags):
                cfg_p = formal_benchmark.serving(
                    proto,
                    model,
                    uuids,
                    bundle,
                    {},
                    manifests,
                    split="test",
                    seed=301,
                    rate_rpm=12.0,
                    video_fraction=0.5,
                    pattern="poisson",
                    suite="sweep",
                    slo_scale=slo_scale,
                    resolution_distribution="uniform",
                )
                exp_p = f"formal-wan22-test-poisson-s301-r12rpm-v0p5-slo{stag}-sweep"
                assert cfg_p["experiment"] == exp_p
                assert cfg_p["benchmark"]["slo_scale"] == slo_scale
                assert cfg_p["benchmark"]["absolute_rate"]["requested_rate_rpm"] == 12.0
                observed_experiments.append(cfg_p["experiment"])

                cfg_g = formal_benchmark.serving(
                    proto,
                    model,
                    uuids,
                    bundle,
                    {},
                    manifests,
                    split="test",
                    seed=301,
                    rate_rpm=12.0,
                    video_fraction=0.5,
                    pattern="gamma",
                    cv=2.0,
                    suite="sweep",
                    slo_scale=slo_scale,
                    resolution_distribution="uniform",
                )
                exp_g = f"formal-wan22-test-gamma-s301-r12rpm-v0p5-slo{stag}-cv2p0-sweep"
                assert cfg_g["experiment"] == exp_g
                assert cfg_g["benchmark"]["slo_scale"] == slo_scale
                assert cfg_g["benchmark"]["absolute_rate"]["requested_rate_rpm"] == 12.0
                observed_experiments.append(cfg_g["experiment"])

        assert len(observed_experiments) == 10
        assert len(set(observed_experiments)) == 10

    def test_formal_benchmark_serving_5point_modality_mix_sweep(self) -> None:
        from unittest.mock import patch

        import formal_benchmark
        import native_serving
        from ditserve.evaluation.common import canonical_hash, seal

        model = "wan22"
        uuids = [f"GPU-00000000-0000-0000-0000-00000000030{i}" for i in range(8)]
        proto = formal_benchmark.protocol(ROOT / "configs" / "benchmarks" / "formal-mixed-v2.json")
        assert canonical_hash(proto) == "1531fe434654134d508d59fc88073a68585ef43a400a9113c0b300f0a08734de"

        manifests = {
            key: {
                "path": f"/shared/data/DITSERVE/models/{key}/manifest.json",
                "sha256": str(index + 1) * 64,
            }
            for index, key in enumerate((model, "sd35"))
        }
        classes = {}
        for sid, shape in proto["shapes"].items():
            if shape["scope"] != "formal" or shape["model"] not in (model, "sd35"):
                continue
            if shape["kind"] == "video":
                cost = {
                    "kind": "video",
                    "steps": shape["steps"],
                    "prepare_ms": 0.0,
                    "step_compute_ms": 100.0 * (shape["height"] / 480.0),
                    "finish_ms": 50.0,
                    "encode_ms": 10.0,
                    "links": {str(degree): {"gbps": 5.0, "latency_ms": 0.1} for degree in (2, 3, 4, 6, 8)},
                }
            else:
                cost = {"kind": "image", "service_ms": 250.0 * (shape["height"] / 512.0) ** 2}
            classes[sid] = {
                "spec": formal_benchmark.shape_spec(proto, sid),
                "cost": cost,
                "source": {
                    "gpu_uuids": list(uuids),
                    "resize_ms": 10.0,
                    "quantum_overhead_ms": 1.0,
                },
            }
        bundle = seal(
            {
                "schema": formal_benchmark.BUNDLE_SCHEMA,
                "protocol_sha256": canonical_hash(proto),
                "model": model,
                "model_manifests": manifests,
                "gpu_uuids": list(uuids),
                "classes": classes,
            }
        )
        pools = {
            kind: {
                "kind": kind,
                "sha256": str(index + 3) * 64,
                "samples": [
                    {"sample_id": f"{kind}-{sample}", "prompt": f"{kind} prompt {sample}"}
                    for sample in range(100)
                ],
            }
            for index, kind in enumerate(("image", "video"))
        }

        mix_points = (
            (0.1, "0p1", {"video": 10, "image": 90}),
            (0.3, "0p3", {"video": 30, "image": 70}),
            (0.5, "0p5", {"video": 50, "image": 50}),
            (0.7, "0p7", {"video": 70, "image": 30}),
            (0.9, "0p9", {"video": 90, "image": 10}),
        )
        observed_experiments = []
        with patch.object(formal_benchmark, "split_pools", return_value=pools):
            for video_fraction, vtag, expected_counts in mix_points:
                cfg_p = formal_benchmark.serving(
                    proto,
                    model,
                    uuids,
                    bundle,
                    {},
                    manifests,
                    split="test",
                    seed=301,
                    rate_rpm=12.0,
                    video_fraction=video_fraction,
                    pattern="poisson",
                    suite="sweep",
                    slo_scale=1.5,
                    resolution_distribution="uniform",
                )
                exp_p = f"formal-wan22-test-poisson-s301-r12rpm-v{vtag}-slo1p5-sweep"
                assert cfg_p["experiment"] == exp_p
                assert cfg_p["benchmark"]["slo_scale"] == 1.5
                assert cfg_p["benchmark"]["video_fraction"] == video_fraction
                assert cfg_p["benchmark"]["absolute_rate"]["requested_rate_rpm"] == 12.0
                assert cfg_p["benchmark"]["trace"]["manifest"]["counts"] == expected_counts
                observed_experiments.append(cfg_p["experiment"])

                cfg_g = formal_benchmark.serving(
                    proto,
                    model,
                    uuids,
                    bundle,
                    {},
                    manifests,
                    split="test",
                    seed=301,
                    rate_rpm=12.0,
                    video_fraction=video_fraction,
                    pattern="gamma",
                    cv=2.0,
                    suite="sweep",
                    slo_scale=1.5,
                    resolution_distribution="uniform",
                )
                exp_g = f"formal-wan22-test-gamma-s301-r12rpm-v{vtag}-slo1p5-cv2p0-sweep"
                assert cfg_g["experiment"] == exp_g
                assert cfg_g["benchmark"]["slo_scale"] == 1.5
                assert cfg_g["benchmark"]["video_fraction"] == video_fraction
                assert cfg_g["benchmark"]["absolute_rate"]["requested_rate_rpm"] == 12.0
                assert cfg_g["benchmark"]["trace"]["manifest"]["counts"] == expected_counts
                observed_experiments.append(cfg_g["experiment"])

        assert len(observed_experiments) == 10
        assert len(set(observed_experiments)) == 10

    def test_formal_benchmark_serving_gpu_scalability_2_4_8_16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import patch

        import formal_benchmark
        import native_serving
        from ditserve.evaluation.common import canonical_hash, seal

        model = "wan22"
        uuids_16 = [f"GPU-00000000-0000-0000-0000-0000000004{i:02d}" for i in range(16)]
        uuids_8 = uuids_16[:8]
        proto = formal_benchmark.protocol(ROOT / "configs" / "benchmarks" / "formal-mixed-v2.json")

        manifests = {
            key: {
                "path": f"/shared/data/DITSERVE/models/{key}/manifest.json",
                "sha256": str(index + 1) * 64,
            }
            for index, key in enumerate((model, "sd35"))
        }
        classes = {}
        for sid, shape in proto["shapes"].items():
            if shape["scope"] != "formal" or shape["model"] not in (model, "sd35"):
                continue
            if shape["kind"] == "video":
                cost = {
                    "kind": "video",
                    "steps": shape["steps"],
                    "prepare_ms": 0.0,
                    "step_compute_ms": 100.0 * (shape["height"] / 480.0),
                    "finish_ms": 50.0,
                    "encode_ms": 10.0,
                    "links": {str(degree): {"gbps": 5.0, "latency_ms": 0.1} for degree in (2, 3, 4, 6, 8)},
                }
            else:
                cost = {"kind": "image", "service_ms": 250.0 * (shape["height"] / 512.0) ** 2}
            classes[sid] = {
                "spec": formal_benchmark.shape_spec(proto, sid),
                "cost": cost,
                "source": {
                    "gpu_uuids": list(uuids_8),
                    "resize_ms": 10.0,
                    "quantum_overhead_ms": 1.0,
                },
            }
        bundle = seal(
            {
                "schema": formal_benchmark.BUNDLE_SCHEMA,
                "protocol_sha256": canonical_hash(proto),
                "model": model,
                "model_manifests": manifests,
                "gpu_uuids": list(uuids_8),
                "classes": classes,
            }
        )
        pools = {
            kind: {
                "kind": kind,
                "sha256": str(index + 3) * 64,
                "samples": [
                    {"sample_id": f"{kind}-{sample}", "prompt": f"{kind} prompt {sample}"}
                    for sample in range(250)
                ],
            }
            for index, kind in enumerate(("image", "video"))
        }

        monkeypatch.setenv("DITSERVE_REMOTE_ENDPOINT", "127.0.0.1:29750")
        scale_points = (
            (2, 3.0, "-g2gpu"),
            (4, 6.0, "-g4gpu"),
            (8, 12.0, ""),
            (16, 24.0, "-g16gpu"),
        )
        observed_experiments = []
        with patch.object(formal_benchmark, "split_pools", return_value=pools):
            for num_gpus, rate_rpm, gpu_tag in scale_points:
                active_uuids = uuids_16[:num_gpus]
                cfg_p = formal_benchmark.serving(
                    proto,
                    model,
                    active_uuids,
                    bundle,
                    {},
                    manifests,
                    split="test",
                    seed=301,
                    rate_rpm=rate_rpm,
                    video_fraction=0.5,
                    pattern="poisson",
                    suite="scalability",
                    slo_scale=1.5,
                    resolution_distribution="uniform",
                )
                rtag = str(int(rate_rpm))
                assert cfg_p["experiment"] == f"formal-wan22-test-poisson-s301-r{rtag}rpm-v0p5-slo1p5{gpu_tag}-scalability"
                phase = cfg_p["phases"][0]
                assert len(phase["gpus"]) == num_gpus
                expected_counts = {2: 24, 4: 50, 8: 100, 16: 200}
                workload_rows = phase["matrix"]["/workload"][0]
                assert len(workload_rows) == expected_counts[num_gpus]
                assert set(phase["matrix"]["/policy"]) == {"ditserve", "edf-cfg2", "block-cfg2"}
                assert all(
                    native_serving.FORM_DEGREES[f] <= min(8, num_gpus)
                    for f in phase["options"]["forms"]
                )
                for idx in range(num_gpus):
                    wan_w = cfg_p["workers"][f"wan{idx}"]
                    img_w = cfg_p["workers"][f"image{idx}"]
                    if idx < 8:
                        assert "DITSERVE_REMOTE_WORKER" not in wan_w["environment"]
                        assert "script" not in wan_w
                    else:
                        assert wan_w["environment"]["DITSERVE_REMOTE_WORKER"] == "1"
                        assert wan_w["environment"]["DITSERVE_REMOTE_ENDPOINT"] == "127.0.0.1:29750"
                        assert wan_w["environment"]["DITSERVE_REMOTE_SLOT"] == str(idx)
                        assert wan_w["script"].endswith("remote_worker_proxy.py")
                        assert img_w["script"].endswith("remote_worker_proxy.py")
                observed_experiments.append(cfg_p["experiment"])

                cfg_g = formal_benchmark.serving(
                    proto,
                    model,
                    active_uuids,
                    bundle,
                    {},
                    manifests,
                    split="test",
                    seed=301,
                    rate_rpm=rate_rpm,
                    video_fraction=0.5,
                    pattern="gamma",
                    cv=2.0,
                    suite="scalability",
                    slo_scale=1.5,
                    resolution_distribution="uniform",
                )
                assert cfg_g["experiment"] == f"formal-wan22-test-gamma-s301-r{rtag}rpm-v0p5-slo1p5-cv2p0{gpu_tag}-scalability"
                observed_experiments.append(cfg_g["experiment"])

        assert len(observed_experiments) == 8
        assert len(set(observed_experiments)) == 8

    def test_remote_worker_daemon_and_proxy_end_to_end_tcp_rpc_and_doorbell(
        self, tmp_path: pathlib.Path
    ) -> None:
        import struct
        import threading
        import time

        from ditserve.runtime.remote_bridge import (
            RemoteWorkerDaemon,
            ping_daemon,
            query_remote_occupancy,
            release_daemon,
        )

        daemon = RemoteWorkerDaemon(
            host="127.0.0.1",
            port=0,
            coord_dir=tmp_path,
        )
        daemon.start_background()
        try:
            endpoint = daemon.endpoint
            assert (tmp_path / "node1.json").is_file()
            pong = ping_daemon(endpoint)
            assert pong["ok"] is True

            cfg_file = tmp_path / "mock_worker.json"
            cfg_file.write_text(json.dumps({"backend": "mock", "options": {"execution": "block"}}), encoding="utf-8")
            doorbell_file = tmp_path / "doorbell_gpu8.bin"
            doorbell_file.write_bytes(struct.pack("<Q", 0))

            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT / "workers"),
                DITSERVE_REMOTE_WORKER="1",
                DITSERVE_REMOTE_ENDPOINT=endpoint,
                DITSERVE_REMOTE_SLOT="8",
                DITSERVE_DOORBELL_FILE=str(doorbell_file),
                CUDA_VISIBLE_DEVICES="GPU-00000000-0000-0000-0000-000000000408",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OMP_NUM_THREADS="4",
            )
            proxy_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "ditserve.runtime.worker",
                    "--backend",
                    "mock",
                    "--config",
                    str(cfg_file),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                assert proxy_proc.stdin is not None and proxy_proc.stdout is not None
                proxy_proc.stdin.write(json.dumps({"version": 1, "id": 1, "op": "hello"}) + "\n")
                proxy_proc.stdin.flush()
                hello_reply = json.loads(proxy_proc.stdout.readline())
                assert hello_reply["ok"] is True
                assert hello_reply["result"]["backend"] == "mock"
                with daemon._lock:
                    active_sessions = list(daemon._sessions.values())
                assert len(active_sessions) == 1
                remote_pid = active_sessions[0].worker_pid
                proc_env_path = pathlib.Path(f"/proc/{remote_pid}/environ")
                if proc_env_path.is_file():
                    raw_env = proc_env_path.read_bytes().decode("utf-8", errors="replace")
                    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in raw_env
                    assert "OMP_NUM_THREADS=4" in raw_env

                # Ring doorbell on Node 0 and verify it mirrors to Node 1's local doorbell file in <50ms
                doorbell_file.write_bytes(struct.pack("<Q", 7))
                deadline = time.monotonic() + 1.0
                mirrored = 0
                while time.monotonic() < deadline:
                    with daemon._lock:
                        sessions = list(daemon._sessions.values())
                    if sessions and sessions[0].bell_path.is_file():
                        raw = sessions[0].bell_path.read_bytes()
                        if len(raw) == 8:
                            mirrored = struct.unpack("<Q", raw)[0]
                            if mirrored == 7:
                                break
                    time.sleep(0.002)
                assert mirrored == 7

                proxy_proc.stdin.write(json.dumps({"version": 1, "id": 2, "op": "shutdown"}) + "\n")
                proxy_proc.stdin.flush()
                shut_reply = json.loads(proxy_proc.stdout.readline())
                assert shut_reply["ok"] is True
                assert shut_reply["result"]["shutdown"] is True
                proxy_proc.wait(timeout=5)
            finally:
                if proxy_proc.poll() is None:
                    proxy_proc.kill()
        finally:
            release_daemon(daemon.endpoint)
            daemon.shutdown()

    def test_mixed_resolution_negative_optimization_guard_and_b3_retirement_pricing(self) -> None:
        profile = {
            "provenance": "synthetic",
            "links": {"2": {"gbps": 10.0, "latency_ms": 0.0}},
            "classes": {
                "video": {
                    "kind": "video",
                    "steps": 4,
                    "blocks_per_step": 4,
                    "prepare_ms": 5.0,
                    "encode_ms": 5.0,
                    "step_compute_ms": 80.0,
                    "comm_mb_per_step": 0.0,
                    "collectives_per_step": 0,
                    "finish_ms": 5.0,
                },
                "sd35_512": {
                    "kind": "image",
                    "service_ms": 1778.0,
                    "steps": 28,
                    "prepare_ms": 100.0,
                    "finish_ms": 75.0,
                    "base_pixels": 262144,
                    "batch_step_ms": {"1": 57.0, "2": 105.0, "4": 210.0},
                },
                "sd35_1024": {
                    "kind": "image",
                    "service_ms": 4879.0,
                    "steps": 28,
                    "prepare_ms": 100.0,
                    "finish_ms": 120.0,
                    "base_pixels": 1048576,
                    "batch_step_ms": {"1": 168.0, "2": 310.0, "4": 610.0},
                },
                "sd35_1440": {
                    "kind": "image",
                    "service_ms": 15058.0,
                    "steps": 28,
                    "prepare_ms": 100.0,
                    "finish_ms": 230.0,
                    "base_pixels": 2073600,
                    "batch_step_ms": {"1": 531.0, "2": 980.0, "4": 1960.0},
                },
            },
        }
        w512 = {
            "request_id": "i512",
            "kind": "image",
            "class": "sd35_512",
            "arrival_ms": 0.0,
            "deadline_ms": 20000.0,
            "continuous_batch": True,
            "spec": {"steps": 28, "height": 512, "width": 512},
        }
        w1024 = {
            "request_id": "i1024",
            "kind": "image",
            "class": "sd35_1024",
            "arrival_ms": 0.0,
            "deadline_ms": 21000.0,
            "continuous_batch": True,
            "spec": {"steps": 28, "height": 1024, "width": 1024},
        }
        w1440 = {
            "request_id": "i1440",
            "kind": "image",
            "class": "sd35_1440",
            "arrival_ms": 0.0,
            "deadline_ms": 30000.0,
            "continuous_batch": True,
            "spec": {"steps": 28, "height": 1440, "width": 1440},
        }

        base_opts = {"continuous_batch": 4, "sequence_packing": True, "bubble_free": True}
        # 1. 512^2 + 1024^2 (< 5x step dilation) co-batches under default guard (28 shared steps)
        sim_mod = _rpc_simulate(
            {"workload": [w512, w1024], "profile": profile, "policy": "ditserve", "gpus": 1, "options": base_opts}
        )
        assert sim_mod["batches"] == 28
        assert sim_mod["batched_images"] == 56
        assert sim_mod["metrics"]["deadline_attainment"] == 1.0

        # 2. 512^2 + 1440^2 (~10.3x step dilation) is blocked by default max_step_dilation=5.0:
        #    runs sequentially as 28 + 28 = 56 solo steps (mean batch 1.0), so i512 finishes < 2 s!
        sim_ext = _rpc_simulate(
            {"workload": [w512, w1440], "profile": profile, "policy": "ditserve", "gpus": 1, "options": base_opts}
        )
        by_id_ext = {r["request_id"]: r for r in sim_ext["requests"]}
        assert sim_ext["batches"] == 56
        assert sim_ext["batched_images"] == 56
        assert by_id_ext["i512"]["finish_ms"] < 2000.0
        assert sim_ext["metrics"]["deadline_attainment"] == 1.0

        # Disabling dilation cap (max_step_dilation=12.0) demonstrates the 9x+ latency blow-up on i512
        sim_uncapped = _rpc_simulate(
            {
                "workload": [w512, w1440],
                "profile": profile,
                "policy": "ditserve",
                "gpus": 1,
                "options": dict(base_opts, max_step_dilation=12.0),
            }
        )
        by_id_uncapped = {r["request_id"]: r for r in sim_uncapped["requests"]}
        assert sim_uncapped["batches"] == 28
        assert by_id_uncapped["i512"]["finish_ms"] > 15000.0

        # 3. B=3 multi-phase retirement simulation protects middle member rb even when max_step_dilation=12.0
        wa = dict(w512, request_id="ra", arrival_ms=0.0, deadline_ms=3500.0)
        wb = dict(w512, request_id="rb", arrival_ms=1000.0, deadline_ms=5200.0)
        wc = dict(w1440, request_id="rc", arrival_ms=1500.0, deadline_ms=30000.0)
        sim_b3 = _rpc_simulate(
            {
                "workload": [wa, wb, wc],
                "profile": profile,
                "policy": "ditserve",
                "gpus": 1,
                "options": dict(base_opts, max_step_dilation=12.0),
            }
        )
        assert sim_b3["metrics"]["deadline_attainment"] == 1.0
        by_id_b3 = {r["request_id"]: r for r in sim_b3["requests"]}
        assert by_id_b3["rb"]["finish_ms"] <= 5200.0

        # 4. In-flight piggyback asymmetry: when 1440^2 is ALREADY mid-execution (arrival 0 ms)
        #    and 512^2 arrives mid-run (arrival 5000 ms, deadline 19900 ms) on 1 GPU, piggybacking
        #    beats waiting serially behind 1440^2!
        w1440_first = dict(w1440, request_id="lead1440", arrival_ms=0.0, deadline_ms=25000.0)
        w512_mid = dict(w512, request_id="mid512", arrival_ms=5000.0, deadline_ms=19900.0)
        sim_piggy = _rpc_simulate(
            {
                "workload": [w1440_first, w512_mid],
                "profile": profile,
                "policy": "ditserve",
                "gpus": 1,
                "options": base_opts,
            }
        )
        assert sim_piggy["metrics"]["deadline_attainment"] == 1.0
        assert sim_piggy["batches"] < 56



