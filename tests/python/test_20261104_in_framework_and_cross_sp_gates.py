#!/usr/bin/env python3
"""Unit tests for Fix 1 (In-framework Policy Comparison) and Fix 2 (Relaxed Cross-SP Acceptance Gates)."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "benchmarks", ROOT / "workers", ROOT / "tests" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import challenge1_contrasts as c1
import external_sd35_cross_sp_check as sd35_cross
import external_tetriserve_service_numerics as tetri_num
import external_tetriserve_sp2_diagnostic as tetri_sp2
import external_wan_topology_step as wan_topo
import external_xdit_numerics as xdit_num
import formal_benchmark as formal
import native_serving as nserv
import paper_results
from test_20261104_external_sd35_cross_sp import pair as _cross_pair
from test_20261104_external_tetriserve_sp2_diagnostic import receipt as _sp2_receipt
from test_20261104_external_wan_topology_step import rows as _wan_rows


class InFrameworkAndCrossSPGatesTest(unittest.TestCase):
    def test_fix1_in_framework_policy_suite_and_declaration(self) -> None:
        expected = [
            "static-cfg2sp4",
            "tetriserve",
            "sbp-esp",
            "edf-bestfit",
            "ditserve",
        ]
        self.assertEqual(nserv.SUITES["in-framework-main"], expected)
        self.assertEqual(nserv.in_framework_main_policies(), expected)
        self.assertEqual(
            nserv.in_framework_main_policies("static-cfg2sp3")[0],
            "static-cfg2sp3",
        )
        with self.assertRaises(ValueError):
            nserv.in_framework_main_policies("invalid-static-form")

        self.assertEqual(nserv.IN_FRAMEWORK_MECHANISM_DISPLAY["tetriserve"], "TetriServe-policy")
        self.assertEqual(nserv.IN_FRAMEWORK_MECHANISM_DISPLAY["edf-bestfit"], "EDF-BF")
        self.assertEqual(nserv.IN_FRAMEWORK_MECHANISM_DISPLAY["static-cfg2sp4"], "Static-CFG2SP4")
        self.assertEqual(paper_results.mechanism_label("tetriserve"), "TetriServe-policy")
        self.assertEqual(paper_results.mechanism_label("static-cfg2sp4"), "Static-CFG2SP4")
        self.assertEqual(list(c1.IN_FRAMEWORK_MAIN_ARMS), expected)
        for arm in expected:
            self.assertIn(arm, c1.ARM_SPECS)

    def test_fix1_formal_v2_and_external_reproduction_contracts_align(self) -> None:
        v2 = formal.protocol(formal.DEFAULT_V2_PROTOCOL)
        ext = json.loads((ROOT / "configs/benchmarks/external-reproduction-v1.json").read_text(encoding="utf-8"))

        v2_base = v2["baseline_comparison_contract"]
        self.assertEqual(v2_base["mode"], "in_framework_policy_comparison")
        self.assertTrue(v2_base["shared_executor"])
        self.assertEqual(v2_base["main_table_policies"], nserv.SUITES["in-framework-main"])
        self.assertEqual(v2_base["mechanism_labels"]["tetriserve"], "TetriServe-policy")
        self.assertIn("tetriserve", v2_base["external_native_intersection"])
        self.assertIn("xdit", v2_base["external_native_intersection"])

        tracks = ext["baseline_comparison_tracks"]
        self.assertIn("in_framework_policy_comparison", tracks)
        self.assertIn("external_system_native_intersection", tracks)
        self.assertEqual(
            tracks["in_framework_policy_comparison"]["policies"],
            ["Static-<Form>", "TetriServe-policy", "SBP-ESP", "EDF-BF", "DitServe"],
        )

        hierarchy = ext["quality_and_numerical_verification"]["acceptance_gate_hierarchy"]
        self.assertEqual(
            hierarchy["same_parallelism_segmentation_gate"]["atol"],
            1e-5,
        )
        self.assertEqual(
            hierarchy["cross_sp_acceptance_gate"]["mode"],
            "semantic_quality_equivalence",
        )
        self.assertTrue(
            hierarchy["cross_sp_acceptance_gate"]["strict_elementwise_1e5_is_diagnostic_only"],
        )

    def test_fix2_xdit_cross_sp_decoded_numerics_passes_semantic_gate_despite_bf16_drift(self) -> None:
        from unittest.mock import patch
        with TemporaryDirectory() as tmpdir, patch.object(xdit_num, "check_storage", return_value={"used_gib": 1, "percent": 1}):
            root = Path(tmpdir)
            shape = [2, 4, 4, 3]
            rng = np.random.default_rng(42)
            sp1 = rng.uniform(0.1, 0.9, size=shape).astype(np.float32)
            # Simulate realistic BF16 Ulysses SP reduction-order drift (~2e-3 > 1e-5)
            sp4 = (sp1 + rng.normal(0.0, 1.5e-3, size=shape)).clip(0.0, 1.0).astype(np.float32)
            prefix = json.dumps({"shape": shape, "dtype": "float32"}, sort_keys=True).encode()
            paths, expected = [], []
            for name, arr in (("sp1.raw", sp1), ("sp4.raw", sp4)):
                p = root / name
                raw = arr.tobytes()
                p.write_bytes(raw)
                paths.append(p)
                expected.append({
                    "file_sha256": hashlib.sha256(raw).hexdigest(),
                    "native_output_sha256": hashlib.sha256(prefix + raw).hexdigest(),
                })
            metrics = xdit_num.compare_files(paths[0], paths[1], shape, expected)
            self.assertFalse(metrics["strict_decoded_consistency_passed"])
            self.assertGreater(metrics["failed_elements"], 0)
            self.assertTrue(metrics["cross_sp_quality_gate_passed"])
            self.assertEqual(metrics["acceptance_gate_mode"], "semantic_quality_equivalence")
            self.assertTrue(metrics["strict_elementwise_1e5_is_diagnostic_only"])
            self.assertGreater(metrics["psnr_db"], 40.0)
            self.assertGreater(metrics["cosine_similarity"], 0.999)
            self.assertLess(metrics["relative_l2_error"], 0.01)

    def test_fix2_tetriserve_service_and_sp2_diagnostic_two_tier_gates(self) -> None:
        # 1) Service numerics: request 100 on SP1 (strict 1e-5) + request 101 on SP2 (cross-SP drift)
        ref = np.linspace(0.1, 1.0, 64, dtype=np.float32)
        same_sp1 = ref.copy()
        cross_sp2 = ref + np.float32(5e-3)
        c100 = tetri_num.compare_arrays(ref, same_sp1)
        c101 = tetri_num.compare_arrays(ref, cross_sp2)
        self.assertTrue(c100["passed"])
        self.assertFalse(c101["passed"])
        self.assertTrue(c101["cross_sp_quality_gate_passed"])

        two_tier = tetri_num.evaluate_two_tier_service_gate(
            {100: {"latents": c100, "decoded": c100}, 101: {"latents": c101, "decoded": c101}},
            {100: [0], 101: [0, 1]},
        )
        self.assertTrue(two_tier["same_sp_segmentation_gate_passed"])
        self.assertTrue(two_tier["cross_sp_quality_gate_passed"])
        self.assertTrue(two_tier["two_tier_acceptance_passed"])

        # If a same-SP1 request drifts beyond 1e-5, Tier 1 must fail!
        bad_same = tetri_num.evaluate_two_tier_service_gate(
            {100: {"latents": c101, "decoded": c101}, 101: {"latents": c101, "decoded": c101}},
            {100: [0], 101: [0, 1]},
        )
        self.assertFalse(bad_same["same_sp_segmentation_gate_passed"])
        self.assertFalse(bad_same["two_tier_acceptance_passed"])

        # 2) SP2 diagnostic: round_sp2_full vs round_sp2_chunked is bitwise equal, while sp1_reference differs
        ranks = _sp2_receipt()
        ranks[0]["comparison_inputs"]["sp1_reference"]["latents"]["sha256"] = "b" * 64
        ranks[0]["comparison_inputs"]["sp1_reference"]["decoded"]["sha256"] = "c" * 64
        for pair_name in tetri_sp2.PAIRS:
            if pair_name.startswith("sp1_reference_"):
                for tensor_key in ("latents", "decoded"):
                    ranks[0]["comparisons"][pair_name][tensor_key].update(
                        passed=False,
                        bitwise_equal=False,
                        max_abs_error=1.25,
                        rmse=0.12,
                        relative_l2_error=0.11,
                        cosine_similarity=0.985,
                    )
        gates = tetri_sp2.assess_two_tier_gates(ranks)
        self.assertTrue(gates["same_sp_segmentation_gate_passed"])
        self.assertTrue(gates["cross_sp_semantic_gate_passed"])
        self.assertTrue(gates["two_tier_acceptance_passed"])

    def test_fix2_sd35_cross_sp_and_wan_topology_gates(self) -> None:
        same_pair = _cross_pair()
        cross_pair = _cross_pair()
        for b in cross_pair["boundaries"].values():
            b["actual"]["sha256"] = "b" * 64
            b.update(
                passed=False,
                bitwise_equal=False,
                max_abs_error=0.08,
                rmse=0.01,
                relative_l2_error=0.02,
                cosine_similarity=0.998,
            )
        cross_pair.update(sd35_cross.first_differences(cross_pair["boundaries"]))

        same_eval = sd35_cross.evaluate_cross_sp_step_gate(same_pair, pair_name="original_sp1_vs_converted_sp1")
        cross_eval = sd35_cross.evaluate_cross_sp_step_gate(cross_pair, pair_name="sp1_vs_sp2")
        self.assertTrue(same_eval["passed"])
        self.assertTrue(cross_eval["passed"])
        self.assertFalse(cross_eval["strict_elementwise_1e5_passed"])
        self.assertTrue(cross_eval["cross_sp_quality_gate_passed"])

        # Wan topology step summary includes cross_sp_acceptance_gate
        r2 = _wan_rows(2)
        summary = wan_topo.aggregate(r2, r2[0]["binding"], 2, {"pin": "immutable"}, "a" * 64)
        self.assertIn("cross_sp_acceptance_gate", summary)
        self.assertEqual(
            summary["cross_sp_acceptance_gate"]["mode"],
            "semantic_quality_equivalence",
        )
        self.assertTrue(summary["cross_sp_acceptance_gate"]["strict_elementwise_1e5_is_diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
