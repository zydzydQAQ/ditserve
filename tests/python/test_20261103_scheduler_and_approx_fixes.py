"""Regression tests for 2026-11-03 scheduler, CFG-reuse online drift, and VBench receipt binding fixes."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "benchmarks", ROOT / "workers"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    import torch as _REAL_TORCH
except ImportError:
    _REAL_TORCH = None

import p4_stream_quality
from ditserve.evaluation.common import seal
from ditserve.models.wan.cfg_reuse import AsymmetricCFGStepper


class _FakeTensor:
    def __init__(self, arr, dtype="float32"):
        self._arr = np.asarray(arr, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self):
        return self._arr.shape

    def numel(self):
        return int(self._arr.size)

    def element_size(self):
        return 4

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, dtype):
        return _FakeTensor(self._arr.copy(), dtype=dtype)

    def clone(self):
        return _FakeTensor(self._arr.copy(), dtype=self.dtype)

    def flatten(self):
        return _FakeTensor(self._arr.reshape(-1), dtype=self.dtype)

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self._arr, axis=dim), dtype=self.dtype)

    def clamp(self, min=None, max=None):
        return _FakeTensor(np.clip(self._arr, min, max), dtype=self.dtype)

    def item(self):
        return float(self._arr.reshape(-1)[0])

    def __add__(self, other):
        other_arr = other._arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._arr + other_arr, dtype=self.dtype)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_arr = other._arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._arr - other_arr, dtype=self.dtype)

    def __mul__(self, other):
        other_arr = other._arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._arr * other_arr, dtype=self.dtype)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_arr = other._arr if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self._arr / other_arr, dtype=self.dtype)

    def __neg__(self):
        return _FakeTensor(-self._arr, dtype=self.dtype)


def _make_torch_backend():
    if _REAL_TORCH is not None:
        return _REAL_TORCH

    def vector_norm(tensor, ord=2):
        flat = tensor._arr.reshape(-1)
        if ord == 1:
            val = float(np.sum(np.abs(flat)))
        else:
            val = float(np.linalg.norm(flat, ord=ord))
        return _FakeTensor([val])

    def cosine_similarity(a, b):
        va = a._arr.reshape(-1)
        vb = b._arr.reshape(-1)
        denom = max(1e-12, float(np.linalg.norm(va) * np.linalg.norm(vb)))
        return _FakeTensor([float(np.dot(va, vb)) / denom])

    def clamp(tensor, min=None, max=None):
        return tensor.clamp(min=min, max=max)

    return SimpleNamespace(
        float32="float32",
        ones=lambda shape, dtype="float32": _FakeTensor(np.ones(shape, dtype=np.float32), dtype=dtype),
        zeros=lambda shape, dtype="float32": _FakeTensor(np.zeros(shape, dtype=np.float32), dtype=dtype),
        allclose=lambda a, b: bool(np.allclose(a._arr, b._arr)),
        clamp=clamp,
        linalg=SimpleNamespace(vector_norm=vector_norm),
        nn=SimpleNamespace(functional=SimpleNamespace(cosine_similarity=cosine_similarity)),
    )


class SchedulerAndApproxFixesTests(unittest.TestCase):
    def setUp(self):
        self.torch = _make_torch_backend()

    def test_adaptive_cfg_stepper_updates_online_curvature_and_avoids_fixed_k_collapse(self):
        stepper = AsymmetricCFGStepper(
            self.torch,
            folded_steps=(),
            curvature_threshold=0.05,
            min_full_cfg_steps=12,
            max_folded_steps=6,
            final_step_exact_anchor=True,
        )
        base_cond = self.torch.ones((1, 4, 4), dtype=self.torch.float32)
        base_uncond = self.torch.zeros((1, 4, 4), dtype=self.torch.float32)

        # Warmup steps 0..11 with smooth trajectory -> low curvature (~0.014 <= 0.05)
        for s in range(12):
            scale = 0.96 ** s
            stepper.observe_cfg_pair(s, base_cond * scale, base_uncond)

        self.assertTrue(stepper.should_fold(12, 20))
        # Step 12 folds smoothly
        _ = stepper.fold_prediction(12, base_cond * (0.96 ** 12), 5.0)
        # Step 13 encounters a sharp conditional-branch direction flip (online drift spike)
        flipped_cond = (-base_cond) * (0.96 ** 13)
        _ = stepper.fold_prediction(13, flipped_cond, 5.0)

        # Because step 13 had high conditional drift (> 0.05), step 14 must NOT fold;
        # it forces an exact 2-branch CFG refresh instead of blindly folding like fixed-K.
        self.assertGreater(stepper._last_curvature, 0.05)
        self.assertFalse(stepper.should_fold(14, 20))

        # After step 14 refreshes the exact CFG pair and step 15 stabilizes, folding resumes.
        stepper.observe_cfg_pair(14, flipped_cond * 0.96, base_uncond)
        stepper.observe_cfg_pair(15, flipped_cond * (0.96 ** 2), base_uncond)
        self.assertLessEqual(stepper._last_curvature, 0.05)
        self.assertTrue(stepper.should_fold(16, 20))

    def test_adaptive_cfg_stepper_export_import_preserves_online_drift_state(self):
        options = dict(
            folded_steps=(),
            curvature_threshold=0.05,
            min_full_cfg_steps=2,
            max_folded_steps=4,
            final_step_exact_anchor=True,
        )
        src = AsymmetricCFGStepper(self.torch, **options)
        cond = self.torch.ones((1, 4, 4), dtype=self.torch.float32)
        uncond = self.torch.zeros((1, 4, 4), dtype=self.torch.float32)
        src.observe_cfg_pair(0, cond, uncond)
        src.observe_cfg_pair(1, cond * 0.97, uncond)
        self.assertTrue(src.should_fold(2, 10))
        _ = src.fold_prediction(2, cond * 0.94, 5.0)

        metadata, tensors = src.export_state()
        self.assertEqual(metadata["runtime"]["consecutive_folds"], 1)
        self.assertIn("last_cond", tensors)

        dst = AsymmetricCFGStepper(self.torch, **options)
        dst.import_state(metadata, tensors)
        self.assertEqual(dst._consecutive_folds, 1)
        self.assertAlmostEqual(dst._last_curvature, src._last_curvature, places=6)
        self.assertTrue(self.torch.allclose(dst._last_cond_f32, src._last_cond_f32))

    def test_p4_stream_quality_validates_and_binds_optional_vbench_receipt(self):
        vbench_receipt = seal({
            "schema_version": "ditserve.benchmark.vbench_evaluation.v1",
            "status": "completed",
            "verified": True,
            "manifest_sha256": "a" * 64,
            "repository": "https://github.com/Vchitect/VBench",
            "revision": p4_stream_quality.VBENCH_REVISION,
            "scope": "pinned official VBench evaluation",
            "total_score": None,
            "dimensions": {
                "subject_consistency": {"score": 0.9412, "video_count": 16, "mode": "vbench_standard"},
                "temporal_flickering": {"score": 0.9825, "video_count": 16, "mode": "custom_input"},
            },
            "reason": None,
        })
        state = p4_stream_quality.validate_vbench_receipt(vbench_receipt)
        self.assertEqual(state["status"], "measured")
        self.assertEqual(state["receipt_sha256"], vbench_receipt["sha256"])
        self.assertAlmostEqual(state["dimensions"]["subject_consistency"]["score"], 0.9412)

        # Omitted receipt stays 'unknown'
        unknown = p4_stream_quality.validate_vbench_receipt(None)
        self.assertEqual(unknown["status"], "unknown")

        # Tampered or unverified receipt raises ValueError
        tampered = copy.deepcopy(vbench_receipt)
        tampered["dimensions"]["subject_consistency"]["score"] = 0.9999
        with self.assertRaisesRegex(ValueError, "sealed object hash mismatch"):
            p4_stream_quality.validate_vbench_receipt(tampered)

        unverified = seal({**{k: v for k, v in vbench_receipt.items() if k != "sha256"}, "verified": False})
        with self.assertRaisesRegex(ValueError, "not verified"):
            p4_stream_quality.validate_vbench_receipt(unverified)


import test_study as fixtures
from ditserve.runtime.client import CoreBridge


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class PreAdvanceDoorbellStopTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeStudyTest.setUpClass.__func__)
    setUp = fixtures.NativeStudyTest.setUp
    config = fixtures.NativeStudyTest.config
    cluster_phase = fixtures.NativeStudyTest.cluster_phase
    cluster_workers = fixtures.NativeStudyTest.cluster_workers
    run_study = fixtures.NativeStudyTest.run_study
    assert_completed = fixtures.NativeStudyTest.assert_completed

    def test_doorbell_rung_during_initial_staging_skips_advance_before_block_execution(self):
        phase = self.cluster_phase(
            "block-cfg2",
            gpus=2,
            text_prefetch=0,
            options=dict(
                sp_transport="export",
                doorbell=True,
                lazy_preemption=False,
                quantum_blocks=1,
                transfer_dir=str(self.root / "shm-pre-advance"),
            ),
        )
        phase["workload"] = [
            dict(
                request_id="v0",
                kind="video",
                class_name="video",
                worker="v0",
                arrival_ms=0,
                slo_ms=20000,
                spec=dict(steps=2, blocks_per_step=4),
            ),
            dict(
                request_id="i0",
                kind="image",
                class_name="image",
                worker="i0",
                arrival_ms=30,
                slo_ms=90,
                spec=dict(steps=1),
            ),
        ]
        for row in phase["workload"]:
            row["class"] = row.pop("class_name")
        workers = self.cluster_workers(2)
        for g in range(2):
            workers["v%d" % g]["config"]["options"]["costs"] = dict(prepare_ms=140, step_ms=400, finish_ms=20)
        run = self.run_study(self.config([phase], workers=workers), name="pre-advance-doorbell")
        self.assert_completed(run)
        cell = run["phases"][0]["cells"][0]
        v0 = next(r for r in cell["requests"] if r["request_id"] == "v0")
        stages = [
            stage["stage"]
            for event in v0.get("controller_timing", [])
            for stage in event.get("stages", [])
        ]
        self.assertIn("advances_skipped_doorbell", stages)
        self.assertTrue(v0.get("preemption_stop_samples"))


if __name__ == "__main__":
    unittest.main()
