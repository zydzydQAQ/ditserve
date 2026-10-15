import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve.benchmark.backend_quality import check_backend_quality
from ditserve.benchmark.common import atomic_json, file_hash, seal
from test_benchmark_runner import request


class BackendQualityTests(unittest.TestCase):
    def fixture(self, root, attention="MATH"):
        import numpy as np
        output = root / "output.npy"
        np.save(output, np.zeros((1, 1, 2, 2, 3), dtype=np.float32))
        artifact = dict(output_path=str(output), output_sha256=file_hash(output),
                        generation=request("r", "long", 0, 1000))
        identity = dict(gpu=dict(uuid="test-gpu"), precision=dict(dit="float16"),
                        attention=dict(sdpa_backend=attention))
        run = seal(dict(backend="wan", measured=True, status="completed",
                        execution_identity=identity, references={"r": artifact}))
        atomic_json(root / "run.json", run)
        return {"r": artifact}, identity

    def test_flash_cannot_be_used_as_its_own_math_quality_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            references, identity = self.fixture(root, "FLASH_ATTENTION")
            with self.assertRaisesRegex(ValueError, "MATH backend"):
                check_backend_quality(references, root, root, candidate_identity=identity)

    def test_missing_lpips_never_passes_and_gpu_identity_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            references, identity = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "GPU and numerical recipe"):
                check_backend_quality(references, root, root, candidate_identity=dict(identity, gpu=dict(uuid="other")))
            with patch("ditserve.gpu.quality.evaluate_quality", return_value={"status": "unknown", "lpips_status": "unavailable"}):
                gate = check_backend_quality(references, root, root, candidate_identity=identity)
            self.assertEqual(gate["status"], "failed")

    def test_gate_binds_raw_evidence_and_applies_quality_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            references, identity = self.fixture(root)
            metrics = dict(status="ok", lpips_status="ok", finite=True, range_valid=True, lpips_mean=.049)
            with patch("ditserve.gpu.quality.evaluate_quality", return_value=metrics):
                gate = check_backend_quality(references, root, root, candidate_identity=identity)
            self.assertEqual(gate["status"], "passed")
            self.assertEqual(gate["requests"]["r"]["candidate_sha256"], references["r"]["output_sha256"])
            metrics["lpips_mean"] = .051
            with patch("ditserve.gpu.quality.evaluate_quality", return_value=metrics):
                self.assertEqual(check_backend_quality(references, root, root, candidate_identity=identity)["status"], "failed")
            (root / "output.npy").write_bytes(b"modified")
            with self.assertRaises((ValueError, LookupError)):
                check_backend_quality(references, root, root, candidate_identity=identity)


if __name__ == "__main__":
    unittest.main()
