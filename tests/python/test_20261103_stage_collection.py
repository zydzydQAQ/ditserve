"""Collection preserves contract source hashes and cannot fill missing P0 cases."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import collect_stage_evidence
import stage_gates
from test_20261102_stage_gates import fixture, write


class StageCollectionTests(unittest.TestCase):
    def prepare(self, root):
        manifest = fixture(root)
        runs = [root / row["path"] for row in manifest.pop("runs")]
        manifest["runs"] = []
        result = stage_gates.evaluate(manifest, root)
        result["source_manifest"] = write(root, "contract-evidence.json", manifest)
        write(root, "contract-receipt.json", result)
        return root / "contract-receipt.json", runs

    def test_complete_and_incomplete_collections_recompute_eligibility(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract, runs = self.prepare(root)
            manifest, result = collect_stage_evidence.collect(contract, runs)
            self.assertEqual(result["stages"]["P0"]["status"], "passed")
            self.assertTrue(all(Path(r["path"]).is_absolute() for r in manifest["runs"]))
            incomplete, status = collect_stage_evidence.collect(contract, runs[:1])
            self.assertEqual(status["stages"]["P0"]["status"], "blocked")
            self.assertEqual(status["stages"]["P1"]["status"], "pending")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                collect_stage_evidence.collect(contract, runs + runs[:1])

    def test_contract_tampering_and_unqualified_runs_are_not_promoted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract, runs = self.prepare(root)
            data = json.loads(runs[0].read_text())
            for row in data["requests"]:
                row["outputs"][-1]["generation"]["model_integrity"] = "not_verified"
            runs[0].write_text(json.dumps(data))
            _, result = collect_stage_evidence.collect(contract, runs)
            self.assertEqual(result["stages"]["P0"]["status"], "blocked")
            (root / "contract.log").write_text("CUDA passed only\n")
            with self.assertRaises(ValueError):
                collect_stage_evidence.collect(contract, runs)


if __name__ == "__main__":
    unittest.main()
