import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import paper_results


def record():
    cells, requests = [], []
    for index, (policy, video_finish, image_finish) in enumerate((("edf-cfg2", 50, 60), ("block-cfg2", 70, 80))):
        cells.append({"cell": index, "policy": policy, "status": "completed", "elapsed_ms": 1000, "gpus": 8})
        requests.extend([
            {"cell": index, "kind": "video", "status": "completed", "arrival_ms": 0,
             "deadline_ms": 100, "finish_ms": video_finish, "preemption_stop_samples": []},
            {"cell": index, "kind": "image", "status": "completed", "arrival_ms": 0,
             "deadline_ms": 100, "finish_ms": image_finish},
        ])
    return {
        "schema": "ditserve.run.v1", "experiment": "fixture", "run_id": "r1", "status": "completed",
        "config": {"benchmark": {"native_serving": True, "models": ["Wan", "SD3.5"], "seed": 201}},
        "config_sha256": "c" * 64, "source": {"core_sha256": "a" * 64},
        "phases": [{"cells": cells}], "requests": requests, "metrics": {}, "budget": {"used_gpu_seconds": 16},
    }


class PaperResultsTest(unittest.TestCase):
    def manifest(self, root, run_record):
        (root / "run.json").write_text(json.dumps(run_record), encoding="utf-8")
        value = {"schema": "ditserve.paper-input.v1", "rows": [
            {"label": "No block", "run": "run.json", "policy": "edf-cfg2"},
            {"label": "Block", "run": "run.json", "policy": "block-cfg2"},
        ]}
        path = root / "manifest.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_each_row_uses_one_complete_policy_cell(self):
        with TemporaryDirectory() as directory:
            path = self.manifest(Path(directory), record())
            result = paper_results.build(json.loads(path.read_text()), path)
            self.assertEqual(len(result["rows"]), 2)
            no_block, block = result["rows"]
            self.assertEqual(no_block["source"]["policy"], "edf-cfg2")
            self.assertEqual(no_block["metrics"]["by_kind"]["video"]["latency_ms"]["p50"], 50)
            self.assertEqual(block["metrics"]["by_kind"]["video"]["latency_ms"]["p50"], 70)
            self.assertEqual(no_block["metrics"]["gpu_seconds_per_ontime_request"], 4)
            self.assertIn("0.07 / 0.07", block["tex"])
            self.assertNotIn("0.05 / 0.05", block["tex"])

    def test_rejects_unmeasured_incomplete_or_ambiguous_sources(self):
        for mutate in (
            lambda value: value.update(status="failed"),
            lambda value: value["config"]["benchmark"].update(native_serving=False),
            lambda value: value.pop("source"),
        ):
            changed = record()
            mutate(changed)
            with TemporaryDirectory() as directory, self.assertRaises(ValueError):
                path = self.manifest(Path(directory), changed)
                paper_results.build(json.loads(path.read_text()), path)
        changed = record()
        changed["phases"][0]["cells"].append(copy.deepcopy(changed["phases"][0]["cells"][0]))
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "exactly one"):
            path = self.manifest(Path(directory), changed)
            paper_results.build(json.loads(path.read_text()), path)

    def test_cli_writes_json_and_tex_without_overwriting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.manifest(root, record())
            argv = ["paper_results.py", "--manifest", str(path), "--output", str(root / "rows.json"),
                    "--tex-output", str(root / "rows.tex")]
            with patch.object(sys, "argv", argv):
                paper_results.main()
            self.assertEqual(len(json.loads((root / "rows.json").read_text())["rows"]), 2)
            self.assertIn("one run/policy cell per row", (root / "rows.tex").read_text())
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                paper_results.main()


if __name__ == "__main__":
    unittest.main()
