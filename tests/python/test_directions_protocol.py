import copy
from pathlib import Path
import tempfile
import unittest

from ditserve.directions.protocol import load_config, freeze_config, freeze_phase
from ditserve.research.common import file_digest


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_immutable_within_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            first = freeze_config(config, directory)
            self.assertEqual(first, freeze_config(config, directory))
            changed = copy.deepcopy(config)
            changed["direction1"]["steps"] = 4
            with self.assertRaises(ValueError):
                freeze_config(changed, directory)

    def test_order_and_new_budget_match_authorization(self):
        config = load_config()
        self.assertEqual(config["order"], [1, 2, 3])
        self.assertEqual(sum(config["stage_budgets_gpu_seconds"].values()), 86400)
        self.assertEqual(config["budget_gpu_seconds"], 86400)

    def test_executable_snapshot_includes_actual_input_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            record = freeze_phase(directory, 2, "screen", load_config())
            source = Path(record["execution_root"])
            self.assertIn("configs/research_cfg.json", record["source_files"])
            for name, expected in record["source_files"].items():
                self.assertEqual(file_digest(source / name), expected)


if __name__ == "__main__":
    unittest.main()
