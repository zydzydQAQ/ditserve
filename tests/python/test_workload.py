"""Workload payload and trace-loading tests."""

import json
import tempfile
import unittest
from pathlib import Path

from ditserve.workload import generator_payload, load_trace


class GeneratorPayloadTest(unittest.TestCase):
    def test_defaults_match_schema_v2(self):
        payload = generator_payload()
        self.assertEqual(payload["count"], 64)
        self.assertEqual(payload["pattern"], "steady")
        self.assertEqual(payload["shapes"], [[512, 512, 1], [768, 768, 1]])
        self.assertEqual(payload["step_choices"], [20, 30])

    def test_shapes_are_plain_lists(self):
        payload = generator_payload(shapes=((320, 320, 1),))
        self.assertIsInstance(payload["shapes"][0], list)

    def test_unknown_pattern_rejected(self):
        with self.assertRaises(ValueError):
            generator_payload(pattern="random")


class LoadTraceTest(unittest.TestCase):
    def test_roundtrip(self):
        specs = [{"request_id": "a", "arrival_ms": 0, "deadline_ms": 100}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(specs), encoding="utf-8")
            self.assertEqual(load_trace(path), specs)

    def test_non_array_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text('{"request_id": "a"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "array"):
                load_trace(path)


if __name__ == "__main__":
    unittest.main()
