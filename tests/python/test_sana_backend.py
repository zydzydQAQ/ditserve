"""Sana request lifecycle against a stand-in adapter; no torch or weights."""

import tempfile
import unittest
from unittest.mock import patch

from ditserve.runtime.worker import Worker


class FakeAdapter:
    def __init__(self, path, device):
        self.calls, self.closed = [], False

    def generate(self, prompt, *, seed=0):
        self.calls.append((prompt, seed))
        return [[[float(seed)]]], dict(ms=7.5, finite=True)

    def close(self):
        self.closed = True


class SanaBackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        patcher = patch("ditserve.models.sana.runtime.SanaAdapter", FakeAdapter)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.worker = Worker("sana", dict(model_path="/models/sana", work_dir=self.directory.name))
        self.sequence = 0

    def call(self, op, **fields):
        self.sequence += 1
        return self.worker.dispatch(dict(version=1, id=self.sequence, op=op, **fields))

    def test_one_pipeline_call_per_request(self):
        self.call("prepare", state_id="a", spec=dict(prompt="cat", seed=3))
        with self.assertRaisesRegex(ValueError, "completed model work"):
            self.call("finish", state_id="a")
        result = self.call("advance", state_id="a")
        self.assertTrue(result["progress"]["done"])
        self.assertEqual(result["model_ms"], 7.5)
        with self.assertRaisesRegex(ValueError, "no remaining"):
            self.call("advance", state_id="a")
        output = self.call("finish", state_id="a", output_mode="check")["output"]
        self.assertEqual((output["shape"], output["output_mode"]), ([1, 1, 1], "check"))
        self.call("release", state_id="a")
        self.assertEqual(self.worker.backend.runtime.calls, [("cat", 3)])
        self.call("shutdown")
        self.assertTrue(self.worker.backend.runtime.closed)

    def test_rejects_options_and_malformed_specs(self):
        with self.assertRaisesRegex(ValueError, "takes no options"):
            Worker("sana", dict(model_path="/m", options=dict(candidate="eager"))).dispatch(
                dict(version=1, id=1, op="prepare", state_id="a", spec=dict(prompt="x")))
        with self.assertRaisesRegex(ValueError, "string prompt"):
            self.call("prepare", state_id="b", spec=dict(prompt=3))


if __name__ == "__main__":
    unittest.main()
