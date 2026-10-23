"""CPU subprocess coverage of native encoder telemetry, never model evidence."""

import json
import unittest

import test_study as study
from ditserve.runtime.client import CoreBridge


# Only the test's temporary frozen package is changed. Large sentinel timings
# ensure telemetry does not silently add encoding to the existing model_ms.
FIXTURE = r'''
_original_prepare = MockBackend.prepare
_original_encode_start = MockBackend.encode_start
_original_encode_poll = MockBackend.encode_poll

def _encoding_fixture(self, request):
    self.fixture_sequence = getattr(self, "fixture_sequence", 0) + 1
    spec = request["spec"]
    receipt = dict(schema="cpu-telemetry-fixture", sequence=self.fixture_sequence,
                   request_id=spec["request_id"], nested=dict(kept=[1, None, "opaque"]))
    if self.config.get("options", {}).get("disabled_cache"):
        receipt = None
    return dict(negative_prompt_cache=receipt, cpu_prompt_encode_ms=1000. + self.fixture_sequence)

def _prepare_with_encoding(self, request):
    ticket = request.get("encode_ticket")
    value = (self.fixture_tickets.pop(ticket) if ticket else _encoding_fixture(self, request))
    result = _original_prepare(self, request)
    result.update(value)
    return result

def _start_with_encoding(self, request):
    result = _original_encode_start(self, request)
    if not hasattr(self, "fixture_tickets"):
        self.fixture_tickets = {}
    self.fixture_tickets[result["ticket"]] = _encoding_fixture(self, request)
    return result

def _poll_with_encoding(self, request):
    result = _original_encode_poll(self, request)
    if result["ready"]:
        value = self.fixture_tickets[request["ticket"]]
        result["model_ms"] = value["cpu_prompt_encode_ms"]
        if value["negative_prompt_cache"] is not None:
            result["negative_prompt_cache"] = value["negative_prompt_cache"]
    return result

MockBackend.prepare = _prepare_with_encoding
MockBackend.encode_start = _start_with_encoding
MockBackend.encode_poll = _poll_with_encoding
'''


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class NativeEncodingTelemetryTest(unittest.TestCase):
    PROFILE = study.NativeStudyTest.PROFILE
    setUp = study.NativeStudyTest.setUp
    request = staticmethod(study.NativeStudyTest.request)
    phase = study.NativeStudyTest.phase
    config = study.NativeStudyTest.config
    run_study = study.NativeStudyTest.run_study
    assert_completed = study.NativeStudyTest.assert_completed
    assert_archive = study.NativeStudyTest.assert_archive

    @classmethod
    def setUpClass(cls):
        study.NativeStudyTest.setUpClass.__func__(cls)
        path = cls.package_root / "ditserve/models/mock.py"
        path.write_text(path.read_text() + FIXTURE)

    def execute(self, depth, disabled=False):
        phase = self.phase(text_prefetch_depth=depth,
            warmup=[dict(worker="model", spec=dict(steps=2))],
            workload=[self.request("a"), self.request("b")])
        result = self.run_study(self.config([phase], options=dict(disabled_cache=disabled)))
        self.assert_completed(result)
        self.assertEqual(result["metrics"]["completed"], 2)
        return result

    def check_event(self, event, row, op, receipt, duration):
        self.assertEqual(event["op"], op)
        self.assertEqual(event["worker"], "model")
        self.assertEqual(event["pid"], row["worker_pids"]["model"])
        self.assertEqual(event.get("negative_prompt_cache"), receipt)
        self.assertEqual(event["cpu_prompt_encode_ms"], duration)

    def test_synchronous_prepare_and_warmup_keep_opaque_receipts(self):
        value = self.execute(0)
        warmed = value["phases"][0]["cells"][0]["warmup"][0]
        self.assertEqual(warmed["spec"]["request_id"], "warmup-0")
        self.assertTrue(warmed["state_id"])
        self.assertEqual(warmed["negative_prompt_cache"]["sequence"], 1)
        self.assertEqual(warmed["cpu_prompt_encode_ms"], 1001.)
        self.assertEqual(len(warmed["encoding_events"]), 1)
        self.assertEqual(warmed["encoding_events"][0]["pid"], warmed["worker_pid"])
        self.assertEqual(warmed["encoding_events"][0]["negative_prompt_cache"], warmed["negative_prompt_cache"])
        for index, row in enumerate(value["requests"]):
            receipt = dict(schema="cpu-telemetry-fixture", sequence=index + 2,
                           request_id=row["request_id"], nested=dict(kept=[1, None, "opaque"]))
            self.assertEqual(row["negative_prompt_cache"], receipt)
            self.assertEqual(row["cpu_prompt_encode_ms"], 1002. + index)
            self.assertEqual(len(row["encoding_events"]), 1)
            self.check_event(row["encoding_events"][0], row, "prepare", receipt, 1002. + index)
            self.assertNotIn("ticket", row["encoding_events"][0])
            self.assertEqual(row["operation_counts"]["prepare"], 1)
            self.assertNotIn("encode_start", row["operation_counts"])
            self.assertEqual(row["model_ms"], 0.)
        archived = self.assert_archive(value)
        self.assertEqual(archived["requests"], value["requests"])
        self.assertEqual(archived["phases"][0]["cells"][0]["warmup"], value["phases"][0]["cells"][0]["warmup"])

    def test_prefetch_poll_and_prepare_preserve_same_ticket_and_receipt(self):
        value = self.execute(1)
        for row in value["requests"]:
            events = row["encoding_events"]
            self.assertEqual([event["op"] for event in events], ["encode_poll", "prepare"])
            self.assertTrue(events[0]["ticket"])
            self.assertEqual(events[0]["ticket"], events[1]["ticket"])
            for event in events:
                self.check_event(event, row, event["op"], row["negative_prompt_cache"], row["cpu_prompt_encode_ms"])
            self.assertEqual(row["encode_ms"], row["cpu_prompt_encode_ms"])
            self.assertEqual(row["model_ms"], 0.)
            self.assertEqual(row["operation_counts"]["prepare"], 1)
            self.assertEqual(row["operation_counts"]["encode_start"], 1)

    def test_disabled_cache_keeps_explicit_null_and_cpu_encoding_time(self):
        value = self.execute(1, disabled=True)
        warmed = value["phases"][0]["cells"][0]["warmup"][0]
        self.assertIsNone(warmed["negative_prompt_cache"])
        self.assertEqual(warmed["cpu_prompt_encode_ms"], 1001.)
        for row in value["requests"]:
            self.assertIn("negative_prompt_cache", row)
            self.assertIsNone(row["negative_prompt_cache"])
            self.assertNotIn("negative_prompt_cache", row["encoding_events"][0])
            self.assertIsNone(row["encoding_events"][1]["negative_prompt_cache"])
            self.assertEqual(row["encode_ms"], row["cpu_prompt_encode_ms"])
            self.assertEqual(row["model_ms"], 0.)


if __name__ == "__main__":
    unittest.main()
