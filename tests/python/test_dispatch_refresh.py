"""CPU subprocess tests of dispatch eligibility and bounded host telemetry.

Only a temporary frozen MockBackend is instrumented. Cross-worker barriers and
event ordering establish the scenarios; none of these timings are model data.
"""

import copy
import unittest

import test_study as study
from ditserve.runtime.client import CoreBridge, CoreError


FIXTURE = r'''
from pathlib import Path

_refresh_start = MockBackend.encode_start
_refresh_poll = MockBackend.encode_poll
_refresh_prepare = MockBackend.prepare
_refresh_advance = MockBackend.advance
_refresh_release = MockBackend.release

def _refresh_marker(self, name):
    return Path(self.config['options']['fixture_dir']) / name

def _refresh_wait(self, name):
    deadline = time.monotonic() + 3
    while not _refresh_marker(self, name).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('dispatch fixture barrier timed out: ' + name)
        time.sleep(.001)

def _start(self, request):
    result = _refresh_start(self, request)
    self.refresh_receipts = getattr(self, 'refresh_receipts', {})
    self.refresh_receipts[result['ticket']] = dict(
        schema='dispatch-refresh-fixture', request_id=request['spec']['request_id'],
        nested=dict(values=[None, 1, 'preserved']))
    return result

def _poll(self, request):
    options = self.config['options']
    mode = options['dispatch_fixture']
    if mode == 'failure':
        raise RuntimeError('injected encode_poll failure')
    if not getattr(self, 'refresh_delayed', False):
        self.refresh_delayed = True
        if mode == 'arrival':
            # Both workers are warmed before trace origin. The test checks
            # that the frozen 200 ms arrival really falls inside this RPC.
            time.sleep(.6)
        elif mode == 'completion':
            _refresh_marker(self, 'poll_entered').touch()
            _refresh_wait(self, 'image_released')
            # Let the image's final response/future publish before returning.
            time.sleep(.05)
    result = _refresh_poll(self, request)
    if result['ready']:
        result.update(model_ms=1234., negative_prompt_cache=self.refresh_receipts[request['ticket']])
    return result

def _prepare(self, request):
    ticket = request.get('encode_ticket')
    receipt = self.refresh_receipts.pop(ticket) if ticket else None
    result = _refresh_prepare(self, request)
    if receipt is not None:
        result.update(cpu_prompt_encode_ms=1234., negative_prompt_cache=receipt)
    return result

def _advance(self, request):
    if (self.config['options']['dispatch_fixture'] == 'completion'
            and request['state_id'] == 'inflight'):
        _refresh_wait(self, 'poll_entered')
    return _refresh_advance(self, request)

def _release(self, request):
    result = _refresh_release(self, request)
    if (self.config['options']['dispatch_fixture'] == 'completion'
            and request['state_id'] == 'inflight'):
        _refresh_marker(self, 'image_released').touch()
    return result

MockBackend.encode_start = _start
MockBackend.encode_poll = _poll
MockBackend.prepare = _prepare
MockBackend.advance = _advance
MockBackend.release = _release
'''


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class DispatchRefreshTest(unittest.TestCase):
    PROFILE = study.NativeStudyTest.PROFILE
    setUp = study.NativeStudyTest.setUp
    request = staticmethod(study.NativeStudyTest.request)
    phase = study.NativeStudyTest.phase
    config = study.NativeStudyTest.config
    run_study = study.NativeStudyTest.run_study
    assert_completed = study.NativeStudyTest.assert_completed

    @classmethod
    def setUpClass(cls):
        study.NativeStudyTest.setUpClass.__func__(cls)
        path = cls.package_root / "ditserve/models/mock.py"
        path.write_text(path.read_text() + FIXTURE)

    def execute(self, mode, refresh=None):
        name = "%s-%s" % (mode, refresh)
        markers = self.root / (name + "-markers")
        markers.mkdir()
        options = dict(dispatch_fixture=mode, fixture_dir=str(markers),
                       encode_delay_ms=35 if mode == "many" else 0)
        workers = {key: dict(backend="mock", config=dict(options=copy.deepcopy(options)))
                   for key in ("model", "image", "other_model", "other_image")}
        video = self.request("video", slo_ms=5000)
        phase = self.phase(text_prefetch_depth=1, record_dispatch_timeline=True,
                           output_mode="digest", workload=[video],
                           warmup=[dict(worker=key, spec=dict(steps=1)) for key in workers])
        if refresh is not None:
            phase["dispatch_refresh"] = refresh
        if mode == "arrival":
            phase["workload"].append(self.request("arriving", kind="image", arrival_ms=200, slo_ms=100))
        elif mode == "completion":
            phase["gpus"].append(dict(video="other_model", image="other_image"))
            phase["options"]["reserve_image_gpus"] = 1
            phase["workload"].append(self.request("inflight", kind="image", slo_ms=100))
        value = self.run_study(self.config([phase], workers=workers), name=name)
        if mode != "failure":
            self.assert_completed(value)
        return value, {row["request_id"]: row for row in value["requests"]}

    def assert_telemetry(self, value, rows, refresh):
        cell = value["phases"][0]["cells"][0]
        self.assertIs(cell["dispatch_refresh"], refresh)
        for row in rows.values():
            self.assertGreaterEqual(row["admission_ms"], row["arrival_ms"])
            self.assertGreaterEqual(row["start_ms"], row["admission_ms"])
            for event in row["dispatch_timeline"]:
                self.assertEqual(event["timing_scope"], "host_dispatch_task")
                self.assertLessEqual(event["start_ms"], event["rpc_completed_ms"])
                self.assertLessEqual(event["rpc_completed_ms"], event["harvested_ms"])
                self.assertLessEqual(event["harvested_ms"], event["finish_ms"])
                self.assertEqual(event["harvest_delay_ms"], event["harvested_ms"] - event["rpc_completed_ms"])
        video = rows["video"]
        stats = video["encoder_rpc"]
        self.assertEqual(stats["timing_scope"], "host_encoder_round_trip")
        self.assertEqual(stats["calls"], sum(video["operation_counts"][op] for op in ("encode_start", "encode_poll")))
        self.assertEqual(stats["failed_calls"], 0)
        self.assertGreaterEqual(stats["total_ms"], stats["max_ms"])
        self.assertGreaterEqual(stats["max_ms"], 0)
        for name, endpoint in (("start", "encode_submit_ms"), ("ready_poll", "encode_ready_ms")):
            window = stats[name]
            self.assertFalse(window["failed"])
            self.assertLessEqual(window["start_ms"], window["finish_ms"])
            self.assertLessEqual(window["finish_ms"], video[endpoint])
        self.assertEqual(set(stats), {"timing_scope", "calls", "total_ms", "max_ms", "failed_calls", "start", "ready_poll"})
        self.assertEqual(video["model_ms"], 0)  # Encoder telemetry is not charged twice.
        events = video["encoding_events"]
        self.assertEqual([event["op"] for event in events], ["encode_poll", "prepare"])
        self.assertEqual(events[0]["ticket"], events[1]["ticket"])
        self.assertEqual(events[0]["negative_prompt_cache"], events[1]["negative_prompt_cache"])
        self.assertEqual(video["cpu_prompt_encode_ms"], 1234.)

    def test_arrival_during_poll_refreshes_before_video_dispatch(self):
        reference_outputs = None
        for refresh in (False, True, None):
            with self.subTest(refresh=refresh):
                value, rows = self.execute("arrival", refresh)
                # The failed performance screen leaves refresh opt-in. The
                # omitted field must follow the explicit false control.
                enabled = refresh is True
                self.assert_telemetry(value, rows, enabled)
                video, image = rows["video"], rows["arriving"]
                poll = video["encoder_rpc"]["ready_poll"]
                self.assertLess(poll["start_ms"], image["arrival_ms"])
                self.assertLess(image["arrival_ms"], poll["finish_ms"])
                video_start = video["dispatch_timeline"][0]["start_ms"]
                if enabled:
                    self.assertLess(image["start_ms"], video_start)
                else:
                    self.assertLess(video_start, image["admission_ms"])
                    self.assertLess(video_start, image["start_ms"])
                outputs = {key: row["outputs"][0]["output"]["sha256"] for key, row in rows.items()}
                if reference_outputs is None:
                    reference_outputs = outputs
                self.assertEqual(reference_outputs, outputs)

    def test_other_gpu_completion_is_harvested_before_planning(self):
        for refresh in (False, True):
            with self.subTest(refresh=refresh):
                value, rows = self.execute("completion", refresh)
                self.assert_telemetry(value, rows, refresh)
                video, image = rows["video"], rows["inflight"]
                poll = video["encoder_rpc"]["ready_poll"]
                event = image["dispatch_timeline"][0]
                self.assertEqual(event["group"], [1])
                self.assertLess(poll["start_ms"], event["rpc_completed_ms"])
                self.assertLess(event["rpc_completed_ms"], poll["finish_ms"])
                self.assertGreater(event["harvest_delay_ms"], 0)
                video_start = video["dispatch_timeline"][0]["start_ms"]
                if refresh:
                    self.assertLess(event["harvested_ms"], video_start)
                else:
                    self.assertLess(video_start, event["harvested_ms"])

    def test_many_polls_keep_bounded_telemetry_and_receipt(self):
        for refresh in (False, True):
            value, rows = self.execute("many", refresh)
            self.assert_telemetry(value, rows, refresh)
            self.assertGreater(rows["video"]["encoder_rpc"]["calls"], 2)

    def test_failed_poll_preserves_bounded_observation_and_cleanup(self):
        for refresh in (False, True):
            value, rows = self.execute("failure", refresh)
            self.assertEqual(value["status"], "failed")
            self.assertIsNone(value["budget"]["active"])
            self.assertTrue(all(session["cleanup_verified"] for session in value["budget"]["sessions"]))
            stats = rows["video"]["encoder_rpc"]
            self.assertEqual(stats["calls"], 2)
            self.assertEqual(stats["failed_calls"], 1)
            self.assertNotIn("ready_poll", stats)
            self.assertGreaterEqual(stats["total_ms"], stats["max_ms"])

    def test_refresh_requires_boolean_cluster_field_before_budget(self):
        for index, invalid in enumerate((0, 1, "true", None, [])):
            with self.subTest(invalid=invalid):
                name = "invalid-refresh-%d" % index
                with self.assertRaisesRegex(CoreError, "dispatch_refresh"):
                    self.run_study(self.config([self.phase(dispatch_refresh=invalid)]), name=name)
                self.assertFalse((self.root / (name + "-budget.json")).exists())
        phase = dict(id="operations", kind="operations", workers=["model"],
                     dispatch_refresh=True, actions=[dict(op="hello")])
        with self.assertRaisesRegex(CoreError, "dispatch_refresh"):
            self.run_study(self.config([phase]), name="invalid-operations-refresh")


if __name__ == "__main__":
    unittest.main()
