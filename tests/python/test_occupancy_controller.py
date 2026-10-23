"""Native guard routing and fail-closed cleanup with CPU-only frozen workers.

The temporary worker maps the nominal GPU backend to MockBackend and replaces
occupancy before main. No driver, CUDA, or model is loaded by this fixture.
"""

import os
import json
import unittest
from unittest.mock import patch

import test_study as study
from ditserve.runtime.client import CoreBridge, CoreError


FIXTURE = r'''
BACKENDS['wan'] = ('mock', 'MockBackend')
_guard_dispatch = Worker.dispatch
def _guard_fixture_dispatch(self, request):
    if request.get('op') == 'occupancy':
        backend = request.get('occupancy_backend', 'nvidia_smi')
        self._test_occupancy_backend = backend
        mode = os.environ.get('DITSERVE_TEST_GUARD_FAILURE', '')
        runtime = bool(request.get('allowed_pids'))
        status = 'unknown' if runtime and backend == 'nvml' and mode == 'unknown' else 'exclusive_observed'
        reported = 'nvidia_smi' if runtime and mode == 'backend' else backend
        return dict(status=status, owned=[], foreign=[], devices=[],
                    occupancy_backend=reported, cuda_initialized=False)
    if request.get('op') == 'shutdown' and hasattr(self, '_test_occupancy_backend'):
        path = Path(self.config['work_dir']) / 'occupancy-shutdown.json'
        path.write_text(json.dumps(dict(occupancy_backend=self._test_occupancy_backend)))
    return _guard_dispatch(self, request)
Worker.dispatch = _guard_fixture_dispatch
'''


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class OccupancyControllerTests(unittest.TestCase):
    setUp = study.NativeStudyTest.setUp
    request = staticmethod(study.NativeStudyTest.request)
    phase = study.NativeStudyTest.phase
    config = study.NativeStudyTest.config
    run_study = study.NativeStudyTest.run_study
    PROFILE = study.NativeStudyTest.PROFILE

    @classmethod
    def setUpClass(cls):
        study.NativeStudyTest.setUpClass.__func__(cls)
        worker = cls.package_root / "ditserve/runtime/worker.py"
        worker.write_text(worker.read_text().replace('if __name__ == "__main__":',
                                                    FIXTURE + '\nif __name__ == "__main__":'))

    def execute(self, backend=None, failure=""):
        phase = self.phase(record_dispatch_timeline=True)
        if backend is not None:
            phase["occupancy_backend"] = backend
        workers = dict(model=dict(backend="wan", device="cuda:0",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000000", config={}),
            image=dict(backend="mock", config={}))
        with patch.dict(os.environ, DITSERVE_TEST_GUARD_FAILURE=failure):
            return self.run_study(self.config([phase], workers=workers),
                                  name="guard-%s-%s" % (backend, failure), allow_gpu=True)

    def test_default_and_explicit_routes_are_recorded_without_changing_outputs(self):
        outputs = []
        for backend in (None, "nvidia_smi", "nvml"):
            with self.subTest(backend=backend):
                value = self.execute(backend)
                self.assertEqual(value["status"], "completed", value.get("failures"))
                cell = value["phases"][0]["cells"][0]
                stats = cell["guard_telemetry"]
                expected = backend or "nvidia_smi"
                self.assertEqual(stats["occupancy_backend"], expected)
                self.assertEqual((stats["cadence_seconds"], stats["probe_timeout_seconds"]), (2., 3.))
                self.assertTrue(stats["fail_closed"])
                self.assertEqual(stats["probe_errors"], 0)
                self.assertEqual(stats["dropped_events"], 0)
                self.assertGreaterEqual(stats["probe_calls"], 1)
                self.assertGreaterEqual(stats["heartbeat_calls"], stats["probe_calls"])
                events = stats["events"]
                self.assertEqual(len(events), stats["probe_calls"])
                self.assertEqual(stats["probe_total_ms"], sum(e["duration_ms"] for e in events))
                for event in events:
                    self.assertEqual(event["occupancy_backend"], expected)
                    self.assertEqual(event["status"], "exclusive_observed")
                    self.assertGreaterEqual(event["finish_monotonic_seconds"], event["start_monotonic_seconds"])
                    self.assertEqual(event["duration_ms"],
                        (event["finish_monotonic_seconds"] - event["start_monotonic_seconds"]) * 1000)
                self.assertGreater(cell["host_origin_monotonic_seconds"], 0)
                markers = [path for path in (self.root / ("guard-%s-" % backend)).rglob("occupancy-shutdown.json")
                           if "cell-0" in path.parts]
                self.assertEqual(len(markers), 1, "resident cell probe did not receive graceful shutdown")
                self.assertEqual(json.loads(markers[0].read_text())["occupancy_backend"], expected)
                outputs.append(value["requests"][0]["outputs"][0]["output"]["sha256"])
        self.assertEqual(len(set(outputs)), 1)

    def test_unknown_or_wrong_backend_aborts_and_reaps_workers(self):
        for failure in ("unknown", "backend"):
            with self.subTest(failure=failure):
                value = self.execute("nvml", failure)
                self.assertEqual(value["status"], "failed")
                self.assertIsNone(value["budget"]["active"])
                self.assertTrue(all(row["cleanup_verified"] for row in value["budget"]["sessions"]))
                stats = value["phases"][0]["cells"][0]["guard_telemetry"]
                self.assertEqual(stats["probe_errors"], 1)

    def test_invalid_probe_backend_rejected_before_workers_start(self):
        for backend in (True, 1, None, "disabled", ""):
            with self.subTest(backend=backend):
                phase = self.phase(occupancy_backend=backend)
                with self.assertRaisesRegex(CoreError, "occupancy_backend"):
                    self.run_study(self.config([phase]), name="invalid-%s" % backend)


if __name__ == "__main__":
    unittest.main()
