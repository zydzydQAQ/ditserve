"""CPU-only integration contracts for the native study controller.

The bridge and frozen package are real; execution uses mock and reporting
adapters. No model download, CUDA import, or GPU probe is needed.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from ditserve import api, results
from ditserve.runtime.client import CoreBridge, CoreError
from ditserve.runtime import PACKAGE_ROOT, ROOT


class FixtureBridge(CoreBridge):
    def __init__(self, package_root):
        super().__init__()
        self.package_root = package_root

    def call(self, command, payload):
        return super().call(command, dict(payload, package_root=str(self.package_root)))


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class NativeStudyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.package_directory.cleanup)
        cls.package_root = Path(cls.package_directory.name) / "python"
        source = PACKAGE_ROOT / "ditserve"
        for path in source.rglob("*.py"):
            target = cls.package_root / "ditserve" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bridge = FixtureBridge(self.package_root)
        paths = [str(ROOT / "workers")]
        if importlib.util.find_spec("openpyxl") is None:
            paths.append(str(ROOT / ".cache/tools"))
        if os.environ.get("PYTHONPATH"):
            paths.append(os.environ["PYTHONPATH"])
        environment = mock.patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join(paths)})
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def request(request_id="r0", **changes):
        value = dict(request_id=request_id, arrival_ms=0, service_ms=20, slo_ms=10000,
                     spec=dict(steps=2, prompt="a ceramic kettle", seed=7))
        value.update(changes)
        return value

    def config(self, phases, options=None, workers=None):
        return dict(schema="ditserve.study.v1", experiment="native",
                    workers=workers or {"model": {"backend": "mock", "config": {"options": options or {}}}},
                    budget=dict(limit_gpu_seconds=None, watchdog_seconds=10), phases=phases)

    def phase(self, phase_id="online", **changes):
        value = dict(id=phase_id, kind="online", worker="model", policy="edf",
                     workload=[self.request()])
        value.update(changes)
        return value

    def run_study(self, config, name="run", **options):
        return api.run(config, run_id=name, work_dir=self.root / name,
                       results_root=self.root / "results", ledger=self.root / (name + "-budget.json"),
                       core=self.bridge, **options)

    def assert_completed(self, value):
        self.assertEqual(value["status"], "completed", json.dumps(value.get("failures", value), indent=2))
        self.assertIsNone(value["budget"]["active"])
        self.assertEqual(value["budget"]["used_gpu_seconds"], 0)
        self.assertTrue(all(item["cleanup_verified"] for item in value["budget"]["sessions"]))

    def assert_archive(self, value, experiment="native", name="run"):
        destination = self.root / "results" / experiment / name
        self.assertEqual(value["archive"]["status"], "completed", value["archive"])
        self.assertEqual({path.name for path in destination.iterdir()}, {"results.xlsx", "REPORT.md"})
        saved = results.load(destination)
        self.assertEqual(saved["schema"], "ditserve.results.v1")
        self.assertEqual(saved["run"]["status"], value["status"])
        self.assertEqual(results.verify(destination)["status"], "verified")
        return saved["run"]

    def test_paired_matrix_counts_resident_workers_and_completed_resume(self):
        phase = self.phase(workload=[self.request("a"), self.request("b")],
                           matrix={"/policy": ["edf", "srtf"]}, paired=True)
        config = self.config([phase])
        value = self.run_study(config)
        self.assert_completed(value)
        cells = value["phases"][0]["cells"]
        self.assertEqual([(cell["cell"], cell["order"]) for cell in cells],
                         [(0, "forward"), (1, "forward"), (1, "reverse"), (0, "reverse")])
        self.assertEqual(value["metrics"]["requests"], 8)
        self.assertEqual(value["metrics"]["completed"], 8)
        self.assertEqual(value["metrics"]["observation_count"], 4)
        self.assertAlmostEqual(value["metrics"]["observation_ms"], sum(cell["elapsed_ms"] for cell in cells))
        self.assertIsNone(value["metrics"]["qualified_goodput_rps"])
        self.assertEqual(value["metrics"]["qualification_status"], "unverified")
        for cell in cells:
            self.assertEqual(len({row["worker_pids"]["model"] for row in cell["requests"]}), 1)
            for row in cell["requests"]:
                self.assertEqual(row["operation_counts"], dict(prepare=1, advance=2, finish=1, release=1))
        self.assertEqual(self.assert_archive(value)["metrics"], value["metrics"])
        ledger = self.root / "run-budget.json"
        before = ledger.read_bytes()
        resumed = self.run_study(config, resume=True)
        self.assert_completed(resumed)
        self.assertEqual(resumed["phases"], value["phases"])
        self.assertEqual(ledger.read_bytes(), before)
        self.assert_archive(resumed)

    def test_gates_require_actual_evidence_before_next_phase(self):
        first = self.phase("profile", kind="profile", gates=[dict(metric="/cells/0/metrics/completed", op="ge", value=1)])
        second = self.phase("quality", requires=[dict(phase="profile", requires="qualified")],
                            gates=[dict(metric="/cells/0/metrics/quality_score", op="ge", value=.9)])
        third = self.phase("after_quality", requires=[dict(phase="quality", requires="qualified")])
        value = self.run_study(self.config([first, second, third]))
        self.assertEqual(value["status"], "incomplete")
        self.assertEqual(value["blocked_phase"], "after_quality")
        self.assertEqual([item["id"] for item in value["phases"]], ["profile", "quality"])
        self.assertTrue(value["phases"][0]["qualification"]["qualified"])
        self.assertFalse(value["phases"][1]["qualification"]["qualified"])
        self.assertEqual(len(value["budget"]["sessions"]), 2)
        self.assertIsNone(value["budget"]["active"])
        self.assertTrue(value["retention"]["pending_dependencies"])

    def test_hot_cold_eviction_preserves_outputs_and_cross_worker_migration(self):
        phase = self.phase("state", kind="state", resident_limit=1,
                           workload=[self.request("long", spec=dict(steps=3, prompt="long", seed=4)),
                                     self.request("urgent", arrival_ms=1, deadline_ms=1000)],
                           matrix={"/eviction": ["hot", "cold"]})
        workers = {name: dict(backend="mock", config=dict(options=dict(delay_ms=3))) for name in ("model", "target")}
        migration = self.phase("migration", kind="state", resident_limit=1, workload=[
            self.request("occupied", worker="target", spec=dict(steps=3, prompt="resident", seed=2)),
            self.request("move", arrival_ms=1, deadline_ms=1000, transitions=[
                dict(at_step=1, target_worker="target", target_device="cpu", mode="cold")])])
        value = self.run_study(self.config([phase, migration], workers=workers))
        self.assert_completed(value)
        hashes = []
        for cell in value["phases"][0]["cells"]:
            rows = {row["request_id"]: row for row in cell["requests"]}
            self.assertGreaterEqual(rows["long"]["operation_counts"].get("snapshot", 0), 1)
            self.assertGreaterEqual(rows["long"]["operation_counts"].get("restore", 0), 1)
            hashes.append({key: row["outputs"][0]["output"]["sha256"] for key, row in rows.items()})
        self.assertEqual(hashes[0], hashes[1])
        resident, moved = value["phases"][1]["cells"][0]["requests"]
        self.assertGreaterEqual(resident["operation_counts"].get("snapshot", 0), 1)
        self.assertGreaterEqual(resident["operation_counts"].get("restore", 0), 1)
        self.assertEqual(set(moved["worker_pids"]), {"model", "target"})
        self.assertEqual(len(set(moved["worker_pids"].values())), 2)
        self.assertEqual(moved["operation_counts"]["migration_release"], 1)
        self.assertEqual(moved["operation_counts"]["restore"], 2)
        self.assertEqual(moved["status"], "completed")

    def test_two_stage_session_and_think_time_are_causal(self):
        stages = [dict(worker="text", spec=dict(steps=2, prompt="blue cup", seed=8), cost_ms=10),
                  dict(worker="image", spec=dict(steps=2, prompt="overwritten", seed=9),
                       prompt_from_previous=True, cost_ms=30)]
        rows = [self.request("first", session_id="chat", stages=stages),
                self.request("second", session_id="chat", stages=stages, after="first", think_ms=17)]
        workers = {name: dict(backend="mock") for name in ("text", "image")}
        value = self.run_study(self.config([self.phase(kind="session", policy="stage_edf", worker="text", workload=rows)], workers=workers))
        self.assert_completed(value)
        first, second = value["requests"]
        self.assertAlmostEqual(second["arrival_ms"], first["finish_ms"] + 17)
        self.assertGreaterEqual(second["start_ms"], second["arrival_ms"])
        for row in (first, second):
            self.assertEqual(row["operation_counts"]["prepare"], 2)
            self.assertEqual(len(row["outputs"]), 2)
            self.assertEqual(row["outputs"][0]["output"]["text"], "blue cup")
            self.assertEqual(row["outputs"][1]["output"]["text"], "blue cup")
            self.assertEqual(set(row["worker_pids"]), {"text", "image"})

    def test_streaming_updates_buffers_and_full_playback_drain(self):
        row = self.request(updates=[dict(arrival_ms=0, version=1, prompt="first"),
                                    dict(arrival_ms=0, version=2, prompt="second")])
        phase = self.phase(kind="streaming", workload=[row], chunks=3, fps=1000,
                           matrix={"/buffer_chunks": [0, 2], "/prompt_policy": ["fifo", "latest_only"]})
        value = self.run_study(self.config([phase]))
        self.assert_completed(value)
        cells = value["phases"][0]["cells"]
        self.assertEqual(len(cells), 4)
        for cell in cells:
            playback = cell["requests"][0]["playback"]
            self.assertEqual(playback["frames_played"], 12)
            self.assertEqual(playback["frames_dropped"], 0)
            events = cell["prompt_events"]
            self.assertEqual(len(events), 2)
            self.assertFalse(any(event["status"] == "pending_at_stop" for event in events))
            applied = [event["version"] for event in events if event["status"] == "applied"]
            self.assertIn(applied, ([1, 2], [2]))
            self.assertEqual(playback["samples"][-1]["version"], 2)
            self.assertEqual(cell["semantic_status"], "unverified")

    def test_sp_actions_use_persistent_independent_ranks(self):
        actions = [dict(op="prepare", state_id="sp", spec=dict(steps=2, prompt="rank", seed=0)),
                   dict(op="advance", state_id="sp", steps=2), dict(op="inventory", state_id="sp"),
                   dict(op="release", state_id="sp")]
        phase = dict(id="sp", kind="sp", workers=["rank0", "rank1"], actions=actions)
        workers = {name: dict(backend="mock") for name in phase["workers"]}
        value = self.run_study(self.config([phase], workers=workers))
        self.assert_completed(value)
        operations = value["phases"][0]["cells"][0]["operations"]
        pids = [rank["worker_pid"] for rank in operations[0]["ranks"]]
        self.assertEqual(len(set(pids)), 2)
        for operation in operations:
            self.assertEqual([rank["worker_pid"] for rank in operation["ranks"]], pids)
        self.assertTrue(all(rank["progress"]["done"] for rank in operations[1]["ranks"]))
        self.assertTrue(all(rank["step"] == 2 for rank in operations[2]["ranks"]))

    def test_failed_resume_appends_attempt_and_rejects_config_changes(self):
        config = self.config([self.phase()], options=dict(fail_on="advance"))
        first = self.run_study(config)
        self.assertEqual(first["status"], "failed")
        self.assertIn("injected mock advance failure", first["phases"][0]["error"])
        self.assertEqual(first["metrics"]["requests"], 1)
        self.assertEqual(first["metrics"]["failed"], 1)
        resumed = self.run_study(config, resume=True)
        self.assertEqual(resumed["status"], "failed")
        self.assertEqual([item["attempt"] for item in resumed["phases"]], [1, 2])
        self.assertEqual(len(resumed["budget"]["sessions"]), 2)
        self.assertIsNone(resumed["budget"]["active"])
        self.assertEqual(resumed["budget"]["sessions"][0], first["budget"]["sessions"][0])
        for session in resumed["budget"]["sessions"]:
            self.assertGreater(session["elapsed_seconds"], 0)
            self.assertEqual(session["gpu_seconds"], 0)
            self.assertTrue(session["cleanup_verified"])
        before = (self.root / "run-budget.json").read_bytes()
        changed = copy.deepcopy(config)
        changed["phases"][0]["policy"] = "srtf"
        with self.assertRaisesRegex(CoreError, "frozen config differs"):
            self.run_study(changed, resume=True)
        self.assertEqual((self.root / "run-budget.json").read_bytes(), before)

    def test_watchdog_settles_failed_cpu_session(self):
        phase = self.phase(watchdog_seconds=.15)
        value = self.run_study(self.config([phase], options=dict(delay_ms=1000)))
        self.assertEqual(value["status"], "failed")
        self.assertRegex(value["phases"][0]["error"], "timed out|watchdog|deadline")
        self.assertIsNone(value["budget"]["active"])
        session = value["budget"]["sessions"][0]
        self.assertGreaterEqual(session["elapsed_seconds"], .15)
        self.assertLess(session["elapsed_seconds"], 3)
        self.assertEqual(session["gpu_seconds"], 0)
        self.assertTrue(session["cleanup_verified"])

    def test_resume_cannot_replace_ledger_or_run_identity(self):
        for name, options in (("done", {}), ("failed", dict(fail_on="advance"))):
            with self.subTest(status=name):
                config = self.config([self.phase()], options=options)
                value = self.run_study(config, name=name)
                self.assertIn(value["status"], ("completed", "failed"))
                ledger = self.root / (name + "-budget.json")
                before = ledger.read_bytes()
                replacement = self.root / (name + "-replacement.json")
                for run_id, requested_ledger, error in ((name, replacement, "budget ledger differs"),
                                                        ("changed-id", ledger, "run_id differs")):
                    with self.assertRaisesRegex(CoreError, error):
                        api.run(config, run_id=run_id, work_dir=self.root / name, resume=True,
                                ledger=requested_ledger, results_root=self.root / "results", core=self.bridge)
                self.assertEqual(ledger.read_bytes(), before)
                self.assertFalse(replacement.exists())

    def test_stream_failure_keeps_entire_frozen_request_denominator(self):
        phase = self.phase(kind="streaming", chunks=2, fps=1000,
                           workload=[self.request("failed"), self.request("waiting", arrival_ms=1000)])
        value = self.run_study(self.config([phase], options=dict(fail_on="advance")))
        self.assertEqual(value["status"], "failed")
        self.assertIsNone(value["budget"]["active"])
        self.assertEqual(value["metrics"]["requests"], 2)
        self.assertGreaterEqual(value["metrics"]["failed"], 1)
        self.assertEqual({row["request_id"] for row in value["requests"]}, {"failed", "waiting"})
        self.assertTrue(all(row.get("observation_id") for row in value["requests"]))

    def test_stream_drain_timeout_cannot_report_success(self):
        phase = self.phase(kind="streaming", chunks=1, fps=1, buffer_chunks=2,
                           playback_drain_seconds=0)
        value = self.run_study(self.config([phase]))
        self.assertEqual(value["status"], "failed")
        self.assertRegex(value["phases"][0]["error"], "drain|playback")
        self.assertEqual(value["metrics"]["requests"], 1)
        self.assertIsNone(value["budget"]["active"])

    def test_invalid_execution_inputs_leave_no_active_budget(self):
        cases = [self.phase(paired="yes"), self.phase(workload=[]),
                 self.phase(workload=[self.request(stages=[])]), self.phase(worker="missing"),
                 self.phase(worker_config=[]), self.phase(worker_config=dict(device="cuda:0")),
                 self.phase(workload=[self.request(updates=[dict(arrival_ms=2, version=1, prompt="first"),
                                                            dict(arrival_ms=1, version=2, prompt="second")])]),
                 dict(id="sp", kind="sp", workers=["model", "model"], actions=[dict(op="hello")])]
        for index, phase in enumerate(cases):
            with self.subTest(phase=phase):
                name = "invalid-%d" % index
                with self.assertRaises(CoreError):
                    self.run_study(self.config([phase]), name=name)
                ledger = self.root / (name + "-budget.json")
                if ledger.exists():
                    value = json.loads(ledger.read_text())
                    self.assertIsNone(value["active"])
                    self.assertEqual(value["sessions"], [])

    def test_worker_config_cannot_bypass_gpu_admission_or_source_binding(self):
        # hello is deliberately harmless even if admission regresses: no model
        # is loaded and no evaluator is invoked by this CPU test.
        phase = dict(id="probe", kind="profile", workers=["model"], actions=[dict(op="hello")])
        gpu = "GPU-00000000-0000-0000-0000-000000000000"
        workers = dict(model=dict(backend="evaluation", device="cuda:0", gpu_uuid=gpu, config={}))
        with self.assertRaisesRegex(CoreError, "allow-gpu"):
            self.run_study(self.config([phase], workers=workers), name="gpu-override")
        ledger_path = self.root / "gpu-override-budget.json"
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            self.assertIsNone(ledger["active"])
            self.assertEqual(ledger["sessions"], [])
        for index, variable in enumerate(("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
            with self.subTest(variable=variable):
                workers = dict(model=dict(backend="mock", environment={variable: "override"}))
                with self.assertRaisesRegex(CoreError, "environment cannot override"):
                    self.run_study(self.config([phase], workers=workers), name="environment-%d" % index)

    def test_prefetch_identity_and_async_decode(self):
        phase = self.phase(text_prefetch=2, decode_mode="async", resident_limit=1,
                           workload=[self.request("a"), self.request("b"), self.request("c")])
        value = self.run_study(self.config([phase], options=dict(encode_delay_ms=5, decode_delay_ms=20)))
        self.assert_completed(value)
        for row in value["requests"]:
            self.assertGreaterEqual(row["encode_ready_ms"], row["encode_submit_ms"])
            self.assertGreaterEqual(row["start_ms"], row["encode_ready_ms"])
            self.assertEqual(row["operation_counts"]["encode_start"], 1)
            self.assertGreaterEqual(row["operation_counts"]["encode_poll"], 1)
            self.assertEqual(row["operation_counts"]["prepare"], 1)
            self.assertEqual(row["operation_counts"]["decode_start"], 1)
            self.assertGreater(row["operation_counts"]["decode_poll"], 1)
            self.assertEqual(len(row["outputs"]), 1)

    def test_simulate_entrypoint_publishes_only_two_files(self):
        config = dict(generator=dict(count=3, seed=9), policies=["edf", "cache_only"])
        value = api.simulate(config, run_id="sim", work_dir=self.root / "sim",
                             results_root=self.root / "results", core=self.bridge)
        self.assertEqual(value["status"], "completed")
        self.assertEqual(len(value["phases"]), 2)
        self.assertEqual(value["budget"]["gpu_seconds"], 0)
        self.assertIn("Synthetic", value["notice"])
        self.assert_archive(value, "synthetic", "sim")


if __name__ == "__main__":
    unittest.main()
