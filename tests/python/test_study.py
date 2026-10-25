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
        for package in ("ditserve", "xfuser"):
            source = PACKAGE_ROOT / package
            for path in source.rglob("*.py"):
                target = cls.package_root / package / path.relative_to(source)
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

    def test_invalid_execution_inputs_leave_no_active_budget(self):
        cases = [self.phase(paired="yes"), self.phase(workload=[]),
                 self.phase(workload=[self.request(stages=[])]), self.phase(worker="missing"),
                 self.phase(worker_config=[]), self.phase(worker_config=dict(device="cuda:0")),
                 self.phase(kind="streaming"),
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

    def test_warmup_runs_before_trace_clock_and_prefetch_is_per_worker(self):
        workers = {"model": {"backend": "mock", "config": {"options": {"encode_delay_ms": 5}}},
                   "image": {"backend": "mock", "config": {"options": {}}}}
        phase = self.phase(text_prefetch=1, text_prefetch_workers=["model"],
                           warmup=[dict(worker="model", spec=dict(steps=2)), dict(worker="image", spec=dict(steps=1))],
                           workload=[self.request("v"), self.request("i", worker="image", kind="image")])
        value = self.run_study(self.config([phase], workers=workers))
        self.assert_completed(value)
        rows = {row["request_id"]: row for row in value["requests"]}
        self.assertEqual(rows["v"]["operation_counts"]["encode_start"], 1)
        self.assertNotIn("encode_start", rows["i"]["operation_counts"])
        warmed = value["phases"][0]["cells"][0]["warmup"]
        self.assertEqual([item["worker"] for item in warmed], ["model", "image"])
        self.assertTrue(all(item["elapsed_ms"] >= 0 for item in warmed))
        with self.assertRaises(CoreError):
            self.run_study(self.config([self.phase(text_prefetch_workers=["missing"])]), name="bad-prefetch")
        with self.assertRaises(CoreError):
            self.run_study(self.config([self.phase(warmup=[dict(worker="missing")])]), name="bad-warmup")

    def test_output_mode_check_keeps_outputs_off_disk(self):
        kept = self.run_study(self.config([self.phase(workload=[self.request("a")])]), name="kept")
        checked = self.run_study(self.config([self.phase(output_mode="check", workload=[self.request("a")])]), name="checked")
        self.assert_completed(kept)
        self.assert_completed(checked)
        self.assertEqual(len(kept["artifacts"]), 1)
        self.assertEqual(checked["artifacts"], [])
        output = checked["requests"][0]["outputs"][0]["output"]
        self.assertEqual(output["output_mode"], "check")
        self.assertNotIn("path", output)
        self.assertNotIn("sha256", output)
        from ditserve.models.base import artifact
        with self.assertRaises(ValueError):
            artifact("x", None, "upload")

    def cluster_phase(self, policy, gpus=4, **changes):
        profile = dict(provenance="synthetic", sp_efficiency=1, resize_ms=1,
                       classes=dict(video=dict(kind="video", steps=3, blocks_per_step=4, prepare_ms=0, encode_ms=0,
                                               step_compute_ms=400, comm_mb_per_step=0, collectives_per_step=0, finish_ms=20),
                                    image=dict(kind="image", service_ms=20)),
                       links={"2": dict(gbps=10, latency_ms=0), "4": dict(gbps=10, latency_ms=0)})
        workload = [dict(request_id="v%d" % i, kind="video", worker="v0", arrival_ms=0, slo_ms=20000,
                         spec=dict(steps=3, blocks_per_step=4)) for i in range(2)]
        workload += [dict(request_id="i%d" % i, kind="image", worker="i0", arrival_ms=100 + 150 * i, slo_ms=300,
                          spec=dict(steps=1)) for i in range(4)]
        for row in workload:
            row["class"] = row["kind"]
        value = dict(id="cluster", kind="cluster", policy=policy, profile=profile, text_prefetch=1,
                     gpus=[dict(video="v%d" % g, image="i%d" % g) for g in range(gpus)], workload=workload)
        value.update(changes)
        return value

    def cluster_workers(self, gpus=4):
        costs = dict(step_ms=400, finish_ms=20)
        workers = {}
        for g in range(gpus):
            workers["v%d" % g] = {"backend": "mock", "config": {"options": {"costs": costs}}}
            workers["i%d" % g] = {"backend": "mock", "config": {"options": {"costs": dict(finish_ms=20)}}}
        return workers

    def test_cluster_executor_runs_gpus_concurrently_with_sp_groups(self):
        value = self.run_study(self.config([self.cluster_phase("block_elastic")], workers=self.cluster_workers()), name="cluster")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertEqual(cell["metrics"]["completed"], 6)
        rows = {row["request_id"]: row for row in cell["requests"]}
        # Two 3-step videos at 400 ms/step on four GPUs: SP2 each, not 2.4 s serial.
        self.assertLess(cell["elapsed_ms"], 2400)
        self.assertTrue(all(max(rows[v]["degrees"]) >= 2 for v in ("v0", "v1")))
        self.assertEqual(cell["by_kind"]["image"]["completed"], 4)
        self.assertFalse(cell["real_backends_pinned_sp1"])
        # Block quanta after the request's own previous quantum are "steady"; those behind another request count as resumes.
        self.assertTrue(any("block_ms_steady" in rows[v] and rows[v]["resumes"] >= 0 for v in ("v0", "v1")))

    def test_mps_configuration_is_validated_before_execution(self):
        for mps in ([], dict(server_pid=0, pipe_directory="/tmp"),
                    dict(server_pid=1, pipe_directory="relative"),
                    dict(server_pid=1, pipe_directory="/tmp", unexpected=True)):
            with self.subTest(mps=mps), self.assertRaises(CoreError):
                self.run_study(self.config([self.phase(mps=mps)]), name="bad-mps")
        phase = self.phase(mps=dict(server_pid=1, pipe_directory="/tmp"),
                           matrix={"/mps/server_pid": [1, 2]})
        with self.assertRaisesRegex(CoreError, "MPS control identity"):
            self.run_study(self.config([phase]), name="changing-mps")
        valid = dict(server_pid=1, pipe_directory="/tmp")
        with self.assertRaisesRegex(CoreError, "single-phase"):
            self.run_study(self.config([self.phase(mps=valid), self.phase("second")]), name="multi-mps")

    def test_cluster_occupancy_backend_is_validated_and_reported(self):
        with self.assertRaises(CoreError):
            self.run_study(self.config([self.cluster_phase("block_elastic", occupancy_backend="smi")],
                                       workers=self.cluster_workers()), name="bad-occupancy")
        value = self.run_study(self.config([self.cluster_phase("block_elastic", occupancy_backend="nvml")],
                                           workers=self.cluster_workers()), name="nvml")
        self.assert_completed(value)
        telemetry = value["phases"][0]["cells"][0]["guard_telemetry"]
        self.assertEqual(telemetry["schema"], "ditserve.guard-telemetry.v1")
        self.assertEqual(telemetry["occupancy_backend"], "nvml")
        self.assertGreater(telemetry["heartbeat_calls"], 0)
        self.assertGreaterEqual(telemetry["lock_wait_max_ms"], 0)
        # CPU fixtures own no GPU devices, so the probe never runs.
        self.assertEqual(telemetry["probe_calls"], 0)
        self.assertFalse(telemetry["occupancy_thread"])
        with self.assertRaises(CoreError):
            self.run_study(self.config([self.cluster_phase("block_elastic", occupancy_thread="yes")],
                                       workers=self.cluster_workers()), name="bad-thread")
        value = self.run_study(self.config([self.cluster_phase("block_elastic", occupancy_backend="nvml", occupancy_thread=True)],
                                           workers=self.cluster_workers()), name="nvml-thread")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertTrue(cell["guard_telemetry"]["occupancy_thread"])
        # Every untransferred video quantum reports its worker time and the gap around it.
        videos = [row for row in cell["requests"] if row["kind"] == "video"]
        self.assertTrue(all(row["timed_quanta"] > 0 and row["dispatch_gap_ms"] >= 0 and row["worker_operation_ms"] > 0
                            for row in videos))

    def test_cluster_video_worker_options_patch_every_video_worker_only(self):
        for bad in ([], "on"):
            with self.subTest(bad=bad), self.assertRaises(CoreError):
                self.run_study(self.config([self.cluster_phase("block_elastic", video_worker_options=bad)],
                                           workers=self.cluster_workers()), name="bad-vwo-%s" % type(bad).__name__)
        with self.assertRaises(CoreError):
            self.run_study(self.config([self.phase(video_worker_options=dict(flag=1))]), name="vwo-online")
        value = self.run_study(self.config([self.cluster_phase("block_elastic", video_worker_options=dict(probe_flag=7))],
                                           workers=self.cluster_workers()), name="vwo")
        self.assert_completed(value)
        written = {path.stem: json.loads(path.read_text()) for path in (self.root / "vwo").rglob("workers/*.json")}
        for g in range(4):
            self.assertEqual(written["v%d" % g]["options"]["probe_flag"], 7)
            self.assertEqual(written["v%d" % g]["options"]["costs"], dict(step_ms=400, finish_ms=20))
            self.assertNotIn("probe_flag", written["i%d" % g]["options"])

    def test_mock_worker_holds_one_grouped_request_inside_a_solver_step(self):
        from ditserve.models.mock import MockBackend
        backend = MockBackend({"options": {}})
        for key in ("a", "b"):
            backend.prepare({"state_id": key, "spec": {"steps": 2, "blocks_per_step": 4}})
        quantum = dict(group=[0], sp_degree=1, blocks=2)
        backend.advance(dict(quantum, state_id="a", epoch=1))
        with self.assertRaisesRegex(ValueError, "inside a solver step"):
            backend.advance(dict(quantum, state_id="b", epoch=2))
        backend.advance(dict(quantum, state_id="a", epoch=1))
        self.assertEqual(backend.advance(dict(quantum, state_id="b", epoch=2))["executed_units"], 2)

    def test_mock_worker_keeps_the_group_of_a_request_inside_a_solver_step(self):
        from ditserve.models.mock import MockBackend
        backend = MockBackend({"options": {}})
        backend.prepare({"state_id": "a", "spec": {"steps": 2, "blocks_per_step": 4}})
        backend.advance(dict(state_id="a", group=[0], sp_degree=1, blocks=2, epoch=1))
        with self.assertRaisesRegex(ValueError, "solver-step boundary"):
            backend.advance(dict(state_id="a", group=[0, 1], sp_degree=2, blocks=2, epoch=2))
        backend.advance(dict(state_id="a", group=[0], sp_degree=1, blocks=2, epoch=1))
        self.assertEqual(backend.advance(dict(state_id="a", group=[0, 1], sp_degree=2, blocks=2, epoch=2))["executed_units"], 2)

    def test_cluster_export_transport_moves_state_into_sp_groups(self):
        phase = self.cluster_phase("block_elastic", options=dict(sp_transport="export"))
        value = self.run_study(self.config([phase], workers=self.cluster_workers()), name="export")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertEqual((cell["sp_transport"], cell["real_backends_pinned_sp1"]), ("export", False))
        self.assertEqual(cell["metrics"]["completed"], 6)
        rows = {row["request_id"]: row for row in cell["requests"]}
        grouped = [row for name, row in rows.items() if name.startswith("v") and max(row["degrees"]) >= 2]
        # Placement depends on timing; at least one video gets an SP group on four idle GPUs.
        self.assertTrue(grouped)
        for row in grouped:
            # One rank prepares; every other rank of an SP group receives an exported copy.
            self.assertGreaterEqual(row["transfers"], 1)
            self.assertGreater(row["transfer_bytes"], 0)
        self.assertEqual(list((self.root / "export").rglob("*.pt")), [])
        # Prompts are encoded ahead; every video is prepared from its ticket on the encoding worker and
        # exported into its group, never re-encoded on a group member.
        videos = [row for name, row in rows.items() if name.startswith("v")]
        self.assertTrue(videos and all(row["prompt_reencoded"] is False for row in videos))

    def test_cluster_doorbell_runs_videos_continuously_and_stops_them_for_images(self):
        with self.assertRaises(CoreError):
            # Continuous execution needs exported, regroupable state.
            self.run_study(self.config([self.cluster_phase("block_elastic", options=dict(doorbell=True))],
                                       workers=self.cluster_workers()), name="doorbell-resume")
        cells = {}
        for name, bell in (("blocks", False), ("doorbell", True)):
            # Two videos fill both GPUs, so every image must wait for a block boundary.
            phase = self.cluster_phase("block_elastic", gpus=2, options=dict(sp_transport="export", quantum_ms=100, doorbell=bell,
                                                                             transfer_dir=str(self.root / ("shm-" + name))))
            value = self.run_study(self.config([phase], workers=self.cluster_workers(gpus=2)), name=name)
            self.assert_completed(value)
            cells[name] = value["phases"][0]["cells"][0]
        bell = cells["doorbell"]["doorbell"]
        self.assertTrue(bell["enabled"])
        self.assertGreater(bell["continuous_quanta"], 0)
        # Images arrive while the videos run, so the controller had to ring at least once.
        self.assertGreater(bell["rings"], 0)
        self.assertFalse(cells["blocks"]["doorbell"]["enabled"])
        for name in cells:
            rows = {row["request_id"]: row for row in cells[name]["requests"]}
            self.assertEqual(cells[name]["by_kind"]["image"]["completed"], 4)
            self.assertTrue(all(rows[v]["status"] == "completed" for v in ("v0", "v1")))
        quanta = {name: sum(row["quanta"] for row in cells[name]["requests"] if row["kind"] == "video") for name in cells}
        self.assertLess(quanta["doorbell"], quanta["blocks"])
        # The doorbell files are removed with the cell.
        self.assertEqual(list((self.root / "shm-doorbell").glob("doorbell-*")), [])

    def test_cluster_export_transport_runs_cfg_forms(self):
        # Only CFG pairs beyond SP1: the two videos take a pair each and the ranks receive cfg=2.
        phase = self.cluster_phase("block_elastic", options=dict(sp_transport="export", forms=["sp1", "cfg2"]))
        value = self.run_study(self.config([phase], workers=self.cluster_workers()), name="forms")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertEqual(cell["metrics"]["completed"], 6)
        rows = {row["request_id"]: row for row in cell["requests"]}
        grouped = [row for name, row in rows.items() if name.startswith("v") and max(row["degrees"]) >= 2]
        self.assertTrue(grouped)
        for row in grouped:
            self.assertEqual([form for form in row["forms"] if form != "sp1"], ["cfg2"])
        self.assertTrue(all(row["forms"] == [] for name, row in rows.items() if name.startswith("i")))

    def test_cluster_export_transport_runs_lockstep_pairs(self):
        # Two videos on a pair of GPUs: with pairing on and a profile that hides communication
        # when two requests advance together, they take the group in lockstep.
        phase = self.cluster_phase("block_elastic", gpus=2, options=dict(sp_transport="export", pairing=True, degrees=[2]))
        phase["profile"]["pair_overlap"] = 1
        phase["profile"]["classes"]["video"]["comm_mb_per_step"] = 4000
        value = self.run_study(self.config([phase], workers=self.cluster_workers(2)), name="pairs")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertEqual(cell["metrics"]["completed"], 6)
        rows = {row["request_id"]: row for row in cell["requests"]}
        self.assertTrue(all(rows[v].get("paired_quanta", 0) > 0 for v in ("v0", "v1")))
        self.assertTrue(all("paired_quanta" not in row for name, row in rows.items() if name.startswith("i")))

    def test_cluster_batches_images_on_the_image_worker(self):
        # image_batch 4 with a batch table: the four images arrive together with deadlines that
        # outlast worker start-up, so they go out as one generate_batch call.
        phase = self.cluster_phase("static_sp1", gpus=1, options=dict(image_batch=4))
        phase["profile"]["classes"]["image"]["batch_ms"] = {"2": 25, "4": 30}
        for row in phase["workload"]:
            if row["kind"] == "image":
                row["arrival_ms"], row["slo_ms"] = 0, 20000
        workers = self.cluster_workers(1)
        workers["i0"]["config"]["options"]["costs"]["batch_ms"] = {"2": 25, "4": 30}
        value = self.run_study(self.config([phase], workers=workers), name="batches")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertEqual(cell["metrics"]["completed"], 6)
        rows = {row["request_id"]: row for row in cell["requests"]}
        self.assertTrue(any(row.get("batched", 0) >= 2 for name, row in rows.items() if name.startswith("i")))

    def test_cluster_continuous_batch_late_join_early_exit_and_failure(self):
        def phase(batch):
            value = self.cluster_phase("static_sp1", gpus=1, options=dict(continuous_batch=batch))
            value["output_mode"] = "check"
            value["artifact_dependencies"] = ["test_quality"]
            value["warmup"] = [dict(worker="i0", spec=dict(steps=2, prompt="warmup"))]
            value["profile"]["classes"]["image"].update(steps=2, prepare_ms=0, finish_ms=20,
                batch_step_ms={"1": 100, "2": 100, "4": 100, "8": 100})
            value["workload"] = [dict(request_id=name, kind="image", worker="i0", **{"class": "image"},
                                      arrival_ms=arrival, slo_ms=5000,
                                      spec=dict(steps=2, prompt=name),
                                      output_mode="file" if name == "b" else "check")
                                 for name, arrival in (("a", 0), ("b", 50), ("c", 150))]
            return value

        workers = self.cluster_workers(1)
        workers["i0"]["config"]["options"]["costs"]["step_ms"] = 100
        value = self.run_study(self.config([phase(4)], workers=workers), name="continuous")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertEqual(cell["metrics"]["completed"], 3)
        stats = cell["continuous_batch"]
        self.assertEqual((stats["rounds"], stats["late_joins"], stats["early_exits"]), (4, 2, 2))
        self.assertEqual(stats["batch_sizes"], {"1": 2, "2": 2})
        self.assertEqual(len(stats["samples"]), 4)
        rows = {row["request_id"]: row for row in cell["requests"]}
        self.assertLess(rows["a"]["finish_ms"], rows["b"]["finish_ms"])
        self.assertLess(rows["b"]["finish_ms"], rows["c"]["finish_ms"])
        self.assertEqual([rows[name]["continuous_rounds"] for name in ("a", "b", "c")], [2, 2, 2])
        self.assertEqual([rows[name]["batched"] for name in ("a", "b", "c")], [2, 2, 1])
        for name, mode in (("a", "check"), ("b", "file"), ("c", "check")):
            row = rows[name]
            self.assertEqual(row["generation"]["prompt"], name)
            self.assertEqual(len(row["outputs"]), 1)
            self.assertEqual(row["outputs"][0]["output"]["output_mode"], mode)
            self.assertGreater(row["worker_operation_ms"], 0)
            if mode == "file":
                self.assertTrue(Path(row["outputs"][0]["output"]["path"]).is_file())
            else:
                self.assertNotIn("path", row["outputs"][0]["output"])

        baseline = self.run_study(self.config([phase(1)], workers=workers), name="continuous-baseline")
        self.assert_completed(baseline)
        stats = baseline["phases"][0]["cells"][0]["continuous_batch"]
        self.assertEqual((stats["rounds"], stats["late_joins"], stats["early_exits"]), (6, 0, 0))
        self.assertEqual(stats["batch_sizes"], {"1": 6})

        broken_workers = self.cluster_workers(1)
        broken_workers["i0"]["config"]["options"]["costs"]["step_ms"] = 100
        broken_workers["i0"]["config"]["options"]["fail_ops"] = ["advance_batch"]
        broken_phase = phase(4)
        broken_phase.pop("warmup", None)
        for row in broken_phase["workload"]:
            row["arrival_ms"] = 0
        failed = self.run_study(self.config([broken_phase], workers=broken_workers), name="continuous-failed")
        self.assertEqual(failed["status"], "failed")
        errors = [row for row in failed["failures"] if "request_id" in row]
        self.assertEqual({row["request_id"] for row in errors}, {"a", "b", "c"})
        rows = failed["requests"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "failed" for row in rows))

    def test_cluster_export_transport_failure_stops_the_cell(self):
        workers = self.cluster_workers()
        for name in ("v1", "v3"):
            workers[name]["config"]["options"]["fail_ops"] = ["block_import"]
        phase = self.cluster_phase("block_elastic", options=dict(sp_transport="export"))
        value = self.run_study(self.config([phase], workers=workers), name="export-fail")
        self.assertEqual(value["status"], "failed", value)
        self.assertIn("injected mock block_import failure", json.dumps(value["failures"]))

    def test_cluster_transport_options_are_validated(self):
        for options, name in ((dict(sp_transport="nccl"), "unknown"),):
            with self.subTest(name=name), self.assertRaises(CoreError):
                self.run_study(self.config([self.cluster_phase("block_elastic", options=options)],
                                           workers=self.cluster_workers()), name="transport-" + name)
        three = self.cluster_phase("block_elastic", gpus=3, options=dict(sp_transport="export"))
        with self.assertRaises(CoreError):
            self.run_study(self.config([three], workers=self.cluster_workers(3)), name="transport-three")

    def test_cluster_chunked_decode_completes_across_quanta(self):
        phase = self.cluster_phase("block_elastic", gpus=1, options=dict(quantum_ms=50))
        phase["profile"]["classes"]["video"].update(decode_chunks=4, finish_ms=80)
        workers = self.cluster_workers(1)
        workers["v0"]["config"]["options"].update(decode_chunks=4)
        value = self.run_study(self.config([phase], workers=workers), name="chunked")
        self.assert_completed(value)
        rows = {row["request_id"]: row for row in value["phases"][0]["cells"][0]["requests"]}
        self.assertEqual(value["phases"][0]["cells"][0]["metrics"]["completed"], 6)
        # 4 chunks of 20 ms at a 50 ms quantum: two finish quanta per video.
        self.assertGreater(rows["v0"]["action_ms"]["finish"], 0)
        self.assertTrue(all(len(row["outputs"]) == 1 if "outputs" in row else True for row in rows.values()))

    def test_cluster_static_sp1_and_invalid_slots(self):
        value = self.run_study(self.config([self.cluster_phase("static_sp1", gpus=2)], workers=self.cluster_workers(2)), name="static")
        self.assert_completed(value)
        cell = value["phases"][0]["cells"][0]
        self.assertTrue(all(row["degrees"] == [1] for row in cell["requests"]))
        bad = self.cluster_phase("block_elastic", gpus=2)
        bad["gpus"][1]["video"] = "v0"
        with self.assertRaises(CoreError):
            self.run_study(self.config([bad], workers=self.cluster_workers(2)), name="bad-slots")
        with self.assertRaises(CoreError):
            self.run_study(self.config([self.cluster_phase("fastest", gpus=2)], workers=self.cluster_workers(2)), name="bad-policy")

    def test_simulate_entrypoint_publishes_only_two_files(self):
        config = dict(generator=dict(count=6, seed=9), policies=["static_sp1", "block_elastic"], gpus=2)
        value = api.simulate(config, run_id="sim", work_dir=self.root / "sim",
                             results_root=self.root / "results", core=self.bridge)
        self.assertEqual(value["status"], "completed")
        self.assertEqual([phase["id"] for phase in value["phases"]], ["static_sp1-c0", "block_elastic-c0"])
        for phase in value["phases"]:
            self.assertEqual(phase["result"]["metrics"]["completed"], 6)
            self.assertEqual(phase["result"]["provenance"], "synthetic")
        self.assertEqual(value["budget"]["gpu_seconds"], 0)
        self.assertIn("Synthetic", value["notice"])
        self.assert_archive(value, "synthetic", "sim")


if __name__ == "__main__":
    unittest.main()
