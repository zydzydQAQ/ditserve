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

    PROFILE = dict(provenance="synthetic", sp_efficiency=1, resize_ms=0, links={},
                   classes=dict(video=dict(kind="video", steps=2, blocks_per_step=4, prepare_ms=0, encode_ms=0,
                                           step_compute_ms=20, comm_mb_per_step=0, collectives_per_step=0, finish_ms=5),
                                image=dict(kind="image", service_ms=5)))

    @staticmethod
    def request(request_id="r0", kind="video", **changes):
        value = dict(request_id=request_id, kind=kind, arrival_ms=0, service_ms=20, slo_ms=10000,
                     spec=dict(steps=2, blocks_per_step=4, prompt="a ceramic kettle", seed=7) if kind == "video"
                     else dict(steps=1, prompt="a ceramic kettle", seed=7))
        value["class"] = kind
        value.update(changes)
        return value

    def config(self, phases, options=None, workers=None):
        workers = workers or {"model": {"backend": "mock", "config": {"options": options or {}}},
                              "image": {"backend": "mock", "config": {"options": {}}}}
        return dict(schema="ditserve.study.v1", experiment="native", workers=workers,
                    budget=dict(limit_gpu_seconds=None, watchdog_seconds=10), phases=phases)

    def phase(self, phase_id="serve", **changes):
        value = dict(id=phase_id, kind="cluster", policy="static_sp1", profile=self.PROFILE, options=dict(quantum_ms=20),
                     gpus=[dict(video="model", image="image")], workload=[self.request()])
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
                           matrix={"/policy": ["static_sp1", "block_elastic"]}, paired=True)
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
                counts = row["operation_counts"]
                self.assertEqual((counts["prepare"], counts["finish"], counts["release"]), (1, 1, 1))
                self.assertGreaterEqual(counts["advance"], 2)
        self.assertEqual(self.assert_archive(value)["metrics"], value["metrics"])
        ledger = self.root / "run-budget.json"
        before = ledger.read_bytes()
        resumed = self.run_study(config, resume=True)
        self.assert_completed(resumed)
        self.assertEqual(resumed["phases"], value["phases"])
        self.assertEqual(ledger.read_bytes(), before)
        self.assert_archive(resumed)

    def test_gates_require_actual_evidence_before_next_phase(self):
        first = self.phase("profile", gates=[dict(metric="/cells/0/metrics/completed", op="ge", value=1)])
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

    def test_operations_use_persistent_independent_ranks(self):
        actions = [dict(op="prepare", state_id="sp", spec=dict(steps=2, prompt="rank", seed=0)),
                   dict(op="advance", state_id="sp", steps=1), dict(op="advance", state_id="sp", steps=1),
                   dict(op="release", state_id="sp")]
        phase = dict(id="sp", kind="operations", workers=["rank0", "rank1"], actions=actions)
        workers = {name: dict(backend="mock") for name in phase["workers"]}
        value = self.run_study(self.config([phase], workers=workers))
        self.assert_completed(value)
        operations = value["phases"][0]["cells"][0]["operations"]
        pids = [rank["worker_pid"] for rank in operations[0]["ranks"]]
        self.assertEqual(len(set(pids)), 2)
        for operation in operations:
            self.assertEqual([rank["worker_pid"] for rank in operation["ranks"]], pids)
        self.assertTrue(all(rank["progress"]["done"] and rank["progress"]["step"] == 2 for rank in operations[2]["ranks"]))

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
        changed["phases"][0]["policy"] = "block_elastic"
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
        cases = [self.phase(paired="yes"), self.phase(workload=[]), self.phase(kind="online"),
                 self.phase(gpus=[dict(video="missing", image="image")]),
                 self.phase(worker_config=[]), self.phase(worker_config=dict(device="cuda:0")),
                 self.phase(workload=[self.request(kind="image", **{"class": "video"})]),
                 dict(id="sp", kind="operations", workers=["model", "model"], actions=[dict(op="hello")])]
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
        phase = dict(id="probe", kind="operations", workers=["model"], actions=[dict(op="hello")])
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

    def test_prefetch_encodes_each_video_once_before_it_starts(self):
        phase = self.phase(text_prefetch=2, workload=[self.request("a"), self.request("b"), self.request("c")])
        value = self.run_study(self.config([phase], options=dict(encode_delay_ms=5)))
        self.assert_completed(value)
        for row in value["requests"]:
            self.assertGreaterEqual(row["encode_ready_ms"], row["encode_submit_ms"])
            self.assertGreaterEqual(row["start_ms"], row["encode_ready_ms"])
            self.assertEqual(row["operation_counts"]["encode_start"], 1)
            self.assertGreaterEqual(row["operation_counts"]["encode_poll"], 1)
            self.assertEqual(row["operation_counts"]["prepare"], 1)
            self.assertEqual(len(row["outputs"]), 1)

    def test_warmup_runs_before_trace_clock_and_images_skip_prefetch(self):
        phase = self.phase(text_prefetch=1,
                           warmup=[dict(worker="model", spec=dict(steps=2)), dict(worker="image", spec=dict(steps=1))],
                           workload=[self.request("v"), self.request("i", kind="image")])
        value = self.run_study(self.config([phase], options=dict(encode_delay_ms=5)))
        self.assert_completed(value)
        rows = {row["request_id"]: row for row in value["requests"]}
        self.assertEqual(rows["v"]["operation_counts"]["encode_start"], 1)
        self.assertNotIn("encode_start", rows["i"]["operation_counts"])
        warmed = value["phases"][0]["cells"][0]["warmup"]
        self.assertEqual([item["worker"] for item in warmed], ["model", "image"])
        self.assertTrue(all(item["elapsed_ms"] >= 0 for item in warmed))
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
        value = dict(id="cluster", kind="cluster", policy=policy, profile=profile, text_prefetch=1, options=dict(quantum_ms=200),
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
