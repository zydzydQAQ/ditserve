"""Native controller observation deadlines, complete cohorts and verified cleanup."""

import copy
from pathlib import Path
import sys
import unittest

import test_study as fixtures
from ditserve.runtime.client import CoreBridge, CoreError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from p2_benefit import expanded
from p3_window_review import cutoff_receipt


@unittest.skipUnless(CoreBridge().available(), "build the independent native core first")
class ReplayCutoffTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeStudyTest.setUpClass.__func__)
    setUp = fixtures.NativeStudyTest.setUp
    config = fixtures.NativeStudyTest.config
    cluster_phase = fixtures.NativeStudyTest.cluster_phase
    cluster_workers = fixtures.NativeStudyTest.cluster_workers
    run_study = fixtures.NativeStudyTest.run_study

    def fixture(self, *, warm=True, paired=False):
        phase = self.cluster_phase("block-cfg2", gpus=2, text_prefetch=0,
            options=dict(sp_transport="export", quantum_blocks=1, doorbell=True, lazy_preemption=False))
        phase["replay_cutoff"] = dict(schema="ditserve.replay-cutoff.v1", after_ms=250., mode="cancel_and_censor")
        phase["profile"]["classes"]["video"].update(steps=20, step_compute_ms=400., finish_ms=0)
        phase["workload"] = [
            dict(request_id="delivered", kind="image", **{"class": "image"}, arrival_ms=0., slo_ms=150., spec=dict(steps=1)),
            dict(request_id="active", kind="video", **{"class": "video"}, arrival_ms=30., slo_ms=20000.,
                 spec=dict(steps=20, blocks_per_step=4)),
            dict(request_id="future", kind="image", **{"class": "image"}, arrival_ms=350., slo_ms=1000., spec=dict(steps=1)),
        ]
        phase["matrix"] = {"/policy": ["block-cfg2", "edf-cfg2"]}
        phase["paired"] = paired
        if warm:
            phase["warmup"] = [dict(worker="v0", spec=dict(steps=1, blocks_per_step=1))]
        workers = self.cluster_workers(2)
        for index in range(2):
            workers[f"v{index}"]["config"]["options"]["costs"] = dict(step_ms=400., finish_ms=0.)
        return phase, workers

    def assert_cleanup(self, record):
        self.assertIsNone(record["budget"]["active"])
        self.assertTrue(all(session["cleanup_verified"] for session in record["budget"]["sessions"]))
        self.assertEqual(record["budget"]["used_gpu_seconds"], 0)
        self.assertEqual(record["phases"][0]["final_occupancy"]["owned"], [])

    def assert_receipts(self, record):
        """Run the actual P3 evidence parser before the fixture is removed."""
        planned = expanded(record["config"]["phases"][0])
        phase = record["phases"][0]
        for position, cell in enumerate(phase["cells"]):
            if "replay_cutoff" not in cell:
                continue
            selector = dict(phase=phase["id"], attempt=phase["attempt"], cell=cell["cell"], order=cell["order"])
            rows = [row for row in record["requests"] if all(row[key] == value for key, value in selector.items())]
            arm = dict(config=record["config"], phase=planned[cell["cell"]], selector=selector,
                       position=position, workload=planned[cell["cell"]]["workload"])
            self.assertIs(cutoff_receipt(record, arm, cell, rows), cell["replay_cutoff"])

    def test_cutoff_starts_after_warmup_and_preserves_every_policy_cohort(self):
        phase, workers = self.fixture()
        record = self.run_study(self.config([phase], workers=workers), name="cutoff-cohorts")
        self.assertEqual(record["status"], "censored", record.get("failures"))
        self.assertEqual(len(record["requests"]), 6)
        self.assert_cleanup(record)
        self.assert_receipts(record)
        self.assertFalse(record["phases"][0]["qualification"]["qualified"])
        previous_end = None
        for position, cell in enumerate(record["phases"][0]["cells"]):
            self.assertEqual(cell["status"], "censored")
            self.assertGreaterEqual(cell["warmup"][0]["elapsed_ms"], 350.)
            receipt = cell["replay_cutoff"]
            self.assertEqual(receipt["status"], "triggered")
            self.assertEqual(receipt["observation_horizon_ms"], 250.)
            self.assertEqual(receipt["config_sha256"], record["config_sha256"])
            self.assertEqual(receipt["observation_id"], f"cluster/1/{position}")
            self.assertEqual(receipt["policy"], cell["policy"])
            self.assertEqual(receipt["counts"], dict(planned=3, completed=1, failed=0,
                censored=1, not_executed=1, interrupted=0))
            self.assertFalse(receipt["physical_stop_bound"])
            self.assertTrue(receipt["cleanup_verified"])
            self.assertTrue(cell["cleanup_verified"])
            self.assertEqual(cell["final_occupancy"]["owned"], [])
            self.assertGreaterEqual(receipt["observed_at_ms"], 250.)
            self.assertGreaterEqual(receipt["termination_finished_ms"], receipt["observed_at_ms"])
            origin = receipt["replay_origin_monotonic_seconds"]
            if previous_end is not None:
                self.assertGreaterEqual(origin, previous_end)
            previous_end = origin + receipt["termination_finished_ms"] / 1000
            for row in cell["requests"]:
                self.assertEqual(row["observation_ms"], 250.)
                if row["status"] == "completed":
                    self.assertLessEqual(row["finish_ms"], 250.)
                else:
                    self.assertNotIn("finish_ms", row)
        self.assertEqual(record["metrics"]["requests"], 6)
        self.assertEqual(record["metrics"]["censored"], 2)
        self.assertEqual(record["metrics"]["not_executed"], 2)

    def test_real_model_failure_keeps_later_forward_and_reverse_denominators(self):
        phase, workers = self.fixture(warm=False, paired=True)
        phase["replay_cutoff"]["after_ms"] = 1000.
        phase["workload"] = [phase["workload"][1], phase["workload"][2]]
        phase["workload"][0]["arrival_ms"] = 0.
        phase["workload"][1]["arrival_ms"] = 2000.
        for index in range(2):
            workers[f"v{index}"]["config"]["options"]["fail_on"] = "advance"
        record = self.run_study(self.config([phase], workers=workers), name="cutoff-model-failure")
        self.assertEqual(record["status"], "failed")
        self.assert_cleanup(record)
        self.assert_receipts(record)
        cells = record["phases"][0]["cells"]
        self.assertEqual([cell["status"] for cell in cells], ["failed"] + ["not_executed"] * 3)
        self.assertEqual(cells[0]["replay_cutoff"]["status"], "not_reached")
        self.assertIn("injected mock advance failure", cells[0]["error"])
        self.assertEqual(len(record["requests"]), 8)
        self.assertEqual(sum(row["status"] == "failed" for row in record["requests"]), 1)
        self.assertEqual(sum(row["status"] == "not_executed" for row in record["requests"]), 7)
        self.assertEqual({row["observation_id"] for row in record["requests"]}, {f"cluster/1/{i}" for i in range(4)})

    def test_watchdog_before_replay_cutoff_is_a_failure(self):
        phase, workers = self.fixture(warm=False)
        phase["watchdog_seconds"] = .4
        phase["replay_cutoff"]["after_ms"] = 2000.
        phase["workload"] = [phase["workload"][1]]
        phase["workload"][0]["arrival_ms"] = 0.
        for index in range(2):
            workers[f"v{index}"]["config"]["options"]["costs"]["step_ms"] = 4000.
        record = self.run_study(self.config([phase], workers=workers), name="cutoff-watchdog")
        self.assertEqual(record["status"], "failed")
        self.assert_cleanup(record)
        self.assert_receipts(record)
        self.assertEqual(len(record["requests"]), 2)
        self.assertFalse(any(row["status"] == "censored" for row in record["requests"]))
        self.assertFalse(any(cell.get("replay_cutoff", {}).get("status") == "triggered"
                             for cell in record["phases"][0]["cells"]))

    def test_real_rank_failure_survives_peer_cutoff(self):
        phase, workers = self.fixture(warm=False)
        phase["workload"] = [phase["workload"][1], phase["workload"][2]]
        phase["workload"][0]["arrival_ms"] = 0.
        phase["workload"][1]["arrival_ms"] = 2000.
        workers["v0"]["config"]["options"]["fail_on"] = "advance"
        workers["v1"]["config"]["options"]["costs"]["step_ms"] = 4000.
        record = self.run_study(self.config([phase], workers=workers), name="cutoff-real-fault")
        self.assertEqual(record["status"], "failed")
        self.assert_cleanup(record)
        self.assert_receipts(record)
        cells = record["phases"][0]["cells"]
        self.assertEqual([cell["status"] for cell in cells], ["failed", "not_executed"])
        self.assertEqual(cells[0]["replay_cutoff"]["status"], "triggered")
        self.assertIn("injected mock advance failure", cells[0]["error"])
        self.assertEqual(len(record["requests"]), 4)
        self.assertEqual(sum(row["status"] == "failed" for row in record["requests"]), 1)
        self.assertEqual(sum(row["status"] == "not_executed" for row in record["requests"]), 3)

    def test_default_is_off_and_early_completion_is_not_truncation(self):
        for enabled in (False, True):
            phase, workers = self.fixture(warm=False)
            phase["workload"] = [phase["workload"][0]]
            if not enabled:
                phase.pop("replay_cutoff")
            record = self.run_study(self.config([phase], workers=workers), name=f"cutoff-early-{enabled}")
            self.assertEqual(record["status"], "completed")
            self.assert_cleanup(record)
            self.assert_receipts(record)
            for cell in record["phases"][0]["cells"]:
                if enabled:
                    self.assertEqual(cell["replay_cutoff"]["status"], "not_reached")
                    self.assertLess(cell["replay_cutoff"]["observation_horizon_ms"], 250.)
                else:
                    self.assertNotIn("replay_cutoff", cell)

    def test_cutoff_schema_and_matrix_identity_fail_before_worker_start(self):
        phase, workers = self.fixture(warm=False)
        variants = []
        for field, value in [("after_ms", 0), ("after_ms", -1), ("mode", "watchdog"), ("schema", "other")]:
            changed = copy.deepcopy(phase); changed["replay_cutoff"][field] = value; variants.append([changed])
        changed = copy.deepcopy(phase); changed["matrix"]["/replay_cutoff/after_ms"] = [251.]; variants.append([changed])
        changed = copy.deepcopy(phase); changed["matrix"]["/kind"] = ["online"]; variants.append([changed])
        changed = copy.deepcopy(phase); changed["matrix"]["/actions"] = [[]]; variants.append([changed])
        changed = copy.deepcopy(phase); changed["mps"] = dict(server_pid=1, pipe_directory="/tmp"); variants.append([changed])
        other = copy.deepcopy(phase); other["id"] = "other"; variants.append([phase, other])
        for index, phases in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(CoreError):
                self.run_study(self.config(phases, workers=workers), name=f"invalid-cutoff-{index}")
            self.assertFalse((self.root / f"invalid-cutoff-{index}").exists())


if __name__ == "__main__":
    unittest.main()
