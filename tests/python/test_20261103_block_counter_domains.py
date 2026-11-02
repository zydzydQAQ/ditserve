"""Reject ambiguous native/mock block units before starting any worker."""

import unittest

import test_study as fixtures
from ditserve.runtime.client import CoreBridge, CoreError


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class BlockCounterDomainsTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeStudyTest.setUpClass.__func__)
    setUp = fixtures.NativeStudyTest.setUp
    config = fixtures.NativeStudyTest.config
    cluster_phase = fixtures.NativeStudyTest.cluster_phase
    cluster_workers = fixtures.NativeStudyTest.cluster_workers
    run_study = fixtures.NativeStudyTest.run_study

    def test_mixed_video_counters_reject_before_admission_in_either_slot_order(self):
        for native in range(2):
            with self.subTest(native_slot=native):
                phase = self.cluster_phase("block-cfg2", gpus=2,
                                           options=dict(sp_transport="export"))
                workers = self.cluster_workers(2)
                workers[f"v{native}"]["backend"] = "wan"
                name = f"mixed-counters-{native}"
                with self.assertRaisesRegex(CoreError, "cannot mix Wan rank-local and non-Wan block counters"):
                    self.run_study(self.config([phase], workers=workers), name=name)
                self.assertFalse((self.root / name).exists())
                self.assertFalse((self.root / f"{name}-budget.json").exists())

    def test_all_wan_counters_preserve_the_real_gpu_admission_guard(self):
        phase = self.cluster_phase("block-cfg2", gpus=2,
                                   options=dict(sp_transport="export"))
        workers = self.cluster_workers(2)
        for index in range(2):
            workers[f"v{index}"]["backend"] = "wan"
        with self.assertRaisesRegex(CoreError, "real model execution requires --allow-gpu"):
            self.run_study(self.config([phase], workers=workers), name="all-wan-counters")


if __name__ == "__main__":
    unittest.main()
