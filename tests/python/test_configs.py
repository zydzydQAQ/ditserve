"""Every shipped study binds to every shipped environment, or fails for a stated reason.

Resolution and validation only: fake GPU inventories, no worker or GPU is touched.
"""

import json
import unittest

from ditserve.runtime import CONFIGS
from ditserve.runtime.client import CoreBridge, CoreError, resolve

A100, V100 = "GPU-3ab2fedc-ccc1-007d-f4a1-b6058b3479a3", "GPU-a058066e-bde9-c534-efb1-d2a68c117681"


def gpu(index, uuid, name, memory):
    return dict(index=index, uuid=uuid, name=name, memory_mib=memory, driver_version="570.181")


INVENTORIES = {
    "a100-1x40g": [gpu(0, A100, "NVIDIA A100-PCIE-40GB", 40960), gpu(1, V100, "Tesla V100-PCIE-32GB", 32768)],
    "rtx4090-2x24g": [gpu(i, "GPU-%08d-0000-0000-0000-000000004090" % i, "NVIDIA GeForce RTX 4090", 24564) for i in range(2)],
    "a100-8x40g": [gpu(i, "GPU-%08d-0000-0000-0000-0000000a100f" % i, "NVIDIA A100-PCIE-40GB", 40960) for i in range(8)],
}
# Expected refusals: a serving study needs a measured profile, SP needs two GPUs.
REFUSED = {("serving", "rtx4090-2x24g"): "cost_profile", ("serving", "a100-8x40g"): "cost_profile",
           ("sp_equivalence", "a100-1x40g"): "no phase"}


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class ShippedConfigTests(unittest.TestCase):
    def bind(self, study, env):
        return resolve(str(CONFIGS / "studies" / (study + ".json")), env=env, gpu_inventory=INVENTORIES[env])

    def test_every_study_binds_or_is_refused_for_its_reason(self):
        studies = sorted(path.stem for path in (CONFIGS / "studies").glob("*.json"))
        self.assertGreaterEqual(len(studies), 6)
        for study in studies:
            for env in INVENTORIES:
                with self.subTest(study=study, env=env):
                    if (study, env) in REFUSED:
                        with self.assertRaisesRegex(CoreError, REFUSED[study, env]):
                            self.bind(study, env)
                        continue
                    resolved = self.bind(study, env)
                    self.assertEqual(resolved["environment"]["name"], env)
                    self.assertTrue(resolved["experiment"].startswith(env + "-"))
                    text = json.dumps(resolved)
                    self.assertNotIn("@env.", text)
                    self.assertNotIn(V100, json.dumps(resolved["workers"]))

    def test_the_local_host_never_binds_the_v100(self):
        resolved = self.bind("acceptance", "a100-1x40g")
        uuids = {worker.get("gpu_uuid") for worker in resolved["workers"].values()}
        self.assertEqual(uuids, {A100})
        self.assertEqual(resolved["environment"]["selected"], [A100])

    def test_two_gpus_double_the_slots_and_workload(self):
        one, two = self.bind("block_exact", "a100-1x40g"), self.bind("block_exact", "rtx4090-2x24g")
        self.assertEqual([phase["id"] for phase in one["phases"]], ["same_gpu"])
        self.assertEqual([phase["id"] for phase in two["phases"]], ["same_gpu", "cross_gpu"])
        self.assertEqual(one["environment"]["unplaced_workers"], [dict(worker="remote", gpu=1)])
        serving = self.bind("serving", "a100-1x40g")
        self.assertEqual(len(serving["phases"][0]["gpus"]), 1)
        self.assertEqual(serving["phases"][0]["generator"]["count_per_gpu"], 30)
        acceptance = self.bind("acceptance", "rtx4090-2x24g")
        self.assertEqual(acceptance["phases"][0]["workers"], ["model.0", "model.1"])
        self.assertEqual(len(acceptance["phases"][2]["gpus"]), 2)
        self.assertEqual(acceptance["workers"]["wan.1"]["environment"]["NCCL_P2P_DISABLE"], "1")


if __name__ == "__main__":
    unittest.main()
