import importlib.util
import unittest
from types import SimpleNamespace

from ditserve.models.xdit import collectives
from ditserve.models.xdit.process_group import PROCESS_GROUP


class CollectiveBackendTest(unittest.TestCase):
    def tearDown(self):
        collectives.shutdown()

    def test_nccl_is_default_and_never_intercepts(self):
        self.assertEqual(collectives.backend(), "nccl")
        self.assertIsNone(collectives.all_to_all_flat(object(), object()))
        self.assertFalse(collectives.all_gather_into(object(), object(), (0, 1)))
        self.assertIsNone(collectives.all_gather_chunks(object(), (0, 1)))
        self.assertFalse(collectives.all_reduce_max(object(), (0, 1)))
        self.assertEqual(collectives.configure("nccl"), {"backend": "nccl"})
        self.assertIsNone(PROCESS_GROUP.ALL_TO_ALL)

    def test_unknown_backend_and_missing_world_fail_closed(self):
        with self.assertRaises(ValueError):
            collectives.configure("mpi")
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")
        with self.assertRaises(RuntimeError):
            collectives.configure("gfc", device="cuda:0")   # no initialized world group
        self.assertEqual(collectives.backend(), "nccl")

    def test_unbound_group_is_rejected_under_gfc(self):
        collectives._BACKEND = "gfc"
        try:
            with self.assertRaisesRegex(RuntimeError, "never bound"):
                collectives._ranks_of(SimpleNamespace())
            group = SimpleNamespace()
            collectives.bind(group, [2, 3])
            self.assertEqual(collectives._ranks_of(group), (2, 3))
        finally:
            collectives._BACKEND = "nccl"

    def test_xfuser_hooks_route_through_overrides(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")
        import torch
        try:
            from xfuser.model_executor.layers import usp
            from xfuser.core.distributed.group_coordinator import GroupCoordinator
        except Exception as error:  # xfuser import needs its optional dependencies
            self.skipTest("xfuser unavailable: %r" % (error,))
        seen = []
        PROCESS_GROUP.ALL_TO_ALL = lambda flat, group: (seen.append(flat.numel()), flat.flip(0))[1]
        try:
            x = torch.arange(12, dtype=torch.float16).reshape(3, 4)
            out = usp._sdpa_all_to_all_single(x)
            self.assertEqual(seen, [12])
            self.assertTrue(torch.equal(out, x.flatten().flip(0).reshape(3, 4)))
        finally:
            PROCESS_GROUP.ALL_TO_ALL = None
        coordinator = GroupCoordinator.__new__(GroupCoordinator)
        coordinator.world_size = 2
        coordinator.device_group = None
        coordinator.all_gather_override = lambda output, tensor: (output.copy_(torch.cat([tensor, tensor])), True)[1]
        gathered = coordinator.all_gather(torch.ones(2, 3), dim=0)
        self.assertEqual(tuple(gathered.shape), (4, 3))

    def test_vendored_gfc_is_present_with_license(self):
        from pathlib import Path
        root = Path(collectives.__file__).resolve().parents[3] / "gfc"
        self.assertTrue((root / "LICENSE").is_file() and (root / "VENDORED.md").is_file())
        self.assertIn("MIT", (root / "LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
