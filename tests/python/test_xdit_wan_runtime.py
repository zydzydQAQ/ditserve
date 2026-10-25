"""Execution-level xDiT Wan cache, state, reference and decode contracts."""

import importlib.util
import unittest


PINNED = all(importlib.util.find_spec(name) is not None
             for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang"))


class SnapshotBackend:
    def __init__(self, worker, state):
        from ditserve.models.base import Backend
        from ditserve.models.wan.backend import WanBackend
        self.backend = WanBackend.__new__(WanBackend)
        Backend.__init__(self.backend, {})
        self.backend.runtime = worker
        self.backend.states = {state.request_id: state}
        self.backend.continuations = {}
        self.backend.parallel = self.backend.decode_state = None

    def snapshot(self, state, mode):
        return self.backend.snapshot(dict(state_id=state.request_id, mode=mode))

    def restore(self, state):
        return self.backend.restore(dict(state_id=state.request_id))


def snapshot_backend(worker, state):
    return SnapshotBackend(worker, state)


@unittest.skipUnless(PINNED, "pinned xDiT model environment is required")
class XDitWanRuntimeTests(unittest.TestCase):
    def test_cache_snapshots_and_certified_migration_continue_exactly(self):
        import torch
        from ditserve.models.wan.migration import local_roundtrip
        from ditserve.models.wan.parallel import request_engine
        from ditserve.models.wan.state import refresh_certificate
        from xdit_wan_fixtures import runtime_worker

        native, expected = runtime_worker(torch, cached=True, xdit=False, layers=12)
        worker, state = runtime_worker(torch, cached=True, layers=12)
        for unused in range(10):
            native.step(expected)
            event = worker.step(state)
            self.assertTrue(torch.equal(state.latents, expected.latents))
        self.assertGreater(event["reused_forwards"], 0)

        snapshot_backend(worker, state).snapshot(state, "hot")
        self.assertEqual(state.pending_snapshot["engine"]["revision"], worker.xdit.revision)
        snapshot_backend(worker, state).restore(state)
        native.step(expected)
        worker.step(state)
        self.assertTrue(torch.equal(state.latents, expected.latents))

        state, evidence = local_roundtrip(worker, state, action="full_migrate", migration_id="full-xdit")
        native.step(expected)
        worker.step(state)
        self.assertTrue(torch.equal(state.latents, expected.latents))
        certificate = refresh_certificate(state, engine=request_engine(worker))
        while not certificate["eligible"] and state.step_index < state.spec["steps"]:
            native.step(expected)
            worker.step(state)
            self.assertTrue(torch.equal(state.latents, expected.latents))
            certificate = refresh_certificate(state, engine=request_engine(worker))
        self.assertTrue(certificate["eligible"], certificate["reasons"])
        state, evidence = local_roundtrip(worker, state, action="certified_omit",
                                          migration_id="certified-xdit", certificate=certificate)
        native.step(expected)
        worker.step(state)
        self.assertTrue(torch.equal(state.latents, expected.latents))

        snapshot_backend(worker, state).snapshot(state, "cold")
        self.assertTrue(state.needs_anchor)
        self.assertTrue(all(not context.buffers for context in state.contexts.values()))
        snapshot_backend(worker, state).restore(state)
        event = worker.step(state)
        self.assertTrue(event["forced_cold_anchor"])
        self.assertEqual(event["reused_forwards"], 0)

    def test_storage_snapshot_preserves_aliases_and_rejects_wrong_engine(self):
        import torch
        from ditserve.models.wan.parallel import ParallelContractError
        from ditserve.models.wan.state import state_inventory
        from ditserve.models.wan.storage import restore_timestep_views, snapshot_timestep_views
        from ditserve.models.xdit import XDIT_REVISION
        from xdit_wan_fixtures import runtime_worker

        native, expected = runtime_worker(torch, xdit=False, steps=4)
        worker, state = runtime_worker(torch, steps=4)
        for unused in range(2):
            native.step(expected)
            worker.step(state)
        snapshot_timestep_views(worker, state, "hot", state_inventory(state))
        self.assertEqual(state.pending_snapshot["engine"]["revision"], XDIT_REVISION)
        state.pending_snapshot["engine"]["revision"] = "wrong"
        with self.assertRaisesRegex(ParallelContractError, "engine mismatch"):
            restore_timestep_views(worker, state)
        state.pending_snapshot["engine"]["revision"] = XDIT_REVISION
        restore_timestep_views(worker, state)
        for view in state.scheduler.timestep_list:
            if view is not None:
                self.assertEqual(view.untyped_storage().data_ptr(),
                                 state.scheduler.timesteps.untyped_storage().data_ptr())
        native.step(expected)
        worker.step(state)
        self.assertTrue(torch.equal(state.latents, expected.latents))

    def test_official_reference_and_chunked_decode_match(self):
        import numpy as np
        import torch
        from xdit_wan_fixtures import runtime_worker

        native, state = runtime_worker(torch, xdit=False, steps=2)
        native.release(state)
        reference = native.reference(state.spec)

        worker, state = runtime_worker(torch, steps=2)
        for unused in range(2):
            worker.step(state)
        self.assertTrue(torch.equal(state.latents, reference["latents"]))
        np.testing.assert_array_equal(worker.finish(state), reference["output"])

        chunked, state = runtime_worker(torch, steps=2)
        for unused in range(2):
            chunked.step(state)
        output, info = chunked.decode_chunks(state, 1)
        self.assertIsNone(output)
        self.assertFalse(info["done"])
        output, info = chunked.decode_chunks(state, 1)
        self.assertTrue(info["done"])
        np.testing.assert_array_equal(output, reference["output"])


if __name__ == "__main__":
    unittest.main()
