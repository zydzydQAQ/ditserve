"""Use a real UniPC trajectory to exercise aliases absent from fake solvers."""

import copy
import unittest

from ditserve.research.experiment import state_digest
import test_research_adapter as fixtures


class UniPCTimestepStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ResearchAdapterTests.setUpClass()
        cls.torch = fixtures.ResearchAdapterTests.torch

    def real_state(self):
        from diffusers import UniPCMultistepScheduler
        state, worker, adapter = fixtures.ResearchAdapterTests().fixture()
        state.cache_enabled = state.spec["cache"] = False
        state.spec["threshold"] = 0.
        state.contexts = {}
        state.scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
        state.scheduler.set_timesteps(32, device="cpu")
        state.latents = self.torch.zeros(1, 4, 2, 2)
        for index in range(16):
            state.latents = state.scheduler.step(self.torch.ones_like(state.latents),
                state.scheduler.timesteps[index], state.latents, return_dict=False)[0]
        state.step_index = 16
        return state, worker, adapter

    def test_real_unipc_views_keep_values_aliases_and_future_trajectory(self):
        state, worker, adapter = self.real_state()
        baseline_scheduler = copy.deepcopy(state.scheduler)
        baseline_latents = state.latents.clone()
        before = state_digest(worker, state)
        inventory = adapter.inventory(state)
        self.assertTrue(inventory["transfer_supported"])
        aliases = [r for r in inventory["storages"] if r["distinct_view_alias"]]
        self.assertEqual(len(aliases), 1)
        self.assertTrue(aliases[0]["supported_timestep_views"])
        snapshot = adapter.snapshot(state, "hot")
        self.assertEqual(snapshot["storage_codec"], "known_unipc_timestep_views")
        adapter.restore(state)
        self.assertEqual(state_digest(worker, state), before)
        for scalar in state.scheduler.timestep_list:
            self.assertEqual(scalar.untyped_storage().data_ptr(), state.scheduler.timesteps.untyped_storage().data_ptr())
        for index in range(16, 32):
            baseline_latents = baseline_scheduler.step(self.torch.ones_like(baseline_latents),
                baseline_scheduler.timesteps[index], baseline_latents, return_dict=False)[0]
            state.latents = state.scheduler.step(self.torch.ones_like(state.latents),
                state.scheduler.timesteps[index], state.latents, return_dict=False)[0]
            self.assertTrue(self.torch.equal(state.latents, baseline_latents))
        worker.snapshot.assert_not_called()
        worker.restore.assert_not_called()

    def test_known_timestep_views_do_not_allow_unknown_latent_views(self):
        state, worker, adapter = self.real_state()
        state.scheduler.last_sample = state.latents[:]
        with self.assertRaisesRegex(ValueError, "distinct tensor views"):
            adapter.snapshot(state, "hot")
        worker.snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
