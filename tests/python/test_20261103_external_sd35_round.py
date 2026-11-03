"""CPU continuation protocol tests; these do not claim model numerical validation."""
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_sd35_round as sd


class SD35RoundProtocolTests(unittest.TestCase):
    def state(self):
        request = sd.ImageRequest('a red fox', negative_prompt='blur', steps=5)
        return sd.RoundState(request, ['u', 'c'], ['pooled_u', 'pooled_c'], 1.0,
                             [1000., 800., 600., 400., 200.], [1., .8, .6, .4, .2, 0.])

    def test_continuation_keeps_full_time_table_and_endpoint_sigma(self):
        initial = self.state()
        first_t, first_s = initial.window(2)
        self.assertEqual(first_t, [1000., 800.])
        self.assertEqual(first_s, [1., .8, .6])
        later = replace(initial, cursor=2, latents=2.0)
        last_t, last_s = later.window(5)
        self.assertEqual(last_t, [600., 400., 200.])
        self.assertEqual(last_s, [.6, .4, .2, 0.])
        self.assertEqual(first_t + last_t, initial.timesteps)
        self.assertEqual(first_s[-1], last_s[0])
        self.assertIs(later.timesteps, initial.timesteps)
        self.assertIs(later.sigmas, initial.sigmas)
        self.assertEqual(len(later.transferred_tensors()), 5)
        self.assertEqual(initial.cursor, 0)
        self.assertEqual(later.request.negative_prompt, 'blur')
        self.assertEqual(later.request.guidance_scale, 4.5)

    def test_segmented_time_dependent_euler_oracle_preserves_every_interval(self):
        # Independent scalar oracle verifies the protocol slices, not a model or
        # implementation substitute. Every t and sigma difference affects output.
        state = self.state()
        def solve(value, times, sigmas):
            for t, a, b in zip(times, sigmas, sigmas[1:]):
                value += (b - a) * (2.0 * value + t / 1000.)
            return value
        whole = solve(state.latents, state.timesteps, state.sigmas)
        current = state
        for end in (1, 3, 5):
            times, sigmas = current.window(end)
            current = replace(current, latents=solve(current.latents, times, sigmas), cursor=end)
        self.assertEqual(current.latents, whole)
        self.assertNotEqual(solve(state.latents, state.timesteps[:2] + state.timesteps[:3], state.sigmas), whole)

    def test_replayed_empty_overrun_boolean_and_truncated_state_rejected(self):
        state = replace(self.state(), cursor=2)
        for end in (2, 1, 6, True, 3.0):
            with self.assertRaises(ValueError):
                state.window(end)
        for cursor in (-1, 6, True, 1.0):
            with self.assertRaises(ValueError):
                replace(state, cursor=cursor)
        with self.assertRaisesRegex(ValueError, 'complete original schedule'):
            replace(state, timesteps=state.timesteps[2:], sigmas=state.sigmas[2:])

    def test_cfg_wire_pairs_survive_worker_chunk_transfer_and_rebatch(self):
        # Official runner stores chunk(batch_size), then concatenates received
        # request tensors. Adjacent [u,c] pairs are required for this to be sound.
        per_request = [('u0', 'c0'), ('u1', 'c1'), ('u2', 'c2')]
        wire = sum(per_request, ())
        self.assertEqual([wire[i:i+2] for i in range(0, 6, 2)], per_request)
        grouped = tuple(wire[i] for i in sd.cfg_order(3))
        self.assertEqual(grouped, ('u0', 'u1', 'u2', 'c0', 'c1', 'c2'))
        migrated = per_request[2] + per_request[0]
        self.assertEqual(tuple(migrated[i] for i in sd.cfg_order(2)), ('u2', 'u0', 'c2', 'c0'))
        for size in (0, -1, True, 2.0):
            with self.assertRaises(ValueError):
                sd.cfg_order(size)

    def test_schedule_finite_strictly_decreasing_and_endpoint_required(self):
        sd.validate_schedule([1000., 500.], [1., .5, 0.])
        for t, s in [([], [0.]), ([1.], [1.]), ([1., 1.], [1., .5, 0.]),
                     ([1.], [0., 1.]), ([1.], [1., -.1]),
                     ([float('nan')], [1., 0.]), ([1.], [float('inf'), 0.]),
                     ([True], [1., 0.])]:
            with self.assertRaises(ValueError):
                sd.validate_schedule(t, s)

    def test_request_contract_and_solver_history_fail_closed(self):
        sd.validate_solver(1, {'stochastic_sampling': False})
        for order, config in [(2, {}), (True, {}), (1, {'stochastic_sampling': True})]:
            with self.assertRaises(ValueError):
                sd.validate_solver(order, config)
        for change in ({'height': 513}, {'width': 0}, {'steps': True}, {'seed': -1},
                       {'seed': True}, {'max_sequence_length': 513}, {'guidance_scale': 1.0},
                       {'guidance_scale': float('nan')}, {'negative_prompt': None}):
            with self.assertRaises(ValueError):
                sd.ImageRequest('a fox', **change)

    def test_immutable_profile_rejects_guidance_and_text_length_drift(self):
        # No Torch, fake runner or model instance is constructed for this protocol test.
        profile = type('Profile', (), {'guidance_scale': 4.5, 'max_sequence_length': 256})()
        sd.SD35RoundPipeline._profile(profile, 4.5, 256)
        with self.assertRaisesRegex(ValueError, 'guidance'):
            sd.SD35RoundPipeline._profile(profile, 7.0, 256)
        with self.assertRaisesRegex(ValueError, 'text length'):
            sd.SD35RoundPipeline._profile(profile, 4.5, 512)

    def test_non_frozen_backend_cannot_be_loaded(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'frozen 51740c0'):
                sd.verify_shared_sources(directory)


if __name__ == '__main__':
    unittest.main()
