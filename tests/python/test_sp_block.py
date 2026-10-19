"""Ulysses SP block quanta and step-boundary regroups on a tiny CPU Wan over Gloo.

Exactness here is a CPU contract for the arithmetic and the control flow, not GPU evidence.
"""

import importlib.util
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace

try:  # discover puts tests/python on sys.path; module-path runs do not
    from test_model_state import empty_target, fixture
except ImportError:
    from tests.python.test_model_state import empty_target, fixture

PINNED = all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "cache_dit"))
STEP_BLOCKS = 6  # three blocks per CFG branch


def tiny_model(torch):
    from diffusers import WanTransformer3DModel
    torch.manual_seed(907)
    return WanTransformer3DModel(num_attention_heads=4, attention_head_dim=12, in_channels=16, out_channels=16,
                                 num_layers=3, ffn_dim=24, freq_dim=4, text_dim=4096, patch_size=(1, 2, 2),
                                 rope_max_seq_len=16).eval().to(torch.float16)


def plain_executor(torch, *, empty=False, expand=False):
    from ditserve.models.wan.block import WanBlockExecutor
    if empty:
        worker, state = empty_target(), None
    else:
        worker, state = fixture(cached=False, boundary=2)
    worker.pipe, worker.attention = SimpleNamespace(transformer=tiny_model(torch)), "math"
    worker.expand_timesteps = expand
    return worker, state, WanBlockExecutor(worker)


def sp_executor(torch, rank, ranks, *, empty=False, expand=False, cfg=1, interleave=False):
    """One rank's worker with the Ulysses attachment over the Gloo world."""
    from ditserve.models.wan.block import WanBlockExecutor
    from ditserve.models.wan.parallel import LayoutHandle, build_fixed_groups
    from ditserve.models.wan.sp import PinnedWanUlyssesBackend
    if empty:
        worker, state = empty_target(), None
    else:
        worker, state = fixture(cached=False, boundary=2)
    worker.pipe, worker.attention = SimpleNamespace(transformer=tiny_model(torch)), "math"
    worker.expand_timesteps = expand   # Wan2.2 TI2V per-token timesteps
    groups = build_fixed_groups(backend="gloo")
    parallel = PinnedWanUlyssesBackend(LayoutHandle(rank, ranks, cfg=cfg), groups).attach(worker)
    return worker, state, parallel, WanBlockExecutor(worker, parallel=parallel, interleave=interleave)


def run_quanta(executor, continuation, quanta, until):
    position = 0
    while continuation.base.step_index < until:
        executor.advance(continuation, blocks=quanta[position % len(quanta)])
        position += 1


def save_payload(torch, executor, continuation, path):
    from ditserve.models.wan.block import export_block_state
    payload, unused = export_block_state(executor, continuation)
    torch.save(payload, path)


def load_payload(torch, executor, path):
    from ditserve.models.wan.block import import_block_state
    base, continuation, unused = import_block_state(executor, torch.load(path, map_location="cpu", weights_only=True))
    return base, continuation


def scenario_quanta(torch, rank, world, directory, quanta, expand=False, interleave=False):
    """All ranks in one group; the quanta pattern varies, the result must not."""
    ranks = tuple(range(world))
    worker, state, parallel, executor = sp_executor(torch, rank, ranks, expand=expand, interleave=interleave)
    continuation = executor.attach(state)
    executor.advance(continuation, blocks=1)
    tokens = continuation.packet["hidden"].shape[1]  # this rank's shard of the 8 patch tokens
    run_quanta(executor, continuation, quanta, until=4)
    return dict(latents=state.latents.clone(), tokens=tokens)


def scenario_cfg(torch, rank, world, directory, quanta):
    """CFG parallelism: each half of the group runs one branch; the result must match SP1."""
    ranks = tuple(range(world))
    worker, state, parallel, executor = sp_executor(torch, rank, ranks, cfg=2)
    continuation = executor.attach(state)
    executor.advance(continuation, blocks=1)
    tokens = continuation.packet["hidden"].shape[1]  # this rank's shard within its half
    run_quanta(executor, continuation, quanta, until=4)
    return dict(latents=state.latents.clone(), tokens=tokens, role=parallel.handle.current.cfg_role(rank),
                steps=state.step_index)


def scenario_grow(torch, rank, world, directory):
    """SP2 on (0,1) for one step, then SP4 on (0,1,2,3); ranks 2 and 3 import."""
    path = os.path.join(directory, "boundary.pt")
    if rank < 2:
        worker, state, parallel, executor = sp_executor(torch, rank, (0, 1))
        continuation = executor.attach(state)
        run_quanta(executor, continuation, [STEP_BLOCKS], until=3)
        if rank == 0:
            save_payload(torch, executor, continuation, path)
    else:
        worker, state, parallel, executor = sp_executor(torch, rank, None, empty=True)
    torch.distributed.barrier()
    if rank >= 2:
        state, continuation = load_payload(torch, executor, path)
    parallel.handle.adopt((0, 1, 2, 3), epoch=1)
    run_quanta(executor, continuation, [4, 3, 5], until=4)
    return dict(latents=state.latents.clone(), imported=rank >= 2)


def scenario_move(torch, rank, world, directory):
    """SP2 moves from (0,1) to (2,3) at a step boundary; (0,1) also keeps its copy going."""
    path = os.path.join(directory, "boundary.pt")
    if rank < 2:
        worker, state, parallel, executor = sp_executor(torch, rank, (0, 1))
        continuation = executor.attach(state)
        run_quanta(executor, continuation, [STEP_BLOCKS], until=3)
        if rank == 0:
            save_payload(torch, executor, continuation, path)
    else:
        worker, state, parallel, executor = sp_executor(torch, rank, None, empty=True)
    torch.distributed.barrier()
    if rank >= 2:
        state, continuation = load_payload(torch, executor, path)
        parallel.handle.adopt((2, 3), epoch=1)
    run_quanta(executor, continuation, [2, 5], until=4)
    return dict(latents=state.latents.clone(), imported=rank >= 2)


def scenario_shrink(torch, rank, world, directory):
    """SP4 for one step, then every rank continues alone at SP1 from its replicated copy."""
    worker, state, parallel, executor = sp_executor(torch, rank, (0, 1, 2, 3))
    continuation = executor.attach(state)
    run_quanta(executor, continuation, [STEP_BLOCKS], until=3)
    if rank == 0:
        save_payload(torch, executor, continuation, os.path.join(directory, "boundary.pt"))
    parallel.handle.adopt((rank,), epoch=1)
    run_quanta(executor, continuation, [STEP_BLOCKS], until=4)
    return dict(latents=state.latents.clone())


def scenario_guards(torch, rank, world, directory):
    """A layout change or an export inside an SP step is refused before any collective."""
    from ditserve.models.wan.block import export_block_state
    from ditserve.models.wan.parallel import ParallelContractError
    worker, state, parallel, executor = sp_executor(torch, rank, (0, 1))
    continuation = executor.attach(state)
    executor.advance(continuation, blocks=2)
    errors = {}
    try:
        parallel.handle.adopt((rank,), epoch=1)
    except ParallelContractError as error:
        errors["adopt"] = str(error)
    try:
        export_block_state(executor, continuation)
    except ValueError as error:
        errors["export"] = str(error)
    executor.advance(continuation, blocks=STEP_BLOCKS)
    return dict(errors=errors, step=state.step_index, active=parallel.handle.active is not None)


SCENARIOS = dict(quanta_whole=lambda t, r, w, d: scenario_quanta(t, r, w, d, [STEP_BLOCKS]),
                 quanta_ragged=lambda t, r, w, d: scenario_quanta(t, r, w, d, [1, 4, 2, 5]),
                 expand_whole=lambda t, r, w, d: scenario_quanta(t, r, w, d, [STEP_BLOCKS], expand=True),
                 expand_ragged=lambda t, r, w, d: scenario_quanta(t, r, w, d, [1, 4, 2, 5], expand=True),
                 interleaved=lambda t, r, w, d: scenario_quanta(t, r, w, d, [2, 4], interleave=True),
                 cfg_whole=lambda t, r, w, d: scenario_cfg(t, r, w, d, [STEP_BLOCKS]),
                 cfg_ragged=lambda t, r, w, d: scenario_cfg(t, r, w, d, [1, 2]),
                 grow=scenario_grow, move=scenario_move, shrink=scenario_shrink, guards=scenario_guards)


def _rank(rank, world, port, directory, name):
    import torch
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.save(SCENARIOS[name](torch, rank, world, directory), os.path.join(directory, "rank%d.pt" % rank))
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(PINNED, "pinned CPU dependencies required")
class SpBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch
        worker, state, executor = plain_executor(torch)
        run_quanta(executor, executor.attach(state), [STEP_BLOCKS], until=4)
        cls.reference = state.latents.clone()
        worker, state, executor = plain_executor(torch, expand=True)
        run_quanta(executor, executor.attach(state), [STEP_BLOCKS], until=4)
        cls.reference_expand = state.latents.clone()

    def spawn(self, name, world):
        import torch.multiprocessing as mp
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_rank, args=(world, port, directory, name), nprocs=world, join=True)
            results = [self.torch.load(os.path.join(directory, "rank%d.pt" % rank), weights_only=True)
                       for rank in range(world)]
            boundary = os.path.join(directory, "boundary.pt")
            payload = self.torch.load(boundary, weights_only=True) if os.path.exists(boundary) else None
        return results, payload

    def assert_replicated(self, results):
        for result in results[1:]:
            self.assertTrue(self.torch.equal(result["latents"], results[0]["latents"]))

    def assert_sp1(self, latents, reference=None):
        # CPU kernels reproduce the unsharded result bit for bit; GPU deviation is measured, not assumed.
        reference = self.reference if reference is None else reference
        error = (latents.double() - reference.double()).abs().max().item()
        self.assertTrue(self.torch.equal(latents, reference), "SP deviates from SP1 by %g" % error)

    def test_block_quanta_are_exact_within_each_degree_and_match_sp1(self):
        for world in (2, 4):
            with self.subTest(degree=world):
                whole, unused = self.spawn("quanta_whole", world)
                ragged, unused = self.spawn("quanta_ragged", world)
                self.assertEqual({result["tokens"] for result in whole + ragged}, {8 // world})
                self.assert_replicated(whole)
                self.assert_replicated(ragged)
                self.assertTrue(self.torch.equal(whole[0]["latents"], ragged[0]["latents"]))
                self.assert_sp1(whole[0]["latents"])

    def test_per_token_timesteps_split_with_the_tokens_and_match_sp1(self):
        for world in (2, 4):
            with self.subTest(degree=world):
                whole, unused = self.spawn("expand_whole", world)
                ragged, unused = self.spawn("expand_ragged", world)
                self.assertEqual({result["tokens"] for result in whole + ragged}, {8 // world})
                self.assert_replicated(whole)
                self.assert_replicated(ragged)
                self.assertTrue(self.torch.equal(whole[0]["latents"], ragged[0]["latents"]))
                self.assert_sp1(whole[0]["latents"], self.reference_expand)

    def test_interleaved_branches_inside_sp_groups_match_sp1(self):
        for world in (2, 4):
            with self.subTest(degree=world):
                results, unused = self.spawn("interleaved", world)
                self.assertEqual({result["tokens"] for result in results}, {8 // world})
                self.assert_replicated(results)
                self.assert_sp1(results[0]["latents"])

    def test_cfg_parallel_halves_match_sp1(self):
        for world in (2, 4):
            with self.subTest(world=world):
                whole, unused = self.spawn("cfg_whole", world)
                ragged, unused = self.spawn("cfg_ragged", world)
                half = world // 2
                self.assertEqual({result["tokens"] for result in whole + ragged}, {8 // half})
                self.assertEqual([result["role"] for result in whole], [0] * half + [1] * half)
                self.assertEqual({result["steps"] for result in whole + ragged}, {4})
                self.assert_replicated(whole)
                self.assert_replicated(ragged)
                self.assertTrue(self.torch.equal(whole[0]["latents"], ragged[0]["latents"]))
                self.assert_sp1(whole[0]["latents"])

    def test_growing_a_group_gives_imported_ranks_the_resident_state(self):
        results, payload = self.spawn("grow", 4)
        self.assertEqual([result["imported"] for result in results], [False, False, True, True])
        self.assert_replicated(results)
        self.assert_sp1(results[0]["latents"])
        self.assertEqual(payload["manifest"]["layout"]["committed_step"], 3)

    def test_moving_a_group_matches_continuing_on_the_original_ranks(self):
        results, unused = self.spawn("move", 4)
        self.assert_replicated(results)
        self.assert_sp1(results[2]["latents"])

    def test_shrinking_to_sp1_continues_from_the_replicated_boundary(self):
        results, payload = self.spawn("shrink", 4)
        self.assert_replicated(results)
        self.assert_sp1(results[0]["latents"])
        # A plain single-GPU executor resumed from the SP4 boundary agrees with the SP1 bypass.
        worker, unused, executor = plain_executor(self.torch, empty=True)
        from ditserve.models.wan.block import import_block_state
        state, continuation, unused = import_block_state(executor, payload)
        run_quanta(executor, continuation, [STEP_BLOCKS], until=4)
        self.assertTrue(self.torch.equal(state.latents, results[0]["latents"]))

    def test_layout_and_export_wait_for_the_step_boundary(self):
        results, unused = self.spawn("guards", 2)
        for result in results:
            self.assertIn("healthy step boundary", result["errors"]["adopt"])
            self.assertIn("solver-step boundaries", result["errors"]["export"])
            self.assertEqual(result["step"], 3)
            self.assertFalse(result["active"])


class LayoutTests(unittest.TestCase):
    def test_eight_rank_layouts_and_controller_epochs(self):
        from ditserve.models.wan.parallel import LayoutHandle, ParallelContractError
        handle = LayoutHandle(7, (4, 5, 6, 7))
        self.assertEqual(handle.current.degree, 4)
        self.assertEqual(handle.adopt((6, 7), epoch=9).epoch, 9)
        # Controller epochs come from placements, not a local counter.
        self.assertEqual(handle.adopt(tuple(range(8)), epoch=3).degree, 8)
        with self.assertRaisesRegex(ParallelContractError, "not in the new group"):
            handle.adopt((0, 1), epoch=10)
        with self.assertRaisesRegex(ParallelContractError, "unsupported fixed rank group"):
            handle.adopt((5, 6), epoch=11)
        handle.begin_step("r", 0)
        with self.assertRaisesRegex(ParallelContractError, "step boundary"):
            handle.adopt((7,), epoch=12)
        with self.assertRaisesRegex(ParallelContractError, r"rank must be in \[0, 8\)"):
            LayoutHandle(8)
        from ditserve.models.wan.parallel import FixedLayout
        quad = FixedLayout((0, 1, 2, 3), cfg=2)
        self.assertEqual((quad.sp_degree, quad.cfg_role(2), quad.sp_ranks(2), quad.sp_ranks(0)), (2, 1, (2, 3), (0, 1)))
        self.assertEqual((FixedLayout((0, 1)).cfg_role(1), FixedLayout((0, 1)).sp_ranks(1)), (None, (0, 1)))
        with self.assertRaisesRegex(ParallelContractError, "at least two ranks"):
            FixedLayout((0,), cfg=2)
        with self.assertRaisesRegex(ParallelContractError, "cfg must be"):
            FixedLayout((0, 1), cfg=3)
        self.assertEqual(LayoutHandle(5).adopt((0, 1, 2, 3, 4, 5, 6, 7), epoch=20, cfg=2).sp_degree, 4)

    def test_backend_adopts_controller_groups_only_at_step_boundaries(self):
        from ditserve.models.wan.backend import WanBackend
        from ditserve.models.wan.parallel import LayoutHandle
        backend = WanBackend.__new__(WanBackend)
        backend.parallel = SimpleNamespace(handle=LayoutHandle(1, (0, 1)))
        state = SimpleNamespace(step_index=3)
        backend.continuations = {"r": SimpleNamespace(committed_step=3, packet=None, conditional=None,
                                                      branch="cond", next_block=0)}
        backend._adopt(dict(group=[0, 1, 2, 3], epoch=4), state, "r")
        backend._adopt(dict(group=[0, 1, 2, 3], epoch=4), state, "r")  # the same placement is a no-op
        current = backend.parallel.handle.current
        self.assertEqual((current.ranks, current.epoch), ((0, 1, 2, 3), 4))
        backend.continuations["r"] = SimpleNamespace(committed_step=3, packet={}, conditional=None,
                                                     branch="cond", next_block=2)
        with self.assertRaisesRegex(ValueError, "solver-step boundary"):
            backend._adopt(dict(group=[1], epoch=5), state, "r")
        backend.parallel = None
        backend._adopt(dict(group=[0], epoch=1), state, "r")  # SP1 needs no process group
        with self.assertRaisesRegex(ValueError, "distributed init"):
            backend._adopt(dict(group=[0, 1], epoch=1), state, "r")


if __name__ == "__main__":
    unittest.main()
