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
    from test_model_state import advance as advance_state, empty_target, fixture
except ImportError:
    from tests.python.test_model_state import advance as advance_state, empty_target, fixture

PINNED = all(importlib.util.find_spec(name) for name in ("torch", "diffusers", "cache_dit", "xfuser", "yunchang"))
STEP_BLOCKS = 6  # three blocks per CFG branch


class WanPipelineStub:
    def __init__(self, transformer):
        self.transformer = transformer
        self.vae_scale_factor = 8
        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4


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
    """One rank's worker with the xDiT USP attachment over the Gloo world."""
    from ditserve.models.wan.block import WanBlockExecutor
    from ditserve.models.wan.parallel import LayoutHandle, build_fixed_groups
    from ditserve.models.xdit.wan import XDitWanEngine, XDitWanParallelBackend
    if empty:
        worker, state = empty_target(), None
    else:
        worker, state = fixture(cached=False, boundary=2)
    worker.pipe, worker.attention = WanPipelineStub(tiny_model(torch)), "math"
    worker.engine_name = "xdit"
    worker.expand_timesteps = expand
    groups = build_fixed_groups(backend="gloo")
    worker.xdit = XDitWanEngine(worker, groups=groups)
    parallel = XDitWanParallelBackend(LayoutHandle(rank, ranks, cfg=cfg), groups).attach(worker)
    return worker, state, parallel, WanBlockExecutor(worker, parallel=parallel, interleave=interleave)


def second_state(worker, torch, request_id="request-b", seed=202):
    """Another request on the fixture worker at the same step boundary, with its own inputs."""
    from diffusers import UniPCMultistepScheduler
    from ditserve.models.wan.runtime import RequestState, normalize_spec
    spec = normalize_spec(dict(request_id=request_id, prompt="a teapot", seed=seed, height=32, width=32, frames=5,
                               steps=32, guidance_scale=5., cache=False, threshold=0., max_continuous=1))
    scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(32, device="cpu")
    state = RequestState(spec["request_id"], spec, scheduler=scheduler, cache_enabled=False,
                         latents=(511 - torch.arange(512, dtype=torch.float32)).reshape(1, 16, 2, 4, 4) / 512,
                         prompt_embeds=torch.full((1, 512, 4096), .5, dtype=torch.float16),
                         negative_prompt_embeds=torch.full((1, 512, 4096), -.25, dtype=torch.float16),
                         generator=torch.Generator().manual_seed(spec["seed"]), owner=worker._owner, contexts={})
    state.immutable_spec = {key: value for key, value in spec.items() if key not in ("cache", "threshold", "max_continuous")}
    worker._states[state.request_id] = state
    advance_state(state, 2)
    return state


def run_pair_quanta(executor, first, second, quanta, until):
    position = 0
    while first.base.step_index < until:
        executor.advance_pair(first, second, blocks=quanta[position % len(quanta)])
        position += 1


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


def scenario_pair(torch, rank, world, directory, cfg, quanta):
    """Two requests advance in lockstep on one layout; each must equal its own sequential run."""
    ranks = tuple(range(world))
    worker, state, parallel, executor = sp_executor(torch, rank, ranks, cfg=cfg)
    other = second_state(worker, torch)
    first, second = executor.attach(state), executor.attach(other)
    run_pair_quanta(executor, first, second, quanta, until=4)
    paired = dict(a=state.latents.clone(), b=other.latents.clone())
    # The sequential reference for request-b on the same layout.
    alone = second_state(worker, torch, request_id="request-b2")
    run_quanta(executor, executor.attach(alone), [STEP_BLOCKS], until=4)
    return dict(latents=paired["a"], latents_b=paired["b"], reference_b=alone.latents.clone(),
                tokens=first.packet["hidden"].shape[1] if first.packet is not None else None)


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


def non_divisible_state(state, torch):
    state.latents = (torch.arange(384, dtype=torch.float32) / 384).reshape(1, 16, 1, 6, 4)
    for metadata in (state.spec, state.immutable_spec):
        metadata.update(height=48, width=32, frames=1)
    # A different latent shape cannot reuse the fixture's old UniPC history.
    state.scheduler = type(state.scheduler)(num_train_timesteps=1000)
    state.scheduler.set_timesteps(state.spec["steps"], device="cpu")
    state.step_index = 0
    return state


def scenario_padding(torch, rank, world, directory):
    worker, state, parallel, executor = sp_executor(torch, rank, tuple(range(world)))
    continuation = executor.attach(non_divisible_state(state, torch))
    run_quanta(executor, continuation, [1, 2, 3], until=3)
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


def scenario_sync_failure(torch, rank, world, directory):
    worker, state, parallel, executor = sp_executor(torch, rank, (0, 1))
    continuation = executor.attach(state)
    original_end_step = parallel.end_step

    def end_step(request_id=None):
        if rank == 1:
            def fail():
                raise RuntimeError("sync failure")
            worker._sync = fail
        return original_end_step(request_id)

    parallel.end_step = end_step
    error = None
    try:
        executor.advance(continuation, blocks=STEP_BLOCKS)
    except BaseException as exception:
        error = type(exception).__name__ + ": " + str(exception)
    return dict(error=error, failed=parallel.handle.failed)


def scenario_binding(torch, rank, world, directory, cfg=1):
    from ditserve.models.wan.parallel import pack_cache_off_request, request_engine
    worker, state, parallel, executor = sp_executor(torch, rank, tuple(range(world)), cfg=cfg)
    worker.xdit.activate(state, parallel.handle.current)
    manifest, unused = pack_cache_off_request(worker, state)
    return dict(engine=request_engine(worker), config=dict(worker.pipe.transformer.config),
                state_sha256=manifest["state_sha256"])


def scenario_layout_mismatch(torch, rank, world, directory):
    from ditserve.models.wan.parallel import ParallelContractError
    ranks = tuple(range(world))
    worker, state, parallel, executor = sp_executor(torch, rank, ranks)
    if rank == 1:
        parallel.handle.adopt(ranks, epoch=0, cfg=2)
    continuation = executor.attach(state)
    calls = []
    original_forward = worker.xdit.forward_block

    def counted_forward(*args):
        calls.append(args[1])
        return original_forward(*args)

    worker.xdit.forward_block = counted_forward
    error = None
    try:
        executor.advance(continuation, blocks=STEP_BLOCKS)
    except ParallelContractError as exception:
        error = str(exception)
    return dict(error=error, failed=parallel.handle.failed, forwards=len(calls),
                packet=continuation.packet is not None)


def scenario_cache_migration(torch, rank, world, directory):
    worker, state, parallel, executor = sp_executor(torch, rank, tuple(range(world)))
    request_id = state.request_id
    continuation = executor.attach(state)
    run_quanta(executor, continuation, [STEP_BLOCKS], until=2)
    executor.close(continuation)
    state, unused = parallel.migrate_cache_off(
        state, (0, 1, 2, 3), (0, 1), request_id=request_id, step_index=2, expected_epoch=0)
    continuation = executor.attach(state) if state is not None else None
    if continuation is not None:
        executor.close(continuation)
    state, unused = parallel.migrate_cache_off(
        state, (0, 1), (0, 1, 2, 3), request_id=request_id, step_index=2, expected_epoch=1)
    continuation = executor.attach(state)
    run_quanta(executor, continuation, [1, 2, 3], until=4)
    return dict(latents=state.latents.clone(), epoch=parallel.handle.current.epoch)


SCENARIOS = dict(binding=scenario_binding, sync_failure=scenario_sync_failure,
                 binding_cfg=lambda t, r, w, d: scenario_binding(t, r, w, d, cfg=2),
                 quanta_whole=lambda t, r, w, d: scenario_quanta(t, r, w, d, [STEP_BLOCKS]),
                 quanta_ragged=lambda t, r, w, d: scenario_quanta(t, r, w, d, [1, 4, 2, 5]),
                 expand_whole=lambda t, r, w, d: scenario_quanta(t, r, w, d, [STEP_BLOCKS], expand=True),
                 expand_ragged=lambda t, r, w, d: scenario_quanta(t, r, w, d, [1, 4, 2, 5], expand=True),
                 interleaved=lambda t, r, w, d: scenario_quanta(t, r, w, d, [2, 4], interleave=True),
                 cfg_whole=lambda t, r, w, d: scenario_cfg(t, r, w, d, [STEP_BLOCKS]),
                 cfg_ragged=lambda t, r, w, d: scenario_cfg(t, r, w, d, [1, 2]),
                 pair_sp=lambda t, r, w, d: scenario_pair(t, r, w, d, 1, [2, 1, 3]),
                 pair_cfg=lambda t, r, w, d: scenario_pair(t, r, w, d, 2, [1, 2]),
                 grow=scenario_grow, move=scenario_move, shrink=scenario_shrink,
                 cache_migration=scenario_cache_migration, layout_mismatch=scenario_layout_mismatch,
                 padding=scenario_padding, guards=scenario_guards)


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

    def test_paired_requests_match_their_sequential_runs(self):
        # Request a's lockstep result equals SP1 (the CPU contract); request b's equals its own
        # sequential run on the same layout, under plain SP and under CFG parallelism.
        for name, world in (("pair_sp", 2), ("pair_sp", 4), ("pair_cfg", 2), ("pair_cfg", 4)):
            with self.subTest(scenario=name, world=world):
                results, unused = self.spawn(name, world)
                self.assert_replicated(results)
                self.assert_sp1(results[0]["latents"])
                self.assertTrue(self.torch.equal(results[0]["latents_b"], results[0]["reference_b"]))
                self.assertFalse(self.torch.equal(results[0]["latents_b"], results[0]["latents"]))

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
        worker, unused, unused_executor = plain_executor(self.torch, empty=True)
        from ditserve.models.wan.block import WanBlockExecutor, import_block_state
        from ditserve.models.xdit.wan import XDitWanEngine
        worker.pipe = WanPipelineStub(worker.pipe.transformer)
        engine = XDitWanEngine(worker)
        self.addCleanup(engine.close)
        worker.xdit, worker.engine_name = engine, "xdit"
        executor = WanBlockExecutor(worker)
        state, continuation, unused = import_block_state(executor, payload)
        run_quanta(executor, continuation, [STEP_BLOCKS], until=4)
        self.assertTrue(self.torch.equal(state.latents, results[0]["latents"]))

    def test_non_divisible_tokens_are_padded_without_changing_output(self):
        results, unused = self.spawn("padding", 4)
        self.assert_replicated(results)
        worker, state, executor = plain_executor(self.torch)
        run_quanta(executor, executor.attach(non_divisible_state(state, self.torch)), [6], until=3)
        self.assertTrue(self.torch.equal(results[0]["latents"], state.latents))

    def test_layout_and_export_wait_for_the_step_boundary(self):
        results, unused = self.spawn("guards", 2)
        for result in results:
            self.assertIn("healthy step boundary", result["errors"]["adopt"])
            self.assertIn("solver-step boundaries", result["errors"]["export"])
            self.assertEqual(result["step"], 3)
            self.assertFalse(result["active"])

    def test_engine_metadata_is_stable_across_processes(self):
        for name, cfg, sp in (("binding", 1, 4), ("binding_cfg", 2, 2)):
            results, unused = self.spawn(name, 4)
            self.assertEqual([row["engine"] for row in results], [results[0]["engine"]] * 4,
                             msg=repr(results))
            self.assertEqual([row["state_sha256"] for row in results], [results[0]["state_sha256"]] * 4)
            self.assertEqual(results[0]["engine"]["layout"],
                             {"sp_degree": sp, "cfg": cfg, "boundary": "solver_step"})

    def test_cache_off_group_migration_matches_uninterrupted_execution(self):
        results, unused = self.spawn("cache_migration", 4)
        self.assertEqual([result["epoch"] for result in results], [2] * 4)
        self.assert_replicated(results)
        self.assert_sp1(results[0]["latents"])

    def test_inconsistent_cfg_layout_aborts_before_model_execution(self):
        results, unused = self.spawn("layout_mismatch", 2)
        self.assertTrue(all(result["failed"] for result in results), msg=repr(results))
        self.assertEqual([result["forwards"] for result in results], [0, 0], msg=repr(results))
        self.assertTrue(all("metadata mismatch" in result["error"] for result in results), msg=repr(results))

    def test_synchronizing_failure_aborts_every_rank(self):
        results, unused = self.spawn("sync_failure", 2)
        self.assertTrue(all(result["failed"] for result in results))
        self.assertIn("metadata mismatch", results[0]["error"])
        self.assertIn("sync failure", results[1]["error"])


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

    @unittest.skipUnless(PINNED, "pinned CPU dependencies required")
    def test_nccl_is_rejected_before_activation_without_a_real_gpu(self):
        import torch
        from ditserve.models.wan.parallel import LayoutHandle, ParallelContractError
        from ditserve.models.xdit.wan import XDitWanEngine, XDitWanParallelBackend

        worker, unused, unused_executor = plain_executor(torch, empty=True)
        worker.pipe = WanPipelineStub(worker.pipe.transformer)
        worker.engine_name = "xdit"
        worker.xdit = XDitWanEngine(worker)
        self.addCleanup(worker.xdit.close)
        backend = XDitWanParallelBackend(LayoutHandle(0, (0, 1)), SimpleNamespace(rank=0, backend="nccl"))
        with self.assertRaisesRegex(ParallelContractError, "NCCL.*GPU"):
            backend.attach(worker)
        self.assertIsNone(backend.worker)


if __name__ == "__main__":
    unittest.main()
