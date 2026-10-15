import datetime
import copy
import importlib.util
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
import unittest

from ditserve.research.parallel import (
    FIXED_GROUPS, FixedProcessGroups, KnownTensorSpec, LayoutHandle, ParallelContractError,
    RequestGroupGate, bind_request_group_similarity, build_fixed_groups,
    merge_known_state, reshard_known_state, split_known_state,
    CacheOffGroupMigration, pack_cache_off_request, unpack_cache_off_request,
    CACHE_BUFFER_NAMES, LATENT_CACHE_BUFFER_NAMES, ENCODER_CACHE_BUFFER_NAMES, cache_manager_manifest,
)


def _pattern2_fixture():
    """Actual pinned cache wrapper and CFG bookkeeping, inexpensive CPU blocks."""
    import torch
    from cache_dit import DBCacheConfig
    from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
    from cache_dit.caching.forward_pattern import ForwardPattern
    class Block(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states):
            return hidden_states + 1
    manager = CachedContextManager()
    context = manager.new_context(name="blocks", cache_config=DBCacheConfig(
        Fn_compute_blocks=1, Bn_compute_blocks=0, max_warmup_steps=8, warmup_interval=1,
        residual_diff_threshold=.08, max_continuous_cached_steps=1, enable_separate_cfg=True,
        cfg_compute_first=False, cfg_diff_compute_separate=True, num_inference_steps=12))
    blocks = CachedBlocks_Pattern_0_1_2(torch.nn.ModuleList([Block(), Block(), Block()]),
        forward_pattern=ForwardPattern.Pattern_2, cache_prefix="blocks", cache_context=context,
        context_manager=manager)
    return manager, context, blocks


def _run_pattern2_cfg(manager, context, blocks, handle, groups):
    import torch
    gate = RequestGroupGate(handle, groups)
    binding = bind_request_group_similarity(manager, gate)
    try:
        for index in range(12):
            handle.begin_step("cached-A" if handle.rank < 2 else "cached-B", index)
            with torch.inference_mode(), manager.enter_context(context):
                gate.pre_step(cache_manager_manifest(manager))
                for _ in ("cond", "uncond"):
                    # Text length 3 is intentionally not divisible by SP2/4.
                    blocks(torch.full((1, 8, 4), index * .001), torch.ones(1, 3, 4))
                manifest = cache_manager_manifest(manager)
                if "invalid" in manifest:
                    raise AssertionError(manifest)
                gate.consensus(dict(kind="step_complete", manifest=manifest))
            handle.end_step()
        # Preserve the pinned dependency's actual counter semantics, including
        # its gap handling; do not synthesize an alternating reuse schedule.
        if context.cached_steps != [8, 10, 11] or context.cfg_cached_steps != [8, 10, 11]:
            raise AssertionError((context.cached_steps, context.cfg_cached_steps))
        if set(context.buffers) != CACHE_BUFFER_NAMES:
            raise AssertionError(sorted(context.buffers))
        return gate
    finally:
        binding.close()


def _gloo_pattern2_worker(rank, rendezvous, result_queue):
    try:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(1)
        dist = torch.distributed
        dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=4,
                                timeout=datetime.timedelta(seconds=30))
        groups = build_fixed_groups(timeout_seconds=8, backend="gloo")
        handle = LayoutHandle(rank, (0, 1) if rank < 2 else (2, 3))
        manager, context, blocks = _pattern2_fixture()
        gate = _run_pattern2_cfg(manager, context, blocks, handle, groups)
        # A missing *encoder* cache on one member must be observed before any
        # local can_cache early return can send the members down different paths.
        if rank in (1, 3):
            del context.buffers["blocks_Bn_residual_encoder_buffer_cfg"]
        handle.begin_step("presence-A" if rank < 2 else "presence-B", 12)
        with manager.enter_context(context):
            try:
                gate.pre_step(cache_manager_manifest(manager))
            except ParallelContractError:
                if not handle.failed:
                    raise AssertionError("cache mismatch did not poison the group")
            else:
                raise AssertionError("unequal real encoder-cache presence was accepted")
        result_queue.put(dict(rank=rank, ok=True, cfg_reuse=context.cfg_cached_steps))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback
        result_queue.put(dict(rank=rank, ok=False, error=repr(error), traceback=traceback.format_exc()))


@unittest.skipUnless(importlib.util.find_spec("cache_dit"), "cache-dit unavailable")
class Pattern2CacheSchemaTests(unittest.TestCase):
    def test_real_pattern2_cfg_compute_reuse_and_six_buffer_partition(self):
        import torch
        manager, context, blocks = _pattern2_fixture()
        _run_pattern2_cfg(manager, context, blocks, LayoutHandle(0),
                          FixedProcessGroups(0, {}, backend="gloo", timeout_seconds=3))
        self.assertEqual(len(CACHE_BUFFER_NAMES), 6)
        self.assertEqual(context.executed_steps, 12)
        self.assertEqual(context.transformer_executed_steps, 24)
        specs = tuple(KnownTensorSpec(name, tuple(tensor.shape), str(tensor.dtype))
                      for name, tensor in sorted(context.buffers.items()))
        shards = split_known_state(context.buffers, specs, (0, 1, 2, 3))
        for name in ENCODER_CACHE_BUFFER_NAMES:
            self.assertEqual(shards[0][name].shape[1], 3)
            for rank in range(4):
                self.assertTrue(torch.equal(shards[rank][name], context.buffers[name]))
        for name in LATENT_CACHE_BUFFER_NAMES:
            self.assertEqual(shards[0][name].shape[1], 2)
        restored = merge_known_state(reshard_known_state(shards, specs, (0, 1, 2, 3), (0, 1)), specs, (0, 1))
        for name in CACHE_BUFFER_NAMES:
            self.assertTrue(torch.equal(restored[name], context.buffers[name]))
        shards[1]["blocks_Bn_residual_encoder_buffer"][0, 0, 0] = 1
        with self.assertRaisesRegex(ParallelContractError, "replicated"):
            merge_known_state(shards, specs, (0, 1, 2, 3))

    def test_unknown_encoder_names_and_l1_recipe_remain_rejected(self):
        import torch
        manager, context, _ = _pattern2_fixture()
        with manager.enter_context(context):
            context.buffers["blocks_Fn_unknown_encoder_buffer"] = torch.zeros(1, 3, 4)
            self.assertIn("invalid", cache_manager_manifest(manager))
            context.buffers.clear()
            context.extra_cache_config.l1_hidden_states_diff_threshold = .08
            self.assertIn("invalid", cache_manager_manifest(manager))

    def test_two_real_cfg_request_groups_and_encoder_presence_failure(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            results = context.Queue()
            children = [context.Process(target=_gloo_pattern2_worker, args=(rank, os.path.join(directory, "store"), results))
                        for rank in range(4)]
            try:
                for child in children:
                    child.start()
                deadline = time.monotonic() + 60
                received = [results.get(timeout=max(.1, deadline - time.monotonic())) for _ in children]
                self.assertTrue(all(row["ok"] for row in received), received)
                for child in children:
                    child.join(timeout=5)
                    self.assertEqual(child.exitcode, 0)
            finally:
                for child in children:
                    if child.is_alive():
                        child.terminate()
                    child.join(timeout=2)


def _cache_off_fixture(steps=8):
    import torch
    from diffusers import UniPCMultistepScheduler
    from ditserve.gpu.worker import WanStepWorker, RequestState, normalize_spec
    worker = WanStepWorker.__new__(WanStepWorker)
    worker.torch, worker.device, worker.attention = torch, "cpu", "flash"
    worker._owner, worker._states = object(), {}
    worker._lock, worker.manager = threading.RLock(), SimpleNamespace()
    worker._sync = lambda: None
    worker._memory = lambda: dict(allocated_bytes=0, reserved_bytes=0)
    worker._scheduler_type = UniPCMultistepScheduler
    spec = normalize_spec(dict(request_id="migration-A", prompt="a boat", seed=101,
                               height=32, width=32, frames=5, steps=steps, guidance_scale=5.,
                               cache=False, threshold=0., max_continuous=1))
    scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps, device="cpu")
    state = RequestState(spec["request_id"], spec, scheduler=scheduler,
                         latents=torch.arange(512, dtype=torch.float32).reshape(1, 16, 2, 4, 4) / 512,
                         prompt_embeds=torch.ones(1, 512, 4096, dtype=torch.float16),
                         negative_prompt_embeds=torch.zeros(1, 512, 4096, dtype=torch.float16),
                         generator=torch.Generator().manual_seed(spec["seed"]), owner=worker._owner)
    state.immutable_spec = {k: v for k, v in spec.items() if k not in ("cache", "threshold", "max_continuous")}
    worker._states[state.request_id] = state
    return worker, state


def _advance_cache_off(state, count=1):
    import torch
    for _ in range(count):
        index = state.step_index
        prediction = torch.full_like(state.latents, .05 + .001 * index, dtype=torch.float16)
        state.latents = state.scheduler.step(prediction, state.scheduler.timesteps[index], state.latents,
                                            return_dict=False)[0]
        state.step_index += 1


def _gloo_migration_worker(rank, rendezvous, result_queue):
    try:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        import torch
        torch.set_num_threads(1)
        dist = torch.distributed
        dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=4,
                                timeout=datetime.timedelta(seconds=30))
        groups = build_fixed_groups(timeout_seconds=10, backend="gloo")
        worker, state = _cache_off_fixture()
        reference = copy.deepcopy(state)
        _advance_cache_off(reference, 8)
        _advance_cache_off(state, 4)
        handle = LayoutHandle(rank, (0, 1, 2, 3))
        mover = CacheOffGroupMigration(worker, handle, groups)
        state, shrink = mover.migrate(state, (0, 1, 2, 3), (0, 1), request_id="migration-A", step_index=4, expected_epoch=0)
        if rank >= 2 and (state is not None or worker._states):
            raise AssertionError("departing rank retained request ownership")
        if state is not None:
            _advance_cache_off(state, 2)
        dist.barrier()
        state, grow = mover.migrate(state, (0, 1), (0, 1, 2, 3), request_id="migration-A", step_index=6, expected_epoch=1)
        if state.owner is not worker._owner or worker._states[state.request_id] is not state:
            raise AssertionError("destination owner was not rebound")
        for value in state.scheduler.timestep_list:
            if value.untyped_storage().data_ptr() != state.scheduler.timesteps.untyped_storage().data_ptr():
                raise AssertionError("scheduler timestep view alias was lost")
        _advance_cache_off(state, 2)
        if not torch.equal(state.latents, reference.latents) or not torch.equal(state.generator.get_state(), reference.generator.get_state()):
            raise AssertionError("transport changed the resumed UniPC trajectory")
        # Divergent source RNG must fail uniformly before release/commit.
        worker.release(state)
        worker, state = _cache_off_fixture()
        _advance_cache_off(state, 4)
        handle = LayoutHandle(rank, (0, 1, 2, 3))
        if rank == 1:
            state.generator.manual_seed(999)
        try:
            CacheOffGroupMigration(worker, handle, groups).migrate(state, (0, 1, 2, 3), (0, 1),
                request_id="migration-A", step_index=4, expected_epoch=0)
        except ParallelContractError:
            if not handle.failed or worker._states.get(state.request_id) is not state:
                raise AssertionError("failed transaction released its source before agreement")
        else:
            raise AssertionError("divergent source state was accepted")
        result_queue.put(dict(rank=rank, ok=True, shrink_bytes=shrink["logical_payload_bytes"], grow_bytes=grow["received_tensor_bytes"]))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback
        result_queue.put(dict(rank=rank, ok=False, error=repr(error), traceback=traceback.format_exc()))


class CacheOffWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("diffusers"):
            raise unittest.SkipTest("pinned CPU torch and diffusers required")

    def test_every_solver_phase_preserves_exact_suffix_and_timestep_aliases(self):
        import torch
        torch.set_num_threads(1)
        for boundary in (0, 1, 2, 7):
            worker, state = _cache_off_fixture()
            _advance_cache_off(state, boundary)
            baseline = copy.deepcopy(state)
            manifest, tensors = pack_cache_off_request(worker, state)
            target, discarded = _cache_off_fixture()
            target.release(discarded)
            restored = unpack_cache_off_request(target, manifest, tensors)
            self.assertIs(restored.owner, target._owner)
            self.assertIs(restored.scheduler.disable_corrector, restored.scheduler.config.disable_corrector)
            self.assertNotIn(restored.request_id, target._states)
            for value in restored.scheduler.timestep_list:
                if value is not None:
                    self.assertEqual(value.untyped_storage().data_ptr(), restored.scheduler.timesteps.untyped_storage().data_ptr())
            for _ in range(boundary, 8):
                _advance_cache_off(baseline)
                _advance_cache_off(restored)
                self.assertTrue(torch.equal(baseline.latents, restored.latents))
            self.assertTrue(torch.equal(baseline.generator.get_state(), restored.generator.get_state()))

    def test_unknown_state_cache_views_and_corrupt_payload_fail_closed(self):
        worker, state = _cache_off_fixture()
        state.unexpected = None
        with self.assertRaises(TypeError):
            pack_cache_off_request(worker, state)
        del state.unexpected
        state.spec["cache"] = state.cache_enabled = True
        with self.assertRaises(ParallelContractError):
            pack_cache_off_request(worker, state)
        state.spec["cache"] = state.cache_enabled = False
        _advance_cache_off(state, 2)
        saved = state.scheduler.last_sample
        state.scheduler.last_sample = state.latents[:]
        with self.assertRaises(ValueError):
            pack_cache_off_request(worker, state)
        state.scheduler.last_sample = saved
        manifest, tensors = pack_cache_off_request(worker, state)
        tensors["latents"].reshape(-1)[0] += 1
        with self.assertRaisesRegex(ParallelContractError, "checksum"):
            unpack_cache_off_request(worker, manifest, tensors)

    def test_four_rank_live_transport_roundtrip_and_failure(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            results = context.Queue()
            children = [context.Process(target=_gloo_migration_worker, args=(rank, os.path.join(directory, "store"), results))
                        for rank in range(4)]
            try:
                for child in children:
                    child.start()
                deadline = time.monotonic() + 90
                received = [results.get(timeout=max(.1, deadline - time.monotonic())) for _ in children]
                self.assertTrue(all(row["ok"] for row in received), received)
                for child in children:
                    child.join(timeout=5)
                    self.assertEqual(child.exitcode, 0)
            finally:
                for child in children:
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=2)


def _gloo_contract_worker(rank, rendezvous, result_queue):
    """A single world exercises independent requests, uniform failure, timeout."""
    try:
        # The contract test is local-only, even on hosts with no resolvable
        # hostname. Sandboxed hosts still need permission for loopback sockets.
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        import torch
        torch.set_num_threads(1)
        dist = torch.distributed
        dist.init_process_group("gloo", init_method="file://" + rendezvous,
                                rank=rank, world_size=4,
                                timeout=datetime.timedelta(seconds=20))
        groups = build_fixed_groups(timeout_seconds=3, backend="gloo")
        pair = (0, 1) if rank < 2 else (2, 3)
        handle = LayoutHandle(rank, pair)
        gate = RequestGroupGate(handle, groups)
        handle.begin_step("A" if rank < 2 else "B", 8)
        gate.pre_step(dict(buffers={"Fn": [1, 12, 4]}, counters={"steps": 8}))
        numerator, denominator = gate.mean_pair(torch.tensor(float(rank + 1)), torch.tensor(2.0))
        if numerator.item() != (1.5 if rank < 2 else 3.5) or denominator.item() != 2.0:
            raise AssertionError("request groups mixed scalar reductions")
        handle.end_step()

        mismatch_count = 0
        for field in ("presence", "shape", "request", "cfg", "step", "counter"):
            handle = LayoutHandle(rank, pair)
            handle.begin_step("A" if rank < 2 else "B", 8)
            gate = RequestGroupGate(handle, groups)
            metadata = dict(presence=True, shape=[1, 12, 4], request="same", cfg="cond", step=8, counter=8)
            if rank in (1, 3):
                metadata[field] = "different"
            try:
                gate.pre_step(metadata)
            except ParallelContractError:
                if not handle.failed:
                    raise AssertionError("mismatch did not poison layout")
                mismatch_count += 1
            else:
                raise AssertionError("rank disagreement was accepted")
        # Invalid metadata on only one rank still enters the collective.
        handle = LayoutHandle(rank, pair)
        handle.begin_step("A" if rank < 2 else "B", 9)
        gate = RequestGroupGate(handle, groups)
        try:
            gate.consensus({"bad": object() if rank in (1, 3) else "valid"})
        except ParallelContractError:
            mismatch_count += 1
        else:
            raise AssertionError("invalid peer metadata was accepted")

        dist.barrier()
        # A peer never enters the request collective. The finite group timeout
        # must surface an error instead of hanging the test suite indefinitely.
        timeout_observed = rank != 1
        if rank == 1:
            handle = LayoutHandle(rank, pair)
            handle.begin_step("timeout-request", 0)
            started = time.monotonic()
            try:
                RequestGroupGate(handle, groups).consensus({"test": "missing_peer"})
            except ParallelContractError:
                timeout_observed = handle.failed and time.monotonic() - started < 10
            if not timeout_observed:
                raise AssertionError("missing peer did not fail within its deadline")
        else:
            # Keeping peers alive distinguishes a bounded timeout from an
            # immediate connection-close error. This test-only wait is bounded.
            time.sleep(4.5)
        result_queue.put(dict(rank=rank, ok=True, mismatches=mismatch_count,
                              timeout=timeout_observed))
        dist.destroy_process_group()
    except BaseException as error:
        result_queue.put(dict(rank=rank, ok=False, error=repr(error)))


class LayoutContractTests(unittest.TestCase):
    def test_step_boundary_epoch_and_membership(self):
        handle = LayoutHandle(0)
        self.assertEqual(handle.current.degree, 1)
        handle.begin_step("a", 0)
        with self.assertRaises(ParallelContractError):
            handle.commit((0, 1), expected_epoch=0)
        handle.end_step()
        self.assertEqual(handle.commit((0, 1), expected_epoch=0).epoch, 1)
        with self.assertRaises(ParallelContractError):
            handle.commit((0, 1, 2, 3), expected_epoch=0)
        with self.assertRaises(ParallelContractError):
            handle.commit((2, 3), expected_epoch=1)
        with self.assertRaises(ParallelContractError):
            handle.commit((0, 2), expected_epoch=1)
        handle.begin_step("a", 1, expected_epoch=1)
        handle.fail()
        with self.assertRaises(ParallelContractError):
            handle.end_step()

    def test_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import ditserve.research.parallel; import ditserve.research.dual_state; "
            "import ditserve.research.state_digest; import ditserve.research.adapter; "
            "import ditserve.research.cfg_adapter; "
            "assert 'torch' not in sys.modules"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_schema_is_rejected(self):
        with self.assertRaises(ParallelContractError):
            KnownTensorSpec("guess_from_shape", (1, 16, 4), "torch.float32")
        with self.assertRaises(ParallelContractError):
            KnownTensorSpec("blocks_Fn_residual_buffer", (16, 4), "torch.float32")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class ExactCodecTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.state = {
            "latents": torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 2, 4).clone(),
            "blocks_Fn_residual_buffer": torch.arange(96, dtype=torch.float32).reshape(1, 24, 4).clone(),
            "blocks_Bn_residual_buffer_cfg": torch.arange(96, dtype=torch.float16).reshape(1, 24, 4).clone(),
            "rng_state": torch.arange(16, dtype=torch.uint8),
        }
        self.specs = tuple(KnownTensorSpec(name, tuple(value.shape), str(value.dtype))
                           for name, value in self.state.items())

    def test_exact_roundtrip_including_nan_and_signed_zero_bits(self):
        self.state["latents"].reshape(-1)[0] = float("nan")
        self.state["latents"].reshape(-1)[1] = -0.0
        shards = split_known_state(self.state, self.specs, (0,))
        source = (0,)
        for target in ((0, 1), (0, 1, 2, 3), (0, 1), (0,)):
            shards = reshard_known_state(shards, self.specs, source, target)
            source = target
        restored = merge_known_state(shards, self.specs, (0,))
        for name, value in self.state.items():
            self.assertTrue(self.torch.equal(value.reshape(-1).view(self.torch.uint8),
                                             restored[name].reshape(-1).view(self.torch.uint8)))

    def test_missing_extra_wrong_shape_and_replica_mismatch_rejected(self):
        for invalid in ({}, dict(self.state, unknown=self.torch.zeros(1))):
            with self.assertRaises(ParallelContractError):
                split_known_state(invalid, self.specs, (0, 1))
        shards = split_known_state(self.state, self.specs, (0, 1))
        shards[1]["latents"][0, 0, 0, 0, 0] = 99
        with self.assertRaises(ParallelContractError):
            merge_known_state(shards, self.specs, (0, 1))
        with self.assertRaises(ParallelContractError):
            merge_known_state({0: shards[0]}, self.specs, (0, 1))

    def test_aliased_storage_and_uneven_partitions_rejected(self):
        invalid = dict(self.state)
        invalid["latents"] = self.torch.empty(32).narrow(0, 0, 16).reshape(1, 1, 2, 2, 4)
        with self.assertRaises(ParallelContractError):
            split_known_state(invalid, self.specs, (0, 1))
        name = "blocks_Fn_residual_buffer"
        with self.assertRaises(ParallelContractError):
            split_known_state({name: self.torch.zeros(1, 7, 4)},
                              (KnownTensorSpec(name, (1, 7, 4), "torch.float32"),), (0, 1))

    def test_singleton_gate_never_uses_world(self):
        from unittest.mock import patch
        handle = LayoutHandle(2)
        handle.begin_step("a", 0)
        groups = FixedProcessGroups(2, {}, backend="gloo", timeout_seconds=3)
        with patch.object(self.torch.distributed, "all_reduce", side_effect=AssertionError("WORLD")), \
                patch.object(self.torch.distributed, "all_gather_object", side_effect=AssertionError("WORLD")):
            result = RequestGroupGate(handle, groups).mean_pair(self.torch.tensor(2.), self.torch.tensor(4.))
        self.assertEqual(tuple(x.item() for x in result), (2., 4.))


@unittest.skipUnless(importlib.util.find_spec("cache_dit"), "cache-dit unavailable")
class SimilarityBindingTests(unittest.TestCase):
    def test_manager_binding_is_local_restorable_and_matches_singleton(self):
        import torch
        from unittest.mock import patch
        from cache_dit import DBCacheConfig
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
        manager = CachedContextManager()
        context = manager.new_context(name="blocks", cache_config=DBCacheConfig())
        other = CachedContextManager()
        original = type(manager).similarity
        handle = LayoutHandle(0)
        handle.begin_step("request", 9)
        gate = RequestGroupGate(handle, FixedProcessGroups(0, {}, backend="gloo", timeout_seconds=3))
        binding = bind_request_group_similarity(manager, gate)
        self.assertIs(type(manager).similarity, original)
        self.assertNotIn("similarity", vars(other))
        with manager.enter_context(context), patch.object(torch.distributed, "all_reduce",
                                                          side_effect=AssertionError("WORLD")):
            self.assertTrue(manager.similarity(torch.ones(1, 8, 4), torch.full((1, 8, 4), .99),
                                               threshold=.08))
        binding.close()
        self.assertNotIn("similarity", vars(manager))
        self.assertNotIn("can_cache", vars(manager))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class GlooGroupTests(unittest.TestCase):
    def test_two_requests_uniform_mismatch_and_bounded_timeout(self):
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = os.path.join(directory, "rendezvous")
            processes = [context.Process(target=_gloo_contract_worker,
                                         args=(rank, rendezvous, result_queue)) for rank in range(4)]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 50
                results = []
                for _ in processes:
                    results.append(result_queue.get(timeout=max(.1, deadline - time.monotonic())))
                self.assertEqual({result["rank"] for result in results}, set(range(4)))
                self.assertTrue(all(result["ok"] for result in results), results)
                self.assertTrue(all(result["mismatches"] == 7 and result["timeout"] for result in results))
                for process in processes:
                    process.join(timeout=5)
                    self.assertFalse(process.is_alive(), "Gloo child did not exit")
            except queue.Empty:
                self.fail("Gloo contract test exceeded its 50-second deadline")
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=3)
                result_queue.close()


if __name__ == "__main__":
    unittest.main()
