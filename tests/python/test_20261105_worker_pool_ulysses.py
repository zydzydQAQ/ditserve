"""Cross-worker /dev/shm prompt-embedding races and the interleaved Wan Ulysses dispatch.

CPU contracts only. The Ulysses tests run the interleaved block on two threads that stand
in for two SP ranks, exchanging through an equal-split loopback all-to-all; they prove
routing and arithmetic, not NCCL/GFC stream overlap on a GPU.
"""

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import types
from types import SimpleNamespace
import unittest
from unittest import mock

from ditserve.models.encoder_pool import ProactiveEncoderPool

ROOT = Path(__file__).resolve().parents[2]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DIFFUSERS = HAS_TORCH and importlib.util.find_spec("diffusers") is not None


def _embeds(torch, value, *, shape=(1, 4, 8)):
    return {
        "prompt_embeds": torch.full(shape, float(value), dtype=torch.bfloat16),
        "negative_prompt_embeds": torch.full(shape, -float(value), dtype=torch.bfloat16),
        "pooled_prompt_embeds": torch.full((1, 8), float(value), dtype=torch.bfloat16),
        "negative_pooled_prompt_embeds": None,
    }


@unittest.skipUnless(HAS_TORCH, "CPU torch required")
class SharedShmPoolRaceTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cfg = {"shm_dir": self.tmp.name, "write_shm": True, "pin_memory": False, "max_entries": 1}
        self.worker_a = ProactiveEncoderPool(dict(cfg))
        self.worker_b = ProactiveEncoderPool(dict(cfg))
        self.key = ProactiveEncoderPool.prompt_key("t2i", "sd35", "a lighthouse at dusk", "blurry", 256)
        self.other = ProactiveEncoderPool.prompt_key("t2i", "sd35", "a red kettle", "blurry", 256)

    def slab(self, key):
        return Path(self.tmp.name) / f"{key[:24]}.pt"

    def evict_on_a(self):
        """Worker A stages another prompt; its one-entry LRU evicts (and unlinks) the shared key."""
        self.worker_a.stage_embeddings(self.other, _embeds(self.torch, 9), write_shm=True)

    def test_has_key_then_eviction_still_races_but_ensure_resident_does_not(self):
        self.worker_a.stage_embeddings(self.key, _embeds(self.torch, 3), write_shm=True)
        # has_key is advisory: the owner can unlink the slab between the check and the fetch.
        self.assertTrue(self.worker_b.has_key(self.key))
        self.evict_on_a()
        self.assertFalse(self.slab(self.key).exists())
        with self.assertRaises(KeyError):
            self.worker_b.fetch_to_device(self.key, "cpu")

        # ensure_resident reads the slab into B's own LRU at check time.
        self.worker_a.stage_embeddings(self.key, _embeds(self.torch, 3), write_shm=True)
        self.assertTrue(self.worker_b.ensure_resident(self.key))
        self.assertIn(self.key, self.worker_b._entries)
        self.evict_on_a()
        self.assertFalse(self.slab(self.key).exists())
        fetched = self.worker_b.fetch_to_device(self.key, "cpu")["tensors"]
        self.assertTrue(self.torch.equal(fetched["prompt_embeds"], _embeds(self.torch, 3)["prompt_embeds"]))
        self.assertIsNone(fetched["negative_pooled_prompt_embeds"])
        self.assertEqual(self.worker_b.stats()["shm_adoptions"], 1)

    def test_missing_or_torn_slab_is_a_miss_not_an_error(self):
        self.assertFalse(self.worker_b.ensure_resident(self.key))
        self.slab(self.key).write_bytes(b"\x80\x04 torn slab")
        self.assertTrue(self.worker_b.has_key(self.key))
        self.assertFalse(self.worker_b.ensure_resident(self.key))
        with self.assertRaises(KeyError):
            self.worker_b.fetch_to_device(self.key, "cpu")
        self.assertEqual(self.worker_b.stats()["entries"], 0)

    def test_worker_never_unlinks_a_slab_it_did_not_write(self):
        self.worker_a.stage_embeddings(self.key, _embeds(self.torch, 3), write_shm=True)
        self.assertTrue(self.worker_b.ensure_resident(self.key))
        # B evicts its adopted copy: A's slab stays for every other worker.
        self.worker_b.stage_embeddings(self.other, _embeds(self.torch, 4), write_shm=False)
        self.assertNotIn(self.key, self.worker_b._entries)
        self.assertTrue(self.slab(self.key).is_file())
        self.worker_b.clear()
        self.assertTrue(self.slab(self.key).is_file())

        # B re-stages the same key, replacing the file; A's eviction must keep B's slab.
        self.worker_b.stage_embeddings(self.key, _embeds(self.torch, 5), write_shm=True)
        self.evict_on_a()
        self.assertTrue(self.slab(self.key).is_file())
        third = ProactiveEncoderPool({"shm_dir": self.tmp.name, "pin_memory": False})
        self.assertTrue(third.ensure_resident(self.key))
        self.assertTrue(self.torch.equal(third.fetch_to_device(self.key, "cpu")["tensors"]["prompt_embeds"],
                                         _embeds(self.torch, 5)["prompt_embeds"]))
        # The writer still removes its own slab when it evicts it.
        self.worker_b.stage_embeddings(self.other, _embeds(self.torch, 6), write_shm=False)
        self.assertFalse(self.slab(self.key).exists())

    def sd35(self, pool, encode_calls):
        from ditserve.models.sd35 import SD35Backend

        torch = self.torch
        backend = object.__new__(SD35Backend)
        backend.config = {"model_path": "sd35", "dtype": "bfloat16", "options": {}}
        backend.torch, backend.device = torch, "cpu"
        backend.encoder_pool = pool
        backend.encoders_offloaded = True

        def restore(*_args, **_kwargs):
            raise AssertionError("restore_encoders() must not run on a /dev/shm hit")

        def encode_prompt(**kwargs):
            encode_calls.append(kwargs["prompt"])
            e = _embeds(torch, 7)
            return e["prompt_embeds"], e["negative_prompt_embeds"], e["pooled_prompt_embeds"], None

        backend.restore_encoders = restore
        backend.pipe = SimpleNamespace(module=SimpleNamespace(encode_prompt=encode_prompt))
        return backend

    def test_sd35_shm_hit_survives_owner_eviction_before_fetch(self):
        encode_calls = []
        backend = self.sd35(self.worker_b, encode_calls)
        key = backend._prompt_key("a lighthouse at dusk", "blurry", 256, True)
        self.worker_a.stage_embeddings(key, _embeds(self.torch, 3), write_shm=True)

        original = self.worker_b.ensure_resident

        def check_then_owner_evicts(k):
            hit = original(k)
            self.worker_a.stage_embeddings(self.other, _embeds(self.torch, 9), write_shm=True)
            return hit

        self.worker_b.ensure_resident = check_then_owner_evicts
        arguments = dict(prompt="a lighthouse at dusk", negative_prompt="blurry",
                         max_sequence_length=256, guidance_scale=4.5)
        meta = backend._encode(arguments)
        self.assertFalse(self.slab(key).exists())
        self.assertTrue(meta["encoder_pool_hit"])
        self.assertEqual(encode_calls, [])
        self.assertTrue(self.torch.equal(arguments["prompt_embeds"], _embeds(self.torch, 3)["prompt_embeds"]))

        stage_info = backend._ensure_staged("a lighthouse at dusk", "blurry", 256, True)
        self.assertEqual((stage_info["cached"], stage_info["staged"]), (True, False))

    def test_sd35_fetch_miss_restages_instead_of_failing(self):
        encode_calls = []
        pool = ProactiveEncoderPool({"shm_dir": self.tmp.name, "pin_memory": False})
        backend = self.sd35(pool, encode_calls)
        backend.encoders_offloaded = False
        backend.offload_encoders = lambda *_a, **_k: None
        fetch, misses = pool.fetch_to_device, []

        def evicted_once(key, device, stream=None):
            if not misses:  # e.g. the async encode thread's staging evicted this key
                misses.append(key)
                pool._entries.pop(key)
            return fetch(key, device, stream=stream)

        pool.fetch_to_device = evicted_once
        stage_info, fetched = backend._fetch_staged("a red kettle", "", 256, True)
        self.assertEqual(stage_info["restaged_after_miss"], 1)
        self.assertEqual(encode_calls, ["a red kettle", "a red kettle"])
        self.assertTrue(self.torch.equal(fetched["tensors"]["prompt_embeds"], _embeds(self.torch, 7)["prompt_embeds"]))

    def test_wan_trimmed_encode_shm_hit_survives_owner_eviction(self):
        torch = self.torch
        runtime = SimpleNamespace(model_path=Path("snapshot"), model_revision="rev1", text_encoder_dtype="bfloat16",
                                  pipe=SimpleNamespace(text_encoder=SimpleNamespace(dtype=torch.bfloat16)),
                                  _owner=object())
        spec = dict(request_id="a", prompt="a kettle", height=32, width=32, frames=5, steps=2, seed=7)
        writer = ProactiveEncoderPool({"shm_dir": self.tmp.name, "write_shm": True, "pin_memory": False,
                                       "max_entries": 1})
        writer._encode_single_wan_text_trimmed = lambda _w, _t, _n: (torch.full((1, 4, 2), 2., dtype=torch.bfloat16), 4)
        first = writer.encode_wan_prompt_fast(runtime, dict(spec, guidance_scale=1.0), pin_memory=False)
        reader = ProactiveEncoderPool({"shm_dir": self.tmp.name, "pin_memory": False})
        reader._encode_single_wan_text_trimmed = mock.Mock(side_effect=AssertionError("must hit the slab"))
        original = reader.ensure_resident

        def check_then_owner_evicts(key):
            hit = original(key)
            writer.stage_embeddings(self.other, {"embed": torch.zeros(1)}, write_shm=True)
            return hit

        reader.ensure_resident = check_then_owner_evicts
        second = reader.encode_wan_prompt_fast(runtime, dict(spec, guidance_scale=1.0), pin_memory=False)
        self.assertTrue(second.negative_prompt_cache["pos_cache_hit"])
        self.assertTrue(torch.equal(second.prompt_embeds, first.prompt_embeds))


# ---------------------------------------------------------------------------------------------
# Interleaved Wan Ulysses block: collective dispatch and Q/K norm + RoPE selection.

def _vendored_processor_class(namespace):
    """The real vendored ``xFuserWanAttnProcessor``, compiled from source against stand-ins.

    Importing ``xfuser`` needs optional packages this CPU environment lacks; executing the
    class's own source keeps ``__call__`` (and its FlyDSL selection) exactly as vendored.
    """
    path = ROOT / "workers/xfuser/model_executor/models/transformers/transformer_wan.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "xFuserWanAttnProcessor")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["xFuserWanAttnProcessor"], ast.get_source_segment(path.read_text(), node)


class _LoopbackAllToAll:
    """Equal-split all-to-all across threads standing in for Ulysses ranks."""

    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=60)
        self.slots = [None] * world
        self.local = threading.local()

    def exchange(self, source, output=None):
        import torch
        rank = self.local.rank
        self.slots[rank] = source.chunk(self.world)
        self.barrier.wait()
        received = torch.cat([self.slots[src][rank] for src in range(self.world)])
        self.barrier.wait()
        if output is None:
            return received
        output.copy_(received)
        return output


@unittest.skipUnless(HAS_DIFFUSERS, "CPU torch and diffusers required")
class InterleavedUlyssesDispatchTests(unittest.TestCase):
    WORLD, HEADS, HEAD_DIM, TOKENS, TEXT = 2, 4, 12, 8, 5

    def setUp(self):
        import torch
        import torch.nn.functional as F
        from diffusers.models.transformers.transformer_wan import WanAttnProcessor, WanTransformerBlock

        self.torch = torch
        self.calls = {"fused": 0, "usp": 0}
        lock = threading.Lock()

        def fused_qk_norm_rope(query, key, norm_q, norm_k, cos, sin, heads):
            with lock:
                self.calls["fused"] += 1
            # Arithmetic of the reference, so outputs stay comparable; only the routing differs.
            q, k, _ = self.reference(SimpleNamespace(norm_q=norm_q, norm_k=norm_k, heads=heads),
                                     query, key, key, (cos, sin))
            return q, k

        def usp(query, key, value, **_kwargs):
            with lock:
                self.calls["usp"] += 1
            return F.scaled_dot_product_attention(query, key, value)

        register = SimpleNamespace(register=lambda _base: (lambda cls: cls))
        self.namespace = dict(
            torch=torch, Optional=__import__("typing").Optional, Union=__import__("typing").Union,
            Dict=dict, Any=__import__("typing").Any, Tuple=tuple, WanAttnProcessor=WanAttnProcessor,
            xFuserAttentionProcessorRegister=register, USP=usp, attention=usp,
            get_runtime_state=lambda: SimpleNamespace(get_cross_attention_backend=lambda: None),
            _HAS_FLYDSL=False, fused_qk_norm_rope=fused_qk_norm_rope)
        self.processor_cls, self.call_source = _vendored_processor_class(self.namespace)
        self.reference = self.processor_cls._qk_norm_rope_reference.__get__(object.__new__(self.processor_cls))

        torch.manual_seed(1105)
        dim = self.HEADS * self.HEAD_DIM
        self.native = WanTransformerBlock(dim, 32, self.HEADS, "rms_norm_across_heads", True, 1e-6).eval()
        with torch.no_grad():
            self.native.scale_shift_table.normal_(0, .1)
        self.block = copy.deepcopy(self.native)
        self.block.attn1.processor = self.processor_cls(attention_kwargs={"valid_kv_len": None})
        self.inputs = dict(
            cond=torch.randn(1, self.TOKENS, dim), uncond=torch.randn(1, self.TOKENS, dim),
            encoder_c=torch.randn(1, self.TEXT, dim), encoder_u=torch.randn(1, self.TEXT, dim),
            projected=torch.randn(1, 6, dim) * .1,
            rotary=(torch.randn(1, self.TOKENS, 1, self.HEAD_DIM), torch.randn(1, self.TOKENS, 1, self.HEAD_DIM)))

    def fake_xfuser(self):
        import torch.nn.functional as F

        def attention_function(backend=None):
            return lambda q, k, v, **_kw: (F.scaled_dot_product_attention(q, k, v), None)

        def module(name, **attrs):
            value = types.ModuleType(name)
            value.__path__ = []
            value.__dict__.update(attrs)
            return value

        state = SimpleNamespace(fp8_comms=None, attention_backend="sdpa_math")
        return mock.patch.dict(sys.modules, {
            "xfuser": module("xfuser"),
            "xfuser.core": module("xfuser.core"),
            "xfuser.core.distributed": module(
                "xfuser.core.distributed", get_ulysses_parallel_world_size=lambda: self.WORLD,
                get_ring_parallel_world_size=lambda: 1, get_runtime_state=lambda: state),
            "xfuser.model_executor": module("xfuser.model_executor"),
            "xfuser.model_executor.layers": module("xfuser.model_executor.layers"),
            "xfuser.model_executor.layers.usp": module(
                "xfuser.model_executor.layers.usp", _get_attention_function=attention_function,
                _trim_trailing_kv_padding=lambda k, v, kw, _b=None: (k, v, kw),
                _maybe_wait=lambda tensor: tensor),
        })

    def executor(self):
        from ditserve.models.wan.block import WanBlockExecutor

        torch = self.torch

        class TorchWithoutCuda:  # the interleaved path only names the main stream here
            cuda = SimpleNamespace(current_stream=lambda *_a, **_k: "main-stream")

            def __getattr__(self, name):
                return getattr(torch, name)

        executor = WanBlockExecutor.__new__(WanBlockExecutor)
        executor.worker = SimpleNamespace(torch=TorchWithoutCuda())
        executor.model = SimpleNamespace(blocks=[self.block], _sp_pad={"local_rows": 0},
                                         _self_attn_kwargs={"valid_kv_len": None}, attention_kwargs=None)
        executor.xdit = object()
        executor._record = lambda _packet, _stream: None
        return executor

    def run_ranks(self, loopback):
        """Each thread is one Ulysses rank running the interleaved block on its sequence shard."""
        torch, x = self.torch, self.inputs
        shard = self.TOKENS // self.WORLD
        outputs, errors = [None] * self.WORLD, []

        def rank_main(rank):
            loopback.local.rank = rank
            rows = slice(rank * shard, (rank + 1) * shard)
            rotary = tuple(r[:, rows] for r in x["rotary"])
            packet = dict(hidden=x["cond"][:, rows], encoder=x["encoder_c"], projected=x["projected"],
                          rotary=rotary, local_rows=0, valid_kv_len=None, thw=None)
            other = dict(packet, hidden=x["uncond"][:, rows], encoder=x["encoder_u"])
            try:
                with torch.no_grad():
                    outputs[rank] = self.executor()._forward_interleaved_block(packet, other, 0, side="side-stream")
            except BaseException as error:  # surfaced below; the barrier unblocks the peer
                errors.append(error)
                loopback.barrier.abort()

        threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(self.WORLD)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        if errors:
            raise errors[0]
        cond = torch.cat([out[0] for out in outputs], dim=1)
        uncond = torch.cat([out[1] for out in outputs], dim=1)
        return cond, uncond

    def native_outputs(self):
        x = self.inputs
        with self.torch.no_grad():
            return (self.native(x["cond"], x["encoder_c"], x["projected"], x["rotary"]),
                    self.native(x["uncond"], x["encoder_u"], x["projected"], x["rotary"]))

    def test_vendored_processor_selection_is_the_one_the_helper_mirrors(self):
        self.assertIn("if _HAS_FLYDSL and rotary_emb is not None:", self.call_source)
        self.assertIn("fused_qk_norm_rope(", self.call_source)
        self.assertIs(self.processor_cls.__call__.__globals__, self.namespace)

    def test_gfc_override_carries_every_interleaved_all_to_all(self):
        import torch.distributed._functional_collectives as ft_c
        from ditserve.models.xdit import collectives
        from ditserve.models.xdit.process_group import PROCESS_GROUP

        loopback, group = _LoopbackAllToAll(self.WORLD), object()
        runtime = SimpleNamespace(all2all=lambda source, output, _target, slice_bytes: loopback.exchange(source, output))
        saved = (PROCESS_GROUP.ULYSSES_PG, PROCESS_GROUP.ALL_TO_ALL)
        self.addCleanup(lambda: (setattr(PROCESS_GROUP, "ULYSSES_PG", saved[0]),
                                 setattr(PROCESS_GROUP, "ALL_TO_ALL", saved[1])))
        self.addCleanup(collectives.shutdown)
        # What ProcessGroupPool.activate installs for collective_backend='gfc'.
        collectives._BACKEND = "gfc"
        collectives.bind(group, (0, 1))
        PROCESS_GROUP.ULYSSES_PG, PROCESS_GROUP.ALL_TO_ALL = group, collectives.all_to_all_flat
        nccl = mock.Mock(side_effect=AssertionError("NCCL all_to_all_single bypassed the GFC override"))
        with self.fake_xfuser(), mock.patch.object(ft_c, "all_to_all_single", nccl), \
                mock.patch.object(collectives, "descriptor", lambda ranks: ranks), \
                mock.patch.object(collectives, "_submit", lambda _tensors, fn: fn(runtime)):
            cond, uncond = self.run_ranks(loopback)
        nccl.assert_not_called()
        # cond in, uncond in, cond out, uncond out on each of the two ranks.
        self.assertEqual(collectives.info()["calls"], {"all2all": 4 * self.WORLD})
        expected_cond, expected_uncond = self.native_outputs()
        self.torch.testing.assert_close(cond, expected_cond, rtol=1e-5, atol=1e-5)
        self.torch.testing.assert_close(uncond, expected_uncond, rtol=1e-5, atol=1e-5)

    def test_without_override_nccl_runs_on_the_ulysses_group(self):
        import torch.distributed._functional_collectives as ft_c
        from ditserve.models.xdit import collectives
        from ditserve.models.xdit.process_group import PROCESS_GROUP

        loopback, group, groups = _LoopbackAllToAll(self.WORLD), object(), []
        saved = (PROCESS_GROUP.ULYSSES_PG, PROCESS_GROUP.ALL_TO_ALL)
        self.addCleanup(lambda: (setattr(PROCESS_GROUP, "ULYSSES_PG", saved[0]),
                                 setattr(PROCESS_GROUP, "ALL_TO_ALL", saved[1])))

        def nccl(flat, output_split_sizes=None, input_split_sizes=None, group=None):
            groups.append(group)
            return loopback.exchange(flat)

        # NCCL backend: no override, or GFC's adapter declining (it returns None under NCCL).
        for override in (None, collectives.all_to_all_flat):
            with self.subTest(override=override):
                groups.clear()
                PROCESS_GROUP.ULYSSES_PG, PROCESS_GROUP.ALL_TO_ALL = group, override
                with self.fake_xfuser(), mock.patch.object(ft_c, "all_to_all_single", nccl):
                    cond, _ = self.run_ranks(loopback)
                self.assertEqual(groups, [group] * (4 * self.WORLD))
                self.torch.testing.assert_close(cond, self.native_outputs()[0], rtol=1e-5, atol=1e-5)

    def test_interleaved_and_processor_paths_select_the_same_qk_norm_rope(self):
        import torch.distributed._functional_collectives as ft_c
        from ditserve.models.xdit.process_group import PROCESS_GROUP

        torch, x = self.torch, self.inputs
        saved = (PROCESS_GROUP.ULYSSES_PG, PROCESS_GROUP.ALL_TO_ALL)
        self.addCleanup(lambda: (setattr(PROCESS_GROUP, "ULYSSES_PG", saved[0]),
                                 setattr(PROCESS_GROUP, "ALL_TO_ALL", saved[1])))
        reference = mock.Mock(wraps=self.processor_cls._qk_norm_rope_reference)
        for flydsl in (False, True):
            with self.subTest(flydsl=flydsl), mock.patch.object(
                    self.processor_cls, "_qk_norm_rope_reference",
                    lambda proc, *args: reference(proc, *args)):
                self.namespace["_HAS_FLYDSL"] = flydsl
                # Non-interleaved: the vendored processor's own __call__ on one branch.
                self.calls.update(fused=0, usp=0)
                reference.reset_mock()
                with torch.no_grad():
                    self.block.attn1(self.block.norm1(x["cond"]), None, None, x["rotary"])
                processor_path = (self.calls["fused"], reference.call_count)
                self.assertEqual(self.calls["usp"], 1)
                self.assertEqual(processor_path, (1, 0) if flydsl else (0, 1))

                # Interleaved Ulysses: two branches on each of two ranks. Both collective routes
                # reach the loopback, so this checks only the Q/K norm + RoPE selection.
                self.calls.update(fused=0, usp=0)
                reference.reset_mock()
                loopback = _LoopbackAllToAll(self.WORLD)
                PROCESS_GROUP.ULYSSES_PG = object()
                PROCESS_GROUP.ALL_TO_ALL = lambda flat, _group: loopback.exchange(flat)
                nccl = lambda flat, **_kw: loopback.exchange(flat)
                with self.fake_xfuser(), mock.patch.object(ft_c, "all_to_all_single", nccl):
                    cond, uncond = self.run_ranks(loopback)
                per_branch = (self.calls["fused"] // (2 * self.WORLD), reference.call_count // (2 * self.WORLD))
                self.assertEqual((self.calls["fused"] + reference.call_count), 2 * self.WORLD)
                self.assertEqual(per_branch, processor_path)
                self.assertEqual(self.calls["usp"], 0)
                expected_cond, expected_uncond = self.native_outputs()
                torch.testing.assert_close(cond, expected_cond, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(uncond, expected_uncond, rtol=1e-5, atol=1e-5)

    def test_selection_without_rotary_uses_the_reference(self):
        torch = self.torch
        self.namespace["_HAS_FLYDSL"] = True
        from ditserve.models.wan.block import _processor_qk_norm_rope

        attn = self.block.attn1
        q, k, v = (torch.randn(1, 3, self.HEADS * self.HEAD_DIM) for _ in range(3))
        out = _processor_qk_norm_rope(attn.processor, attn, q, k, v, None)
        self.assertEqual(self.calls["fused"], 0)
        self.assertEqual(tuple(out[0].shape), (1, 3, self.HEADS, self.HEAD_DIM))


if __name__ == "__main__":
    unittest.main()
