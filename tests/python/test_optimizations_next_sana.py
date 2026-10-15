"""Standard-library CPU fakes: no model, torch, numpy or GPU is imported."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import next_sana
from ditserve.optimizations.next_sana import NextSanaAdapter
from ditserve.optimizations.sana import OptimizedSanaAdapter


class Tensor:
    def __init__(self, values, *, dtype="torch.bfloat16", device="cuda:0", stride=None):
        self.values = list(values)
        self.shape, self.dtype, self.device = (len(self.values),), dtype, device
        self.requires_grad, self.is_cuda, self.layout = False, device.startswith("cuda"), "torch.strided"
        self._stride = (1,) if stride is None else stride

    def stride(self):
        return self._stride

    def detach(self):
        return self

    def clone(self, **kwargs):
        return Tensor(self.values, dtype=self.dtype, device=self.device,
                      stride=(1,) if 0 in self._stride else self._stride)

    def copy_(self, other):
        self.values[:] = other.values
        return self


class Event:
    def __init__(self, **kwargs):
        self.records = 0

    def record(self):
        self.records += 1

    def elapsed_time(self, other):
        return .01


class Stream:
    def __init__(self, **kwargs):
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)


class Graph:
    def __init__(self):
        self.action, self.replays, self.resets = None, 0, 0

    def replay(self):
        self.replays += 1
        self.action()

    def reset(self):
        self.resets += 1
        self.action = None


class CUDA:
    Event, Stream = Event, Stream

    def __init__(self):
        self.graphs, self.capturing, self.syncs = [], None, 0
        self.current = Stream()
        self.fail_capture = False

    def synchronize(self, device):
        self.syncs += 1

    def device(self, device):
        return nullcontext()

    def current_stream(self, device):
        return self.current

    def stream(self, stream):
        return nullcontext()

    def CUDAGraph(self):
        value = Graph()
        self.graphs.append(value)
        return value

    @contextmanager
    def graph(self, graph, **kwargs):
        if self.fail_capture:
            raise RuntimeError("capture unavailable")
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None


class Module:
    def __init__(self, fn):
        self.fn = fn

    def forward(self, value):
        return Tensor([self.fn(item) for item in value.values], dtype=value.dtype, device=value.device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Transformer:
    def __init__(self, cuda):
        self.cuda, self.calls = cuda, 0
        self.training = False
        self.dtype = "torch.bfloat16"
        self.config = SimpleNamespace(in_channels=32, guidance_embeds_scale=.1)

    def forward(self, value, *, encoder_hidden_states, timestep, return_dict=False,
                attention_kwargs=None, guidance=None, encoder_attention_mask=None):
        self.calls += 1
        def compute():
            return Tensor([item + encoder_hidden_states.values[0] + timestep.values[0]
                           + (0 if guidance is None else guidance.values[0])
                           for item in value.values], dtype=value.dtype, device=value.device)
        result = compute()
        if self.cuda.capturing:
            self.cuda.capturing.action = lambda: result.copy_(compute())
        return (result,)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Torch:
    preserve_format = "preserve"

    def __init__(self):
        self.cuda, self.compile_options, self.stances = CUDA(), [], []
        self.fail_compile = self.silent_compile = self.recompile = False
        self.compiler = SimpleNamespace(set_stance=self.set_stance)

    def set_stance(self, stance):
        self.stances.append(stance)
        return nullcontext()

    def is_tensor(self, value):
        return isinstance(value, Tensor)

    def device(self, value):
        return value

    def compile(self, original, **options):
        self.compile_options.append(options)
        executable = None
        def invoke(*args, **kwargs):
            nonlocal executable
            if self.fail_compile:
                raise RuntimeError("compile failed")
            if self.silent_compile:
                return original(*args, **kwargs)
            if executable is None or self.recompile:
                executable = options["backend"](original, [])
            return executable(*args, **kwargs)
        return invoke


def adapter(*, extra=False, post_quant=False):
    value = NextSanaAdapter.__new__(NextSanaAdapter)
    value.torch, value.device = Torch(), "cuda:0"
    transformer = Transformer(value.torch.cuda)
    decoder = Module(lambda item: item * 2)
    vae = SimpleNamespace(decoder=decoder, dtype="torch.bfloat16", use_tiling=False, use_slicing=False)
    if post_quant:
        vae.post_quant_conv = Module(lambda item: item + 1)
    def decode(latent):
        if post_quant:
            latent = vae.post_quant_conv(latent)
        return vae.decoder(latent)
    vae.decode = decode
    value.pipe = SimpleNamespace(transformer=transformer, vae=vae,
        image_processor=SimpleNamespace(postprocess=lambda x: x.clone(), resize_and_crop_tensor=lambda x: x))
    value._eager_transformer, value._compiled_transformer = transformer, None
    value._busy, value._owner, value.backend = False, object(), "eager"
    value.compile_records, value.compile_setup_ms = [], 0.
    value._init_next(profile_extra=extra)
    return value


def parent_generate(instance, prompts, *, seeds, encoded=None, check_budget=lambda: None):
    """Exercise the new wrappers through the unchanged parent's public API."""
    import time
    instance._busy = True
    try:
        check_budget()
        instance._sync()
        origin = time.perf_counter()
        value = Tensor(seeds)
        for step in range(2):
            value = instance.pipe.transformer(value, encoder_hidden_states=Tensor([len(prompts[0])]),
                                              timestep=Tensor([step]), return_dict=False)[0]
            check_budget()
        value = instance.pipe.vae.decode(value)
        value = instance.pipe.image_processor.resize_and_crop_tensor(value)
        value = instance.pipe.image_processor.postprocess(value)
        instance._sync()
        return value, dict(origin_perf_counter_s=origin, stages=[dict(stage="text_encode", start_ms=0., end_ms=.01),
            dict(stage="vae_decode", start_ms=.02, end_ms=.03)], ms=1., backend=instance.backend)
    finally:
        instance._busy = False


class NextSanaTests(unittest.TestCase):
    def setUp(self):
        self.backend_options = []
        def inductor(original, inputs, **kwargs):
            self.backend_options.append(kwargs)
            def executable(*args, **kw):
                return original(*args, **kw)
            executable._torchdynamo_orig_callable = original
            return executable
        self.backend_patch = patch.object(next_sana, "_inductor_backend", return_value=inductor)
        self.parent_patch = patch.object(OptimizedSanaAdapter, "generate_batch", parent_generate)
        self.backend_patch.start()
        self.parent_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.addCleanup(self.parent_patch.stop)

    def test_vae_preserves_modules_decode_recipe_and_tiling(self):
        obj = adapter(post_quant=True)
        vae, decoder, post, decode = obj.pipe.vae, obj.pipe.vae.decoder, obj.pipe.vae.post_quant_conv, obj.pipe.vae.decode
        vae.use_tiling = True
        expected, _ = obj.generate("abc", seed=7)
        obj.set_candidate("vae_compile")
        actual, service = obj.generate("abc", seed=7)
        self.assertEqual(actual.values, expected.values)
        self.assertIs(obj.pipe.vae, vae)
        self.assertIs(vae.decoder, decoder)
        self.assertIs(vae.post_quant_conv, post)
        self.assertIs(vae.decode, decode)
        self.assertTrue(vae.use_tiling)
        self.assertEqual(service["candidate"], "vae_compile")
        self.assertFalse(service["candidate_summary"]["qualified"])
        self.assertEqual([row["name"] for row in obj.candidate_summary()["vae_regions"]], ["decoder", "post_quant_conv"])

    def test_autodc_missing_post_quant_is_not_replaced_or_invented(self):
        obj = adapter()
        obj.set_candidate("vae_compile")
        obj.generate("a", seed=1)
        self.assertFalse(hasattr(obj.pipe.vae, "post_quant_conv"))
        self.assertEqual(len(obj.candidate_summary()["vae_regions"]), 1)
        self.assertEqual(self.backend_options, [{"config_patches": {"triton.cudagraphs": False}}])
        self.assertTrue(obj.torch.compile_options[0]["fullgraph"])
        self.assertFalse(obj.torch.compile_options[0]["dynamic"])

    def test_legacy_eager_notification_cannot_disable_candidate(self):
        obj = adapter()
        original = obj.pipe.vae.decoder.forward
        obj.set_candidate("vae_compile")
        selected = obj.pipe.vae.decoder.forward
        obj.set_backend("eager")
        self.assertIs(obj.pipe.vae.decoder.forward, selected)
        self.assertEqual(obj.candidate, "vae_compile")
        with self.assertRaises(ValueError):
            obj.set_backend("compiled")
        obj.set_candidate("eager")
        self.assertEqual(obj.pipe.vae.decoder.forward, original)

    def test_counter_wrap_does_not_copy_dynamo_unwrapping_metadata(self):
        obj = adapter()
        obj.set_candidate("vae_compile")
        obj.generate("a", seed=1)
        obj.generate("a", seed=2)
        summary = obj.candidate_summary()["vae_regions"][0]
        self.assertEqual(summary["graph_invocations"], 2)
        self.assertEqual(summary["graph_compile_count"], 1)
        region = obj._vae_regions["decoder"]
        wrapped = region._backend(lambda x: x, [])
        self.assertFalse(hasattr(wrapped, "_torchdynamo_orig_callable"))

    def test_compile_failure_and_silent_fallback_restore_and_require_explicit_retry(self):
        for silent in (False, True):
            obj = adapter(extra=True)
            original = obj.pipe.vae.decoder.forward
            obj.set_candidate("vae_compile")
            obj.torch.silent_compile, obj.torch.fail_compile = silent, not silent
            with self.subTest(silent=silent), self.assertRaises(RuntimeError):
                obj.generate("a", seed=1)
            self.assertEqual(obj.pipe.vae.decoder.forward, original)
            self.assertEqual(obj.candidate_summary()["status"], "failed")
            with self.assertRaisesRegex(RuntimeError, "explicit set_candidate"):
                obj.generate("a", seed=1)
            obj.set_candidate("eager")
            obj.generate("a", seed=1)

    def test_steady_compile_recompile_is_not_allowed(self):
        obj = adapter()
        obj.set_candidate("vae_compile")
        obj.generate("a", seed=1)
        with obj.steady_context():
            obj.generate("b", seed=2)
        self.assertIn("fail_on_recompile", obj.torch.stances)
        obj.torch.recompile = True
        with obj.steady_context(), self.assertRaisesRegex(RuntimeError, "steady"):
            obj.generate("a", seed=1)
        self.assertEqual(obj._steady_depth, 0)

    def test_graph_matches_eager_and_preserves_transformer_attributes(self):
        obj = adapter()
        model, config = obj.pipe.transformer, obj.pipe.transformer.config
        reference, _ = obj.generate("abc", seed=7)
        obj.set_candidate("transformer_graph")
        actual, service = obj.generate("abc", seed=7)
        self.assertEqual(actual.values, reference.values)
        self.assertIs(obj.pipe.transformer, model)
        self.assertIs(obj.pipe.transformer.config, config)
        self.assertEqual(obj.pipe.transformer.dtype, "torch.bfloat16")
        graph = service["candidate_summary"]["transformer_graph"]
        self.assertEqual(graph["capture_count"], 1)
        self.assertEqual(graph["warmup_eager_calls"], 2)
        self.assertEqual(graph["graph_replays"], 2)
        self.assertTrue(graph["scheduler_and_rng_outside_graph"])

    def test_graph_replay_copies_new_inputs_and_never_aliases_prior_outputs(self):
        obj = adapter()
        obj.set_candidate("transformer_graph")
        def run(value, context):
            return obj.pipe.transformer(Tensor([value]), encoder_hidden_states=Tensor([context]),
                                        timestep=Tensor([2]), return_dict=False)[0]
        first = run(1, 3)
        second = run(9, 5)
        self.assertEqual(first.values, [6])
        self.assertEqual(second.values, [16])
        self.assertIsNot(first, second)
        self.assertIsNot(first.values, obj._transformer_graph.static_output[0].values)
        old, _ = obj.generate("a", seed=4)
        values = list(old.values)
        obj.generate("longer", seed=50)
        self.assertEqual(old.values, values)

    def test_graph_refuses_shape_dtype_device_stride_and_constant_changes(self):
        mutations = [Tensor([1, 2]), Tensor([1], dtype="torch.float32"),
                     Tensor([1], device="cpu"), Tensor([1], stride=(2,))]
        for value in mutations:
            obj = adapter()
            obj.set_candidate("transformer_graph")
            call = dict(encoder_hidden_states=Tensor([1]), timestep=Tensor([0]), return_dict=False)
            obj.pipe.transformer(Tensor([1]), **call)
            with self.subTest(value=value.__dict__), self.assertRaises(ValueError):
                obj.pipe.transformer(value, **call)
            self.assertEqual(obj._transformer_graph.capture_count, 1)
        with self.assertRaises(ValueError):
            obj.pipe.transformer(Tensor([1]), **dict(call, return_dict=True))

    def test_official_kwargs_with_expanded_timestep_guidance_and_int64_mask(self):
        obj = adapter()
        def arguments(step, guidance, mask):
            return dict(encoder_hidden_states=Tensor([3]),
                encoder_attention_mask=Tensor([mask] * 300, dtype="torch.int64"),
                guidance=Tensor([guidance], stride=(0,)),
                timestep=Tensor([step], dtype="torch.float32", stride=(0,)),
                return_dict=False, attention_kwargs=None)
        eager = obj.pipe.transformer(Tensor([2]), **arguments(1, 4, 1))[0]
        obj.set_candidate("transformer_graph")
        actual = obj.pipe.transformer(Tensor([2]), **arguments(1, 4, 1))[0]
        self.assertEqual(eager.values, actual.values)
        self.assertEqual(actual.values, [10])
        updated = obj.pipe.transformer(Tensor([2]), **arguments(2, 5, 0))[0]
        self.assertEqual(updated.values, [12])
        self.assertEqual(actual.values, [10])
        static = obj._transformer_graph.static_inputs[1]
        self.assertEqual(static["timestep"].stride(), (1,))
        self.assertEqual(static["guidance"].stride(), (1,))
        self.assertEqual(static["encoder_attention_mask"].values, [0] * 300)
        self.assertEqual(static["encoder_attention_mask"].dtype, "torch.int64")
        summary = obj._transformer_graph.summary()
        self.assertNotEqual(summary["source_input_signature"], summary["static_input_signature"])
        self.assertEqual(summary["capture_count"], 1)

    def test_graph_capture_failure_restores_forward_and_releases_storage(self):
        obj = adapter(extra=True)
        original, postprocess = obj.pipe.transformer.forward, obj.pipe.image_processor.postprocess
        obj.set_candidate("transformer_graph")
        obj.torch.cuda.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "capture unavailable"):
            obj.generate("a", seed=1)
        self.assertEqual(obj.pipe.transformer.forward, original)
        self.assertIs(obj.pipe.image_processor.postprocess, postprocess)
        self.assertIsNone(obj._transformer_graph.static_inputs)
        self.assertIsNone(obj._transformer_graph.graph)
        self.assertFalse(obj._next_lock.locked())
        self.assertEqual(obj.torch.cuda.graphs[0].resets, 1)

    def test_budget_cancel_after_replay_releases_graph_and_keeps_failure_evidence(self):
        obj = adapter()
        original = obj.pipe.transformer.forward
        obj.set_candidate("transformer_graph")
        def budget():
            if obj._transformer_graph.replays:
                raise RuntimeError("budget cancelled")
        with self.assertRaisesRegex(RuntimeError, "budget cancelled"):
            obj.generate("a", seed=1, check_budget=budget)
        self.assertEqual(obj.pipe.transformer.forward, original)
        self.assertEqual(obj.torch.cuda.graphs[0].resets, 1)
        self.assertIsNone(obj._transformer_graph.static_output)
        self.assertGreaterEqual(obj.torch.cuda.syncs, 2)  # request entry + failure teardown
        self.assertIn("budget cancelled", obj.candidate_summary()["failures"][0]["error"])
        obj.set_candidate("eager")
        obj.generate("a", seed=1)

    def test_graph_requires_warm_capture_before_steady_and_no_extra_replay_sync(self):
        obj = adapter()
        obj.set_candidate("transformer_graph")
        with obj.steady_context(), self.assertRaisesRegex(RuntimeError, "not captured"):
            obj.generate("a", seed=1)
        obj.set_candidate("transformer_graph")
        obj.generate("a", seed=1)
        syncs = obj.torch.cuda.syncs
        with obj.steady_context():
            obj.generate("b", seed=3)
        self.assertEqual(obj.torch.cuda.syncs - syncs, 2)  # parent's entry/completion only
        self.assertEqual(obj._transformer_graph.capture_count, 1)

    def test_failed_cuda_teardown_cannot_be_reused_as_an_eager_fallback(self):
        obj = adapter()
        obj.set_candidate("transformer_graph")
        obj.generate("a", seed=1)
        obj.torch.cuda.graphs[0].reset = lambda: (_ for _ in ()).throw(RuntimeError("poisoned graph"))
        def cancel():
            raise RuntimeError("cancel")
        with self.assertRaisesRegex(RuntimeError, "cancel"):
            obj.generate("a", seed=1, check_budget=cancel)
        self.assertTrue(obj.candidate_summary()["cleanup_failed"])
        with self.assertRaisesRegex(RuntimeError, "fresh process"):
            obj.set_candidate("eager")

    def test_graph_rejects_training_or_adapter_mutations(self):
        for attributes in (dict(training=True), dict(peft_config={"adapter": {}}), dict(_hf_hook=object())):
            obj = adapter()
            obj.set_candidate("transformer_graph")
            for key, value in attributes.items():
                setattr(obj.pipe.transformer, key, value)
            with self.subTest(attributes=attributes), self.assertRaises(RuntimeError):
                obj.generate("a", seed=1)
            self.assertFalse(obj._transformer_graph.summary()["captured"])

    def test_extra_diagnostic_spans_include_each_transformer_and_output_without_sync(self):
        obj = adapter(extra=True)
        value, service = obj.generate("a", seed=1)
        stages = service["stages"]
        self.assertEqual(sum(row["stage"] == "transformer" for row in stages), 2)
        for name in ("text_encode", "vae_decode", "output_resize_and_crop_tensor", "output_postprocess"):
            self.assertIn(name, [row["stage"] for row in stages])
        for row in stages:
            self.assertGreaterEqual(row["end_ms"], row["start_ms"])
        self.assertEqual(obj.torch.cuda.syncs, 2)
        obj.profile_extra = False
        _, service = obj.generate("a", seed=1)
        self.assertEqual(len(service["stages"]), 2)

    def test_original_generate_contract_forwards_exact_seeds_and_encoded_owners(self):
        obj = adapter()
        encoded = object()
        observed = []
        def parent(instance, prompts, *, seeds, encoded, check_budget):
            observed.append((prompts, seeds, encoded, check_budget))
            return "images", dict(stages=[], origin_perf_counter_s=0.)
        callback = lambda: None
        with patch.object(OptimizedSanaAdapter, "generate_batch", parent):
            images, _ = obj.generate_batch(["b", "a"], seeds=[202, 101], encoded=encoded, check_budget=callback)
        self.assertEqual(images, "images")
        self.assertEqual(observed, [(["b", "a"], [202, 101], encoded, callback)])

    def test_busy_owner_refuses_generation_selection_and_close(self):
        obj = adapter()
        obj._next_lock.acquire()
        try:
            for operation in (lambda: obj.generate("a"), lambda: obj.set_backend("eager"),
                              lambda: obj.set_candidate("eager"), obj.close):
                with self.assertRaises(RuntimeError):
                    operation()
        finally:
            obj._next_lock.release()

    def test_close_restores_module_forwards_and_resets_graph(self):
        obj = adapter()
        model, original = obj.pipe.transformer, obj.pipe.transformer.forward
        obj.set_candidate("transformer_graph")
        obj.generate("a", seed=1)
        graph = obj.torch.cuda.graphs[0]
        obj.close()
        self.assertEqual(model.forward, original)
        self.assertEqual(graph.resets, 1)
        self.assertIsNone(obj.pipe)


if __name__ == "__main__":
    unittest.main()
