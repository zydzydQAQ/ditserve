"""Bounded SANA compilation/batching adapter; importing never starts GPU work."""

from contextlib import nullcontext
import time

from ..directions.sana import MODEL_ID, RECIPE, SanaAdapter


class OptimizedSanaAdapter(SanaAdapter):
    """Keep the official recipe and independent per-request CPU RNG streams.

    Only the transformer is compiled. The scheduler, text encoder and VAE stay
    eager. A pipeline is mutable and must have exactly one calling thread.
    """

    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        self._eager_transformer = self.pipe.transformer
        self._compiled_transformer = None
        self.backend = "eager"
        self.compile_records = []
        self.compile_setup_ms = 0.
        self._busy = False

    def compile_transformer(self):
        """Construct a lazy compiler wrapper; first forward performs compilation."""
        if self._compiled_transformer is None:
            from torch._dynamo.backends.registry import lookup_backend
            inductor = lookup_backend("inductor")

            def counted_inductor(graph, inputs, **kwargs):
                record = dict(index=len(self.compile_records), status="running")
                self.compile_records.append(record)
                started = time.perf_counter()
                try:
                    result = inductor(graph, inputs,
                                      config_patches={"triton.cudagraphs": False}, **kwargs)
                    record["status"] = "completed"
                    return result
                except BaseException as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    record["backend_compile_ms"] = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            self._compiled_transformer = self.torch.compile(
                self._eager_transformer, backend=counted_inductor, fullgraph=True,
                dynamic=False)
            self.compile_setup_ms += (time.perf_counter() - started) * 1000
        return self.compilation_summary()

    def compilation_summary(self):
        return dict(backend="inductor", fullgraph=True, dynamic=False, cudagraphs=False,
                    wrapper_setup_ms=self.compile_setup_ms,
                    graph_compile_count=len(self.compile_records),
                    records=[dict(row) for row in self.compile_records])

    def set_backend(self, backend):
        if self._busy:
            raise RuntimeError("cannot change backend during a Sana invocation")
        if backend not in ("eager", "compiled"):
            raise ValueError("Sana backend must be eager or compiled")
        if backend == "compiled":
            self.compile_transformer()
        self.pipe.transformer = (self._compiled_transformer if backend == "compiled"
                                 else self._eager_transformer)
        self.backend = backend

    def steady_context(self):
        return (self.torch.compiler.set_stance("fail_on_recompile")
                if self.backend == "compiled" else nullcontext())

    def memory(self):
        return dict(allocated_bytes=int(self.torch.cuda.memory_allocated(self.device)),
                    reserved_bytes=int(self.torch.cuda.memory_reserved(self.device)),
                    peak_allocated_bytes=int(self.torch.cuda.max_memory_allocated(self.device)))

    def generate(self, prompt, *, seed=0, encoded=None, check_budget=lambda: None):
        return self.generate_batch([prompt], seeds=[seed],
                                   encoded=None if encoded is None else [encoded],
                                   check_budget=check_budget)

    def generate_batch(self, prompts, *, seeds, encoded=None, check_budget=lambda: None):
        import numpy as np
        prompts, seeds = list(prompts), list(seeds)
        if len(prompts) not in (1, 2) or len(seeds) != len(prompts):
            raise ValueError("Sana accepts one or two prompts with one seed per prompt")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Sana prompts must be strings")
        if any(type(seed) is not int for seed in seeds):
            raise TypeError("Sana seeds must be integers")
        if encoded is not None:
            encoded = list(encoded)
            if len(encoded) != len(prompts) or any(
                    item.owner is not self._owner or item.prompt != prompt
                    for item, prompt in zip(encoded, prompts)):
                raise ValueError("encoded Sana prompt belongs to another request/adapter")
        if self._busy:
            raise RuntimeError("concurrent calls on one Sana pipeline are unsupported")
        check_budget()
        arguments = dict(RECIPE)
        if encoded is None:
            arguments["prompt"] = prompts[0] if len(prompts) == 1 else prompts
        else:
            arguments.update(
                prompt_embeds=self.torch.cat([item.embeddings for item in encoded], dim=0),
                prompt_attention_mask=self.torch.cat([item.attention_mask for item in encoded], dim=0))
        # SCM also samples at each denoise step: pass these same streams through
        # the complete pipeline, not just through initial-latent preparation.
        generators = [self.torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
        generator = generators[0] if len(generators) == 1 else generators
        stages, event_pairs = [], []
        originals = []
        origin = time.perf_counter()

        def measured(owner, name, label):
            original = getattr(owner, name)
            originals.append((owner, name, original))
            def wrapped(*args, **kwargs):
                before = self.torch.cuda.Event(enable_timing=True)
                after = self.torch.cuda.Event(enable_timing=True)
                row = dict(stage=label, start_ms=(time.perf_counter() - origin) * 1000)
                stages.append(row)
                before.record()
                try:
                    return original(*args, **kwargs)
                finally:
                    after.record()
                    row["end_ms"] = (time.perf_counter() - origin) * 1000
                    row["host_ms"] = row["end_ms"] - row["start_ms"]
                    event_pairs.append((row, before, after))
            setattr(owner, name, wrapped)

        def boundary(pipe, index, timestep, kwargs):
            stages.append(dict(stage="denoise_step_submitted", index=int(index),
                               at_ms=(time.perf_counter() - origin) * 1000))
            check_budget()
            return kwargs

        self._sync()
        self.torch.cuda.reset_peak_memory_stats(self.device)
        memory_before = self.memory()
        compiles_before = len(self.compile_records)
        self._busy = True
        # Outer pipeline methods provide timing without instrumenting the FX
        # graph or inserting extra GPU synchronization between model stages.
        origin = time.perf_counter()
        try:
            measured(self.pipe, "encode_prompt", "text_encode")
            measured(self.pipe.vae, "decode", "vae_decode")
            with self.torch.inference_mode():
                result = self.pipe(**arguments, generator=generator, output_type="np",
                                   callback_on_step_end=boundary)
            self._sync()
            duration = (time.perf_counter() - origin) * 1000
            for row, before, after in event_pairs:
                row["cuda_elapsed_ms"] = float(before.elapsed_time(after))
            images = result.images
            if (not isinstance(images, np.ndarray) or images.ndim != 4
                    or images.shape[0] != len(prompts) or images.shape[-1] != 3
                    or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1):
                raise RuntimeError("invalid Sana output: expected finite [N,H,W,3] in [0,1]")
            check_budget()
            return images, dict(ms=duration, backend=self.backend, batch_size=len(prompts),
                                origin_perf_counter_s=origin,
                                cost_per_image_ms=duration / len(prompts),
                                prompt_preencoded=encoded is not None,
                                encode_ms=None if encoded is None else sum(item.encode_ms for item in encoded),
                                recipe=dict(RECIPE), finite=True, stages=stages,
                                stage_scope="host enqueue spans and same-stream CUDA events; no extra stage sync",
                                memory_before=memory_before, memory_after=self.memory(),
                                peak_allocated_bytes=self.memory()["peak_allocated_bytes"],
                                compile_delta=len(self.compile_records) - compiles_before,
                                completion_scope="all batch members complete after full pipeline output")
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)
            self._busy = False

    def close(self):
        super().close()
        self._compiled_transformer = self._eager_transformer = None
