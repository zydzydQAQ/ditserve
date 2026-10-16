"""Explicit, fail-closed Sana candidates; importing does not load torch/models.

The original pipeline, modules, BF16 recipe, scheduler RNG, and VAE tiling stay
unchanged. Only selected module forwards are routed through a compiled region
or an eager CUDA graph. Neither candidate is a quality qualification.
"""

from contextlib import contextmanager, nullcontext
import math
import threading
import time

from .compile import OptimizedSanaAdapter


def _inductor_backend():
    from torch._dynamo.backends.registry import lookup_backend
    return lookup_backend("inductor")


class _CompiledRegion:
    def __init__(self, owner, name, original):
        self.owner, self.name, self.original = owner, name, original
        self.records, self.invocations, self.calls, self.failures = [], 0, 0, []
        started = time.perf_counter()
        self.compiled = owner.torch.compile(original, backend=self._backend,
                                            fullgraph=True, dynamic=False)
        self.wrapper_setup_ms = (time.perf_counter() - started) * 1000

    def _backend(self, graph, inputs, **kwargs):
        record = dict(index=len(self.records), status="running")
        self.records.append(record)
        started = time.perf_counter()
        try:
            executable = _inductor_backend()(graph, inputs,
                config_patches={"triton.cudagraphs": False}, **kwargs)
            record["status"] = "completed"
        except BaseException as error:
            record.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            record["backend_compile_ms"] = (time.perf_counter() - started) * 1000

        # Deliberately no functools.wraps: copying _torchdynamo_orig_callable
        # allows Dynamo to unwrap past this actual-execution counter.
        def counted(*args, **kwargs):
            self.invocations += 1
            return executable(*args, **kwargs)
        return counted

    def __call__(self, *args, **kwargs):
        self.owner._check_budget()
        before, compiles = self.invocations, len(self.records)
        try:
            value = self.compiled(*args, **kwargs)
        except BaseException as error:
            self.failures.append(dict(error=type(error).__name__ + ": " + str(error),
                                      attempted_call=self.calls + 1))
            raise
        if self.invocations == before:
            raise RuntimeError("Sana " + self.name + " executed without a compiled graph")
        if self.owner._steady_depth and len(self.records) != compiles:
            raise RuntimeError("Sana " + self.name + " compiled during steady measurement")
        self.calls += 1
        self.owner._check_budget()
        return value

    def summary(self):
        return dict(name=self.name, calls=self.calls, graph_invocations=self.invocations,
                    graph_compile_count=len(self.records), records=[dict(row) for row in self.records],
                    wrapper_setup_ms=self.wrapper_setup_ms, fullgraph=True, dynamic=False,
                    cudagraphs=False, original_module_preserved=True,
                    execution_failures=[dict(row) for row in self.failures])


class _TransformerGraph:
    """One fixed layout, one owner; every input is copied on every invocation."""

    def __init__(self, owner, original):
        self.owner, self.original = owner, original
        self.graph = self.static_inputs = self.static_output = self.signature = None
        self.static_signature = None
        self.calls = self.replays = self.capture_count = self.warmup_calls = 0
        self.capture_records = []

    def _signature(self, value):
        torch = self.owner.torch
        if torch.is_tensor(value):
            if not value.is_cuda or str(value.device) != str(torch.device(self.owner.device)):
                raise ValueError("transformer graph requires tensors on its one resident CUDA device")
            if value.requires_grad or str(value.layout) != "torch.strided":
                raise ValueError("transformer graph requires inference strided tensors")
            return ("tensor", tuple(value.shape), str(value.dtype), str(value.device), tuple(value.stride()))
        if type(value) in (tuple, list):
            return (type(value).__name__, tuple(self._signature(item) for item in value))
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("graph keyword/container keys must be strings")
            return ("dict", tuple((key, self._signature(value[key])) for key in sorted(value)))
        if value is None or type(value) in (bool, int, str):
            return (type(value).__name__, value)
        if type(value) is float and math.isfinite(value):
            return ("float", value)
        raise ValueError("unsupported mutable/non-tensor transformer graph argument")

    def _clone(self, value, *, inputs=False):
        torch = self.owner.torch
        if torch.is_tensor(value):
            result = value.detach().clone(memory_format=torch.preserve_format)
            # Expanded timestep/guidance views commonly have stride zero.
            # Their private clone is dense, so every scalar is copyable without
            # writing overlapping storage. Values/dtype remain identical; the
            # source layout is still checked on every subsequent invocation.
            return result
        if type(value) is tuple:
            return tuple(self._clone(item, inputs=inputs) for item in value)
        if type(value) is list:
            return [self._clone(item, inputs=inputs) for item in value]
        if type(value) is dict:
            return {key: self._clone(item, inputs=inputs) for key, item in value.items()}
        if value is None or type(value) in (bool, int, float, str):
            return value
        raise ValueError("unsupported transformer graph output container")

    def _copy(self, target, source):
        if self.owner.torch.is_tensor(source):
            target.copy_(source)
        elif type(source) in (tuple, list):
            for dst, src in zip(target, source):
                self._copy(dst, src)
        elif type(source) is dict:
            for key in source:
                self._copy(target[key], source[key])

    def _capture(self, inputs, signature):
        owner, cuda = self.owner, self.owner.torch.cuda
        model = owner._eager_transformer
        if getattr(model, "training", False) or getattr(model, "peft_config", None):
            raise RuntimeError("CUDA graph candidate supports eval-only base transformer without active adapters")
        if getattr(model, "_hf_hook", None) is not None:
            raise RuntimeError("CUDA graph candidate does not support transformer offload hooks")
        if owner._steady_depth:
            raise RuntimeError("transformer graph is not captured before steady measurement")
        record = dict(status="running", scope="cold host wall; warmup/capture synchronization included")
        self.capture_records.append(record)
        started = time.perf_counter()
        try:
            owner._check_budget()
            self.static_inputs = self._clone(inputs, inputs=True)
            self.static_signature = self._signature(self.static_inputs)
            with cuda.device(owner.device):
                current = cuda.current_stream(owner.device)
                warm = cuda.Stream(device=owner.device)
                warm.wait_stream(current)
                before_warm = time.perf_counter()
                with cuda.stream(warm):
                    for unused in range(2):
                        owner._check_budget()
                        warm_output = self.original(*self.static_inputs[0], **self.static_inputs[1])
                        self.warmup_calls += 1
                    del warm_output
                current.wait_stream(warm)
                record["warmup_host_ms"] = (time.perf_counter() - before_warm) * 1000
                owner._check_budget()
                self.graph = cuda.CUDAGraph()
                before_capture = time.perf_counter()
                with cuda.graph(self.graph, stream=warm):
                    self.static_output = self.original(*self.static_inputs[0], **self.static_inputs[1])
                current.wait_stream(warm)
                record["capture_host_ms"] = (time.perf_counter() - before_capture) * 1000
            # Reject unsupported output containers before accepting the graph.
            self._signature(self.static_output)
            self.signature = signature
            self.capture_count += 1
            record["status"] = "completed"
            owner._check_budget()
        except BaseException as error:
            record.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            record["cold_host_ms"] = (time.perf_counter() - started) * 1000

    def __call__(self, *args, **kwargs):
        self.owner._check_budget()
        inputs = (args, kwargs)
        signature = self._signature(inputs)
        if self.graph is None:
            self._capture(inputs, signature)
        elif signature != self.signature:
            raise ValueError("transformer graph input shape/dtype/device/stride or constant arguments changed")
        self._copy(self.static_inputs, inputs)
        self.graph.replay()
        self.replays += 1
        # The scheduler receives separately owned tensors. Later replays cannot
        # overwrite either a prior noise prediction or a returned image.
        output = self._clone(self.static_output)
        self.calls += 1
        self.owner._check_budget()
        return output

    def release(self):
        if self.graph is not None or self.static_inputs is not None:
            # Only teardown/failure synchronizes here, never each warm replay.
            self.owner.torch.cuda.synchronize(self.owner.device)
            if self.graph is not None:
                self.graph.reset()
        self.graph = self.static_inputs = self.static_output = self.signature = None
        self.static_signature = None

    def summary(self):
        return dict(capture_count=self.capture_count, graph_replays=self.replays,
            calls=self.calls, warmup_eager_calls=self.warmup_calls, captured=self.graph is not None,
            records=[dict(row) for row in self.capture_records], fixed_layout=True,
            fresh_input_copy_each_call=True, independent_output_copy_each_call=True,
            source_input_signature=self.signature, static_input_signature=self.static_signature,
            expanded_input_copies_may_be_dense=True,
            scheduler_and_rng_outside_graph=True, compiled=False,
            scope="pure eager transformer CUDA graph; no prompt/result reuse or model precision change")


class NextSanaAdapter(OptimizedSanaAdapter):
    """Experimental VAE compile / transformer graph with explicit selection."""

    def __init__(self, model_path, device="cuda:0", *, profile_extra=False):
        super().__init__(model_path, device)
        self._init_next(profile_extra=profile_extra)

    def _init_next(self, *, profile_extra=False):
        self.candidate, self.profile_extra = "eager", bool(profile_extra)
        self._next_lock = threading.Lock()
        self._steady_depth, self._candidate_failed = 0, False
        self._cleanup_failed = False
        self._check_budget = lambda: None
        self._next_originals, self._vae_regions = [], {}
        self._transformer_forward = self._eager_transformer.forward
        self._transformer_graph = None
        self._failures = []

    def _restore(self):
        for module, name, original in reversed(self._next_originals):
            setattr(module, name, original)
        self._next_originals = []

    def _install(self, module, name, value):
        self._next_originals.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def set_backend(self, backend):
        # run_trace's existing eager selection is a compatibility notification,
        # not permission to discard an explicitly selected new candidate.
        if backend != "eager":
            raise ValueError("NextSanaAdapter uses set_candidate; legacy compiled transformer is not supported")
        if self._busy or self._next_lock.locked():
            raise RuntimeError("cannot change backend during a Sana invocation")
        self.backend = "eager"

    def set_candidate(self, candidate):
        if candidate not in ("eager", "vae_compile", "transformer_graph"):
            raise ValueError("unknown Sana candidate")
        if self._busy or self._next_lock.locked():
            raise RuntimeError("cannot change candidate during a Sana invocation")
        if self._cleanup_failed:
            raise RuntimeError("Sana graph teardown failed; a fresh process is required")
        self._restore()
        self.candidate, self._candidate_failed = candidate, False
        self.pipe.transformer = self._eager_transformer
        self.backend = "eager"
        try:
            if candidate == "vae_compile":
                decoder = getattr(self.pipe.vae, "decoder", None)
                if decoder is None or not callable(getattr(decoder, "forward", None)):
                    raise RuntimeError("Sana VAE has no pure decoder module")
                for name in ("decoder", "post_quant_conv"):
                    module = getattr(self.pipe.vae, name, None)
                    if module is None:
                        continue  # AutoencoderDC has no post_quant_conv.
                    if name not in self._vae_regions:
                        self._vae_regions[name] = _CompiledRegion(self, name, module.forward)
                    self._install(module, "forward", self._vae_regions[name])
            elif candidate == "transformer_graph":
                if self._transformer_graph is None:
                    self._transformer_graph = _TransformerGraph(self, self._transformer_forward)
                self._install(self._eager_transformer, "forward", self._transformer_graph)
        except BaseException as error:
            self._fail(error)
            raise
        return self.candidate_summary()

    def _fail(self, error):
        self._candidate_failed = True
        record = dict(candidate=self.candidate, error=type(error).__name__ + ": " + str(error))
        self._failures.append(record)
        self._restore()
        if self._transformer_graph is not None:
            try:
                self._transformer_graph.release()
            except BaseException as cleanup:
                record["cleanup_error"] = type(cleanup).__name__ + ": " + str(cleanup)
                self._cleanup_failed = True
        # Failed compiled wrappers must never be reused accidentally. Detailed
        # compile records remain available in the failure evidence.
        record["vae_regions"] = [region.summary() for region in self._vae_regions.values()]
        self._vae_regions = {}

    @contextmanager
    def steady_context(self):
        self._steady_depth += 1
        stance = (self.torch.compiler.set_stance("fail_on_recompile")
                  if self.candidate == "vae_compile" else nullcontext())
        try:
            with stance:
                yield
        finally:
            self._steady_depth -= 1

    def candidate_summary(self):
        return dict(candidate=self.candidate, status="failed" if self._candidate_failed else "selected",
            qualified=False, dtype=str(getattr(self.pipe.vae, "dtype", "unrecorded")),
            decoder_type=type(getattr(self.pipe.vae, "decoder", None)).__name__,
            vae_tiling=getattr(self.pipe.vae, "use_tiling", None),
            vae_slicing=getattr(self.pipe.vae, "use_slicing", None),
            vae_regions=[region.summary() for region in self._vae_regions.values()],
            transformer_graph=None if self._transformer_graph is None else self._transformer_graph.summary(),
            failures=[dict(row) for row in self._failures], profile_extra=self.profile_extra,
            cleanup_failed=self._cleanup_failed,
            model_structure_preserved=True, legacy_eager_does_not_reset_candidate=True,
            scope="candidate mechanics only; quality/capacity/online qualification is external")

    def generate_batch(self, prompts, *, seeds, encoded=None, check_budget=lambda: None):
        if not self._next_lock.acquire(blocking=False):
            raise RuntimeError("concurrent calls on one Sana candidate are unsupported")
        originals, extra, pairs = [], [], []
        self._check_budget = check_budget
        def measured(module, name, stage):
            original = getattr(module, name)
            originals.append((module, name, original))
            def wrapped(*args, **kwargs):
                before = self.torch.cuda.Event(enable_timing=True)
                after = self.torch.cuda.Event(enable_timing=True)
                row = dict(stage=stage, call_index=sum(item["stage"] == stage for item in extra),
                           start_s=time.perf_counter())
                extra.append(row)
                before.record()
                try:
                    return original(*args, **kwargs)
                finally:
                    after.record()
                    row["end_s"] = time.perf_counter()
                    pairs.append((row, before, after))
            setattr(module, name, wrapped)
        try:
            if self._candidate_failed:
                raise RuntimeError("Sana candidate failed; explicit set_candidate is required before retry")
            if self.profile_extra:
                measured(self._eager_transformer, "forward", "transformer")
                for name in ("resize_and_crop_tensor", "postprocess"):
                    processor = getattr(self.pipe, "image_processor", None)
                    if processor is not None and callable(getattr(processor, name, None)):
                        measured(processor, name, "output_" + name)
            images, service = super().generate_batch(prompts, seeds=seeds, encoded=encoded,
                                                     check_budget=check_budget)
            for row, before, after in pairs:
                row.update(start_ms=(row.pop("start_s") - service["origin_perf_counter_s"]) * 1000,
                           end_ms=(row.pop("end_s") - service["origin_perf_counter_s"]) * 1000,
                           cuda_elapsed_ms=float(before.elapsed_time(after)))
                row["host_ms"] = row["end_ms"] - row["start_ms"]
            service["stages"].extend(extra)
            service["stages"].sort(key=lambda row: row.get("start_ms", row.get("at_ms", 0.)))
            service.update(candidate=self.candidate, candidate_summary=self.candidate_summary(),
                profile_extra=self.profile_extra, extra_stage_scope="host enqueue spans and CUDA stream events; no extra stage synchronization; cold graph capture is separately charged")
            return images, service
        except BaseException as error:
            # First undo diagnostic wrappers; otherwise restoring a candidate
            # could be overwritten by the diagnostic finally block.
            for module, name, original in reversed(originals):
                setattr(module, name, original)
            originals = []
            self._fail(error)
            raise
        finally:
            for module, name, original in reversed(originals):
                setattr(module, name, original)
            self._check_budget = lambda: None
            self._next_lock.release()

    def close(self):
        if self._next_lock.locked() or self._busy:
            raise RuntimeError("cannot close Sana while an invocation owns it")
        self._restore()
        try:
            if self._transformer_graph is not None:
                self._transformer_graph.release()
        finally:
            self._vae_regions = {}
            self._transformer_graph = None
            super().close()
