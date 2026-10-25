"""Optional CUDA event diagnostics; no model loading or experiment control."""

import time


def tensor_inventory(value, torch):
    seen, rows = set(), []

    def visit(item, path):
        if id(item) in seen:
            return
        if torch.is_tensor(item):
            seen.add(id(item))
            rows.append(dict(path=path, shape=list(item.shape), dtype=str(item.dtype),
                device=str(item.device), bytes=int(item.numel() * item.element_size())))
        elif isinstance(item, (dict, list, tuple)):
            seen.add(id(item))
            for key, child in (item.items() if isinstance(item, dict) else enumerate(item)):
                visit(child, path + "." + str(key))
    visit(value, "state")
    return dict(tensors=rows, bytes=sum(row["bytes"] for row in rows),
                scope="logical_tensor_payload_not_allocated_storage_or_checkpoint")


class PipelineProfiler:
    """Restore every hook/method on failure; resolve events after caller syncs."""

    def __init__(self, torch, pipe, device):
        if not str(device).startswith("cuda"):
            raise ValueError("pipeline profiling requires CUDA; CPU timings are not substituted")
        self.torch, self.device = torch, device
        self.pipe = getattr(pipe, "module", pipe)
        self.handles, self.patches, self.spans, self.forwards, self.steps = [], [], [], [], []
        self.pending, self.finite_checks, self.step_start = [], [], 0

    def _begin(self):
        start, end = (self.torch.cuda.Event(enable_timing=True) for unused in range(2))
        start.record()
        return start, end, time.perf_counter()

    def _end(self, name, span):
        start, end, host = span
        end.record()
        row = dict(phase=name, host_ms=(time.perf_counter() - host) * 1000,
                   start=start, end=end)
        self.spans.append(row)
        return row

    def _before(self, module, inputs):
        self.pending.append(self._begin())

    def _after(self, module, inputs, output):
        self.forwards.append(self._end("transformer", self.pending.pop()))

    def _patch(self, owner, name, phase):
        if owner is None or not callable(getattr(owner, name, None)):
            return
        original, own = getattr(owner, name), name in vars(owner)
        previous = vars(owner).get(name)
        def wrapped(*args, **kwargs):
            span = self._begin()
            try:
                return original(*args, **kwargs)
            finally:
                self._end(phase, span)
        setattr(owner, name, wrapped)
        self.patches.append((owner, name, own, previous))

    def __enter__(self):
        cuda = self.torch.cuda
        cuda.reset_peak_memory_stats(self.device)
        self.initial_allocated = int(cuda.memory_allocated(self.device))
        try:
            transformer = self.pipe.transformer
            self.handles.append(transformer.register_forward_pre_hook(self._before))
            self.handles.append(transformer.register_forward_hook(self._after))
            scheduler = getattr(self.pipe.scheduler, "module", self.pipe.scheduler)
            for owner, name, phase in ((self.pipe, "encode_prompt", "text_encode"),
                    (self.pipe, "prepare_latents", "latent_init"),
                    (scheduler, "step", "scheduler"),
                    (self.pipe.vae, "decode", "vae_decode"),
                    (getattr(self.pipe, "image_processor", None), "postprocess", "postprocess"),
                    (getattr(self.pipe, "video_processor", None), "postprocess_video", "postprocess")):
                self._patch(owner, name, phase)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        for owner, name, own, previous in reversed(self.patches):
            if own:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.patches.clear()

    def callback(self, pipe, index, timestep, values):
        if "latents" not in values:
            raise ValueError("profile callback did not expose latents")
        scheduler = getattr(pipe.scheduler, "module", pipe.scheduler)
        self.steps.append(dict(step_index=int(index), forward_indices=list(range(self.step_start, len(self.forwards))),
            latents=tensor_inventory(values["latents"], self.torch),
            sampler=tensor_inventory({key: value for key, value in vars(scheduler).items()
                                      if key != "_internal_dict"}, self.torch)))
        self.finite_checks.append(self.torch.isfinite(values["latents"]).all())
        self.step_start = len(self.forwards)
        return values

    def report(self, expected_steps):
        if ([row["step_index"] for row in self.steps] != list(range(expected_steps))
                or not self.forwards or self.step_start != len(self.forwards) or self.pending):
            raise ValueError("incomplete transformer/callback profiling; unsupported pipeline")
        phases, totals = [], {}
        for span in self.spans:
            row = dict(phase=span["phase"], host_ms=span["host_ms"],
                       cuda_ms=float(span["start"].elapsed_time(span["end"])))
            phases.append(row)
            totals[row["phase"]] = totals.get(row["phase"], 0.) + row["cuda_ms"]
        for step, finite in zip(self.steps, self.finite_checks):
            step["latents_finite"] = bool(finite.item())
            step["transformer_cuda_ms"] = sum(float(self.forwards[i]["start"].elapsed_time(self.forwards[i]["end"]))
                                                for i in step["forward_indices"])
        if not all(row["latents_finite"] for row in self.steps):
            raise ValueError("nonfinite latents in instrumented pipeline")
        cuda = self.torch.cuda
        return dict(phase_ms=totals, profile=dict(status="measured", instrumented=True,
            scope="CUDA current-stream event spans and host spans; not kernel attribution or uninstrumented latency",
            steps=self.steps, stages=phases, forward_count=len(self.forwards),
            memory=dict(initial_allocated_bytes=self.initial_allocated,
                peak_allocated_bytes=int(cuda.max_memory_allocated(self.device)),
                peak_reserved_bytes=int(cuda.max_memory_reserved(self.device)),
                scope="PyTorch process allocator, not total device memory")))
