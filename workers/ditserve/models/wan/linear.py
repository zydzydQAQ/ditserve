"""Opt-in SP2 Linear row-shape repair for pinned Wan execution.

The operation matches the qualified ``full_rows_except_ffn1`` diagnostic:
zero-pad local token rows to the original GEMM shape, then retain the local
rows. It changes neither parameters nor process-global arithmetic flags.
"""

from .parallel import ParallelContractError


LINEAR_POLICIES = ("native", "full_rows_except_ffn1", "full_rows_except_ffn1_scratch")
SCRATCH_LIMIT_BYTES = 256 * 1024**2


def validate_linear_policy(value):
    if value not in LINEAR_POLICIES:
        raise ParallelContractError("unsupported SP Linear policy")
    return value


class FullRowsLinearRepair:
    def __init__(self, backend):
        self.backend = backend
        self.hooks, self.modules, self.pending = [], [], {}
        self.geometry, self.audit = None, None
        self.scratch = None
        self.scratch_stream = None
        self.scratch_busy = False
        self.profiler = None

    def attach(self):
        if self.hooks:
            raise ParallelContractError("Linear repair is already attached")
        torch = self.backend.worker.torch
        model = self.backend.worker.pipe.transformer
        selected = [("blocks.%d.%s" % (index, name), module)
                    for index, block in enumerate(model.blocks)
                    for name, module in block.named_modules()
                    if isinstance(module, torch.nn.Linear) and not name.endswith("ffn.net.0.proj")]
        selected.extend(("proj_out" + ("." + name if name else ""), module)
                        for name, module in model.proj_out.named_modules()
                        if isinstance(module, torch.nn.Linear))
        if not selected or len({id(module) for _, module in selected}) != len(selected):
            raise ParallelContractError("Linear repair requires distinct pinned token modules")
        try:
            for name, module in selected:
                self.modules.append(name)
                self.pending[name] = []
                self.hooks.append(module.register_forward_pre_hook(
                    lambda m, a, k, name=name: self._before(name, m, a, k), with_kwargs=True))
                # The output must be cropped before proj_out's existing SP gather.
                self.hooks.append(module.register_forward_hook(
                    lambda m, a, out, name=name: self._after(name, m, out),
                    prepend=True, always_call=True))
        except BaseException:
            self.close()
            raise
        return self

    def bind(self, state):
        """Rebuild geometry per quantum, including resumed mid-branch packets."""
        if self.geometry is not None or any(self.pending.values()):
            raise ParallelContractError("Linear repair still owns an active quantum")
        layout = self.backend.handle.current
        shape = tuple(state.latents.shape)
        patch = tuple(self.backend.worker.pipe.transformer.config.patch_size)
        if (layout.degree not in (1, 2) or len(shape) != 5 or len(patch) != 3
                or any(dimension % stride for dimension, stride in zip(shape[2:], patch))):
            raise ParallelContractError("Linear repair requires divisible Wan patch geometry and SP1/SP2")
        sequence = 1
        for dimension, stride in zip(shape[2:], patch):
            sequence *= dimension // stride
        if sequence % layout.degree:
            raise ParallelContractError("Linear repair token count does not divide the layout")
        self.geometry = dict(batch=shape[0], global_sequence=sequence,
                             local_sequence=sequence // layout.degree,
                             ranks=tuple(layout.ranks), epoch=layout.epoch)
        self.audit = dict(policy=self.backend.linear_policy, degree=layout.degree,
                          request_id=state.request_id, step_index=state.step_index,
                          global_sequence=sequence, local_sequence=sequence // layout.degree,
                          modules=len(self.modules), applied_calls=0, skipped_calls=0,
                          unchanged_sp1_calls=0, scratch_hits=0, scratch_allocations=0,
                          scratch_fallbacks=0, scratch_peak_bytes=0,
                          allocation_bytes=0, zero_bytes=0, copy_bytes=0,
                          scratch_limit_bytes=SCRATCH_LIMIT_BYTES)

    def _expanded_input(self, value, start, end):
        """A single quantum-local buffer; never hold one full tensor per layer.

        All calls in a quantum use one stream. The output is cloned before a
        later module can overwrite scratch. Oversized inputs retain the old
        allocation path. Clearing at unbind also isolates requests and epochs.
        """
        shape = (value.shape[0], self.geometry["global_sequence"], value.shape[2])
        elements = shape[0] * shape[1] * shape[2]
        size = elements * value.element_size()
        scratch = self.backend.linear_policy == "full_rows_except_ffn1_scratch"
        if not scratch or size > SCRATCH_LIMIT_BYTES:
            self.audit["scratch_fallbacks"] += int(scratch)
            self.audit["allocation_bytes"] += size
            self.audit["zero_bytes"] += size
            full = value.new_zeros(shape)
        else:
            torch = self.backend.worker.torch
            stream = torch.cuda.current_stream(value.device) if value.is_cuda else None
            identity = (str(value.device), value.dtype, None if stream is None else stream.cuda_stream)
            if self.scratch_busy or (self.scratch_stream is not None and self.scratch_stream != identity):
                raise ParallelContractError("Linear scratch cannot overlap calls or change streams within a quantum")
            self.scratch_stream = identity
            if self.scratch is None or self.scratch.numel() < elements:
                # Drop the old allocation before growing. CUDA's caching
                # allocator retains stream ordering on the same stream.
                self.scratch = None
                self.scratch = value.new_empty(elements)
                self.audit["scratch_allocations"] += 1
                self.audit["allocation_bytes"] += size
            else:
                self.audit["scratch_hits"] += 1
            self.audit["scratch_peak_bytes"] = max(
                self.audit["scratch_peak_bytes"], self.scratch.numel() * value.element_size())
            full = self.scratch[:elements].view(shape)
            # Local rows are overwritten completely, so only the peer rows
            # need zeroing, including when the feature width changes.
            full[:, :start, :].zero_()
            full[:, end:, :].zero_()
            self.audit["zero_bytes"] += size - value.numel() * value.element_size()
            self.scratch_busy = True
        full[:, start:end, :] = value
        self.audit["copy_bytes"] += value.numel() * value.element_size()
        return full

    def _before(self, name, module, args, kwargs):
        self.backend._require_active()
        geometry, layout = self.geometry, self.backend.handle.current
        if (geometry is None or geometry["ranks"] != tuple(layout.ranks)
                or geometry["epoch"] != layout.epoch):
            raise ParallelContractError("Linear repair lacks current request geometry")
        value = args[0] if args else kwargs.get("input")
        if value is None or not hasattr(value, "shape"):
            raise ParallelContractError("Linear repair input is missing")
        if (layout.degree == 1 or value.ndim != 3
                or value.shape[1] != geometry["local_sequence"]):
            self.pending[name].append(None)
            self.audit["skipped_calls"] += 1
            self.audit["unchanged_sp1_calls"] += int(layout.degree == 1)
            return
        if value.shape[0] != geometry["batch"] or value.shape[2] != module.in_features:
            raise ParallelContractError("Linear token shape differs from request patch geometry")
        start = tuple(layout.ranks).index(self.backend.handle.rank) * geometry["local_sequence"]
        end = start + geometry["local_sequence"]
        profile = self.profiler.begin(name) if self.profiler is not None else None
        full = self._expanded_input(value, start, end)
        if profile is not None:
            self.profiler.prepared(profile)
        self.pending[name].append((start, end, value.shape[0], geometry["global_sequence"], profile))
        self.audit["applied_calls"] += 1
        if args:
            return (full,) + args[1:], kwargs
        return args, dict(kwargs, input=full)

    def _after(self, name, module, output):
        # always_call clears pending bookkeeping even when the GEMM raises.
        if not self.pending[name]:
            return
        item = self.pending[name].pop()
        if item is None:
            return
        self.scratch_busy = False
        if output is None:
            return
        start, end, batch, sequence, profile = item
        if output.ndim != 3 or tuple(output.shape) != (batch, sequence, module.out_features):
            raise ParallelContractError("expanded Linear output has unexpected shape")
        if profile is not None:
            self.profiler.computed(profile)
        result = output[:, start:end, :].clone(memory_format=self.backend.worker.torch.contiguous_format)
        if profile is not None:
            self.profiler.cropped(profile)
        return result

    def unbind(self):
        if any(self.pending.values()):
            raise ParallelContractError("Linear repair has unfinished module calls")
        audit = dict(self.audit) if self.audit is not None else None
        self.scratch = self.scratch_stream = None
        self.scratch_busy = False
        self.geometry = self.audit = None
        return audit

    def abort(self):
        for values in self.pending.values():
            values.clear()
        self.geometry = self.audit = None
        self.scratch = self.scratch_stream = None
        self.scratch_busy = False

    def close(self):
        for hook in reversed(self.hooks):
            hook.remove()
        self.hooks.clear()
        self.abort()
        self.pending.clear()
        self.modules.clear()
