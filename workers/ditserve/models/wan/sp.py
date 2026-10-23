"""Pinned Wan Ulysses attachment, with no implicit GPU launch.

The attachment supplies split/attention/gather operations for an aligned rank
group. The opt-in two-rank elastic adapter supplies request ownership and
block-quantum binding; GPU qualification remains a separate experiment gate.
Imports are CPU-only/lazy.
"""

import importlib
import importlib.metadata

from .parallel import ParallelContractError, RequestGroupGate


def shard_patch_output(tensor, local_rank, degree):
    """Wan immediately flattens Conv3d output into a token sequence.

    Shard before the transformer blocks. Spatial coordinates are supplied by
    separately sharded RoPE; unpatchification dimensions remain those computed
    from the original input.
    """
    if degree not in (1, 2, 4, 8) or not 0 <= local_rank < degree or tensor.ndim != 5:
        raise ParallelContractError("invalid Wan patch split")
    if degree == 1:
        return tensor
    flat = tensor.flatten(2)
    if flat.shape[2] % degree:
        raise ParallelContractError("Wan token count must divide the SP degree")
    local = flat.chunk(degree, dim=2)[local_rank].contiguous()
    return local.reshape(tensor.shape[0], tensor.shape[1], 1, 1, local.shape[2])


def shard_rope_output(value, local_rank, degree):
    if type(value) is not tuple or len(value) != 2:
        raise ParallelContractError("Wan RoPE must return a cosine/sine pair")
    if degree not in (1, 2, 4, 8) or not 0 <= local_rank < degree:
        raise ParallelContractError("invalid RoPE split")
    if any(tensor.ndim != 4 or tensor.shape[1] % degree for tensor in value):
        raise ParallelContractError("unsupported Wan RoPE shape")
    if degree == 1:
        return value
    return tuple(tensor.chunk(degree, dim=1)[local_rank].contiguous() for tensor in value)


class PinnedWanUlyssesBackend:
    """Attach once, then call step(state) on every rank in the active group.

    Use a fresh worker process; dependency attention registration is process
    local. The original worker source and dependency files are never modified.
    """

    capabilities = dict(
        backend="cache-dit-1.5-ulysses-native-sdpa-flash",
        fixed_sp_degrees=[1, 2, 4, 8], explicit_sp1_bypass=True,
        local_layout_hook_binding=True, distributed_request_transport=False,
        gpu_validated=False, elastic_model_equivalence_validated=False,
    )

    def __init__(self, handle, groups, *, linear_policy="native"):
        from .linear import validate_linear_policy
        self.linear_policy = validate_linear_policy(linear_policy)
        self._linear_repair = None
        self.handle, self.groups = handle, groups
        self.gate = RequestGroupGate(handle, groups)
        self.worker = None
        self._hooks = []
        self._processors = []
        self._configs = {}
        self._active_config = None
        self._bound_epoch = None

    def attach(self, worker):
        if self.worker is not None:
            raise ParallelContractError("backend is already attached")
        expected = {"torch": "2.10.0+cu126", "diffusers": "0.36.0", "cache-dit": "1.5.0"}
        for package, version in expected.items():
            if importlib.metadata.version(package) != version:
                raise ParallelContractError("unsupported pinned dependency: " + package)
        torch = importlib.import_module("torch")
        if self.groups.backend != "nccl" or not torch.cuda.is_available():
            raise ParallelContractError("Wan execution requires initialized NCCL and an available GPU")
        wan = importlib.import_module("diffusers.models.transformers.transformer_wan")
        model = worker.pipe.transformer
        if type(model) is not wan.WanTransformer3DModel:
            raise ParallelContractError("only the pinned WanTransformer3DModel is supported")
        if (tuple(model.config.patch_size) != (1, 2, 2)
                or model.config.num_attention_heads != 12 or model.config.num_layers != 30
                or model.config.in_channels != 16 or model.config.out_channels != 16
                or model.config.added_kv_proj_dim is not None):
            raise ParallelContractError("only Wan2.1 T2V 1.3B is supported")
        if worker.attention != "flash":
            raise ParallelContractError("research SP requires the explicit Flash attention worker")
        if (getattr(model, "_cp_config", None) is not None
                or getattr(model, "_parallel_config", None) is not None
                or getattr(model, "_is_parallelized", False)):
            raise ParallelContractError("model already has a parallel attachment")
        if getattr(model, "gradient_checkpointing", False):
            raise ParallelContractError("gradient checkpointing is outside inference scope")
        cp = importlib.import_module("cache_dit.distributed.core._modeling_parallel")
        for ranks in self.groups.groups:
            if len(ranks) == 1 or self.handle.rank not in ranks:
                continue
            config = cp._ContextParallelConfig(ulysses_degree=len(ranks), ring_degree=1)
            # Ulysses consumes only this 1D mesh. Reuse the already-created group;
            # do not create communicators while handling a request.
            config._ulysses_mesh = torch.distributed.device_mesh.DeviceMesh.from_group(
                self.groups.group(ranks), "cuda", mesh=list(ranks), mesh_dim_names=("ulysses",))
            config._rank = ranks.index(self.handle.rank)
            config._world_size = len(ranks)
            config._device = torch.device(worker.device)
            config._ulysses_local_rank = ranks.index(self.handle.rank)
            self._configs[ranks] = config
        attention = importlib.import_module("cache_dit.attention")
        attention._maybe_register_custom_attn_backends()
        patch = importlib.import_module("cache_dit.distributed.transformers.wan")
        processor_call = getattr(patch, "__patch_WanAttnProcessor__call__")
        backend = self

        class RequestProcessor(wan.WanAttnProcessor):
            _attention_backend = "native"

            def __call__(self, *args, **kwargs):
                backend._require_active()
                self._cp_config = backend._active_config
                self._parallel_config = backend._active_config
                return processor_call(self, *args, **kwargs)

        modules = [module for module in model.modules() if isinstance(module, wan.WanAttention)]
        if not modules or any(type(module.processor) is not wan.WanAttnProcessor for module in modules):
            raise ParallelContractError("attention processors differ from the pinned native recipe")
        self.worker = worker
        try:
            for module in modules:
                self._processors.append((module, module.processor))
                module.set_processor(RequestProcessor())
            self._hooks.append(model.patch_embedding.register_forward_hook(self._patch_hook))
            self._hooks.append(model.rope.register_forward_hook(self._rope_hook))
            self._hooks.append(model.proj_out.register_forward_hook(self._gather_hook))
            if self.linear_policy != "native":
                from .linear import FullRowsLinearRepair
                self._linear_repair = FullRowsLinearRepair(self).attach()
        except BaseException:
            self.close()
            raise
        return self

    def _require_active(self):
        if (self.worker is None or self.handle.active is None or self.handle.failed
                or self._bound_epoch != self.handle.current.epoch):
            raise ParallelContractError("Wan execution is outside its committed step")

    def _patch_hook(self, module, args, output):
        self._require_active()
        layout = self.handle.current
        return shard_patch_output(output, layout.ranks.index(self.handle.rank), layout.degree)

    def _rope_hook(self, module, args, output):
        self._require_active()
        layout = self.handle.current
        return shard_rope_output(output, layout.ranks.index(self.handle.rank), layout.degree)

    def _gather_hook(self, module, args, output):
        self._require_active()
        layout = self.handle.current
        if output.ndim != 3:
            raise ParallelContractError("Wan projected tokens must have three dimensions")
        if layout.degree == 1:
            return output
        torch = importlib.import_module("torch")
        chunks = [torch.empty_like(output) for _ in layout.ranks]
        torch.distributed.all_gather(chunks, output.contiguous(), group=self.groups.group(layout.ranks))
        return torch.cat(chunks, dim=1)

    def begin_step(self, state):
        if self.worker is None:
            raise ParallelContractError("attach a worker first")
        self.handle.begin_step(state.request_id, state.step_index,
                               cfg="paired" if state.spec["guidance_scale"] > 1 else "cond")
        self._bound_epoch = self.handle.current.epoch
        self._active_config = self._configs.get(self.handle.current.ranks)
        try:
            if self._linear_repair is not None:
                self._linear_repair.bind(state)
            self.gate.consensus(dict(kind="pre_step", spec=state.spec, latents_shape=list(state.latents.shape),
                                     latents_dtype=str(state.latents.dtype), sp_linear_policy=self.linear_policy))
        except BaseException:
            self.abort_linear_repair()
            self.handle.fail()
            raise

    def begin_quantum(self, continuation, handle, quantum_seq, blocks):
        """Bind a request only while its synchronized block quantum executes."""
        if self.handle.active is not None:
            raise ParallelContractError("another request is executing on this worker")
        self.handle = handle
        self.gate = RequestGroupGate(handle, self.groups)
        self.begin_step(continuation.base)
        self.gate.consensus(dict(kind="block_begin", quantum_seq=quantum_seq,
                                branch=continuation.branch, next_block=continuation.next_block,
                                blocks=blocks))

    def end_quantum(self, continuation, quantum_seq):
        self._require_active()
        self.worker._sync()
        self.gate.consensus(dict(kind="block_complete", quantum_seq=quantum_seq,
                                step=continuation.base.step_index, branch=continuation.branch,
                                next_block=continuation.next_block,
                                packet=continuation.packet is not None,
                                conditional=continuation.conditional is not None))
        audit = self.linear_repair_audit()
        audit["quantum_seq"] = quantum_seq
        self.handle.end_step()
        self._active_config = self._bound_epoch = None
        return audit

    def end_step(self):
        self._require_active()
        self.worker._sync()
        self.gate.consensus(dict(kind="step_complete"))
        audit = self.linear_repair_audit()
        self.handle.end_step()
        self._active_config = self._bound_epoch = None
        return audit

    def linear_repair_audit(self):
        if self._linear_repair is not None:
            return self._linear_repair.unbind()
        return dict(policy=self.linear_policy, degree=self.handle.current.degree,
                    request_id=self.handle.active["request_id"],
                    step_index=self.handle.active["step_index"], modules=0,
                    applied_calls=0, skipped_calls=0, unchanged_sp1_calls=0)

    def abort_linear_repair(self):
        if self._linear_repair is not None:
            self._linear_repair.abort()

    def step(self, state):
        """One real worker step, with group agreement before and after execution."""
        if self.worker is None:
            raise ParallelContractError("attach a worker first")
        with self.worker._lock:
            self.begin_step(state)
            try:
                result = self.worker.step(state)
                result["linear_repair"] = self.end_step()
                return result
            except BaseException:
                self.abort_linear_repair()
                self.handle.fail()
                raise

    def close(self):
        # Caller must have synchronized/terminated all active ranks beforehand.
        if self._linear_repair is not None:
            self._linear_repair.close()
            self._linear_repair = None
        for hook in reversed(self._hooks):
            hook.remove()
        self._hooks.clear()
        for module, processor in self._processors:
            module.set_processor(processor)
        self._processors.clear()
        self.worker = None
        self._configs.clear()
        self._active_config = self._bound_epoch = None
