"""Experiment-only Wan self-attention Ulysses processors.

The projection, normalization and rotary arithmetic follow diffusers v0.36.0
WanAttnProcessor (Apache-2.0). Every attention head retains the full sequence.
No dependency file or serving processor is patched by importing this module.
"""

import contextlib
from types import SimpleNamespace


SELECTIVE_VARIANTS = ("full_rows_ffn2", "full_rows_ffn2_qkv", "full_rows_except_ffn1")
TILE_VARIANTS = ("global_tile_1024", "global_tile_2048")
DIAGNOSTIC_VARIANTS = ("full_rows_linear", "no_reduced_precision") + SELECTIVE_VARIANTS + TILE_VARIANTS


def selected_linear(name, variant):
    if variant == "full_rows_except_ffn1":
        return not (name.startswith("blocks.") and name.endswith(".ffn.net.0.proj"))
    if variant == "full_rows_ffn2":
        return name.startswith("blocks.") and name.endswith(".ffn.net.2")
    if variant == "full_rows_ffn2_qkv":
        return (selected_linear(name, "full_rows_ffn2") or
                name.startswith("blocks.") and any(name.endswith(".attn1." + suffix)
                                                    for suffix in ("to_q", "to_k", "to_v")))
    return variant == "full_rows_linear" or variant in TILE_VARIANTS


def tile_intersections(start, end, tile):
    """Owned row intersections with a grid anchored at global sequence row 0."""
    if type(tile) is not int or tile <= 0 or not 0 <= start < end:
        raise ValueError("invalid globally aligned tile interval")
    return [(base, max(start, base), min(end, base + tile))
            for base in range(start // tile * tile, end, tile)]


def tiled_linear(torch, value, forward, *, global_start, tile):
    """Run fixed-M GEMMs; each shard keeps its global position inside each tile."""
    pieces = []
    for base, first, last in tile_intersections(global_start, global_start + value.shape[1], tile):
        padded = value.new_zeros((value.shape[0], tile, value.shape[2]))
        padded[:, first - base:last - base, :] = value[:, first - global_start:last - global_start, :]
        output = forward(padded)
        pieces.append(output[:, first - base:last - base, :].clone(memory_format=torch.contiguous_format))
    return torch.cat(pieces, dim=1).contiguous()


def arithmetic_flags(torch):
    matmul = torch.backends.cuda.matmul
    return {name: getattr(matmul, name, None) for name in (
        "allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_fp16_accumulation")}


@contextlib.contextmanager
def no_reduced_precision(backend):
    """Candidate-only arithmetic control; never modifies the SP1 reference."""
    torch = backend.worker.torch
    matmul = torch.backends.cuda.matmul
    before = arithmetic_flags(torch)
    names = ("allow_fp16_reduced_precision_reduction", "allow_fp16_accumulation")
    if any(before[name] is None for name in names):
        raise RuntimeError("pinned torch FP16 arithmetic flags are unavailable")
    audit = dict(variant="no_reduced_precision", diagnostic_only=True,
                 attention_path="cache_dit", reference_flags=before,
                 performance_evidence=False, deployment_qualification=False)
    try:
        for name in names:
            setattr(matmul, name, False)
        audit["candidate_flags"] = arithmetic_flags(torch)
        yield audit
    finally:
        for name in names:
            setattr(matmul, name, before[name])
        audit["restored_flags"] = arithmetic_flags(torch)


@contextlib.contextmanager
def full_rows_linear(backend, variant="full_rows_linear"):
    """Restore full token-row GEMM shapes while retaining native SP attention.

    Zero rows are a diagnostic control, not replicated tokens or an SP speedup.
    Both input placement and output cropping follow the current rank layout.
    """
    torch = backend.worker.torch
    model = backend.worker.pipe.transformer
    hooks, pending, observations = [], {}, {}
    geometry = {}
    if variant not in ("full_rows_linear",) + SELECTIVE_VARIANTS:
        raise ValueError("unknown full-row selection")
    audit = dict(variant=variant, diagnostic_only=True, attention_path="cache_dit",
                 reference_flags=arithmetic_flags(torch), candidate_flags=arithmetic_flags(torch),
                 performance_evidence=False, deployment_qualification=False,
                 modules=[], patch_calls=0, applied_calls=0, skipped_calls=0, shapes=[],
                 hooks_removed=False)

    def before_patch(module, args, kwargs):
        geometry.clear()

    def patch_shape(module, args, output):
        backend._require_active()
        layout = backend.handle.current
        if layout.degree not in (1, 2) or output.ndim != 5:
            raise RuntimeError("full_rows_linear requires a 5D patch output and SP1/SP2")
        sequence = output.shape[2] * output.shape[3] * output.shape[4]
        if sequence % layout.degree:
            raise RuntimeError("patch token sequence does not divide SP degree")
        geometry.update(batch=output.shape[0], global_sequence=sequence,
                        local_sequence=sequence // layout.degree, ranks=tuple(layout.ranks),
                        epoch=layout.epoch)
        audit["patch_calls"] += 1

    def observe(name, value, layout, decision):
        shape = tuple(value.shape)
        key = (name, shape, layout.degree, geometry.get("global_sequence"), decision)
        if key not in observations:
            row = dict(module=name, input_shape=list(shape), input_stride=list(value.stride()),
                       dtype=str(value.dtype), device=str(value.device), degree=layout.degree,
                       global_sequence=geometry.get("global_sequence"), action=decision, calls=0)
            observations[key] = row
            audit["shapes"].append(row)
        observations[key]["calls"] += 1
        return observations[key]

    def before_linear(name, module, args, kwargs):
        backend._require_active()
        value = args[0] if args else kwargs.get("input")
        if value is None or not hasattr(value, "shape"):
            raise RuntimeError("Linear diagnostic input is missing")
        layout = backend.handle.current
        if layout.degree == 1:
            decision = "unchanged_sp1"
        elif not geometry or geometry["ranks"] != tuple(layout.ranks) or geometry["epoch"] != layout.epoch:
            raise RuntimeError("Linear diagnostic lacks current full patch geometry")
        elif value.ndim != 3:
            decision = "skip_non_token_shape"
        elif value.shape[1] != geometry["local_sequence"]:
            decision = "skip_nonlocal_sequence"
        else:
            decision = "pad_full_rows"
        observation = observe(name, value, layout, decision)
        if decision != "pad_full_rows":
            pending[name].append(None)
            audit["skipped_calls"] += 1
            return
        if (value.shape[0] != geometry["batch"] or value.shape[-1] != module.in_features
                or layout.degree != 2):
            raise RuntimeError("Linear token shape differs from active patch geometry")
        local_rank = tuple(layout.ranks).index(backend.handle.rank)
        start = local_rank * geometry["local_sequence"]
        end = start + geometry["local_sequence"]
        full = value.new_zeros((value.shape[0], geometry["global_sequence"], value.shape[-1]))
        full[:, start:end, :] = value
        pending[name].append((start, end, value.shape[0], geometry["global_sequence"], observation))
        observation.update(local_rank=local_rank, row_start=start, row_end=end,
                           padded_shape=list(full.shape), padded_stride=list(full.stride()))
        audit["applied_calls"] += 1
        if args:
            return (full,) + args[1:], kwargs
        return args, dict(kwargs, input=full)

    def after_linear(name, module, args, output):
        # always_call removes a pending call even when the original GEMM fails.
        if not pending[name]:
            return
        item = pending[name].pop()
        if item is None or output is None:
            return
        start, end, batch, sequence, observation = item
        if output.ndim != 3 or tuple(output.shape) != (batch, sequence, module.out_features):
            raise RuntimeError("expanded Linear output has unexpected shape")
        local = output[:, start:end, :].clone(memory_format=torch.contiguous_format)
        observation.update(output_shape=list(local.shape), output_stride=list(local.stride()),
                           output_storage_offset=local.storage_offset())
        return local

    try:
        hooks.append(model.patch_embedding.register_forward_pre_hook(before_patch, with_kwargs=True))
        # Observe the original full Conv3d output before the existing SP split.
        hooks.append(model.patch_embedding.register_forward_hook(patch_shape, prepend=True))
        selected = []
        for index, block in enumerate(model.blocks):
            selected.extend((f"blocks.{index}.{name}", module) for name, module in block.named_modules()
                            if isinstance(module, torch.nn.Linear))
        selected.extend(("proj_out" + ("." + name if name else ""), module)
                        for name, module in model.proj_out.named_modules()
                        if isinstance(module, torch.nn.Linear))
        selected = [(name, module) for name, module in selected if selected_linear(name, variant)]
        if not selected or len({id(module) for _, module in selected}) != len(selected):
            raise RuntimeError("expected distinct token Linear modules")
        for name, module in selected:
            pending[name] = []
            audit["modules"].append(name)
            hooks.append(module.register_forward_pre_hook(
                lambda m, a, k, name=name: before_linear(name, m, a, k), with_kwargs=True))
            # proj_out already has an SP gather hook: crop BEFORE it gathers.
            hooks.append(module.register_forward_hook(
                lambda m, a, output, name=name: after_linear(name, m, a, output),
                prepend=True, always_call=True))
        yield audit
    finally:
        for hook in reversed(hooks):
            hook.remove()
        pending.clear()
        audit["hooks_removed"] = True


@contextlib.contextmanager
def global_tile_linear(backend, variant):
    """The same fixed row grid in SP1 and SP2, scoped to the experiment."""
    if variant not in TILE_VARIANTS:
        raise ValueError("unknown global tile variant")
    torch, model = backend.worker.torch, backend.worker.pipe.transformer
    tile = int(variant.rsplit("_", 1)[1])
    geometry, observations, hooks, originals = {}, {}, [], []
    audit = dict(variant=variant, diagnostic_only=True, attention_path="cache_dit",
                 tile_rows=tile, row_alignment="global_sequence_origin",
                 reference_flags=arithmetic_flags(torch), candidate_flags=arithmetic_flags(torch),
                 performance_evidence=False, deployment_qualification=False,
                 modules=[], patch_calls=0, applied_calls=0, skipped_calls=0, shapes=[],
                 hooks_removed=False, forwards_restored=False)

    def patch_shape(module, args, output):
        backend._require_active()
        layout = backend.handle.current
        if output.ndim != 5 or layout.degree not in (1, 2):
            raise RuntimeError("global tile requires original 5D patch output and SP1/SP2")
        sequence = output.shape[2] * output.shape[3] * output.shape[4]
        if sequence % layout.degree:
            raise RuntimeError("global sequence must divide degree")
        geometry.update(batch=output.shape[0], global_sequence=sequence,
                        local_sequence=sequence // layout.degree, ranks=tuple(layout.ranks), epoch=layout.epoch)
        audit["patch_calls"] += 1

    def wrapped(name, module, original, *args, **kwargs):
        backend._require_active()
        value = args[0] if args else kwargs.get("input")
        layout = backend.handle.current
        if not geometry or geometry["ranks"] != tuple(layout.ranks) or geometry["epoch"] != layout.epoch:
            raise RuntimeError("global tile lacks current patch geometry")
        if value is None or not hasattr(value, "shape"):
            raise RuntimeError("global tile Linear input is missing")
        action = ("skip_non_token_shape" if value.ndim != 3 else
                  "skip_nonlocal_sequence" if value.shape[1] != geometry["local_sequence"] else "global_tile")
        key = (name, tuple(value.shape), layout.degree, geometry["global_sequence"], action)
        if key not in observations:
            observations[key] = dict(module=name, input_shape=list(value.shape), input_stride=list(value.stride()),
                dtype=str(value.dtype), device=str(value.device), degree=layout.degree,
                global_sequence=geometry["global_sequence"], action=action, calls=0)
            audit["shapes"].append(observations[key])
        row = observations[key]
        row["calls"] += 1
        if action != "global_tile":
            audit["skipped_calls"] += 1
            return original(*args, **kwargs)
        if value.shape[0] != geometry["batch"] or value.shape[2] != module.in_features:
            raise RuntimeError("global tile token shape differs from patch geometry")
        start = tuple(layout.ranks).index(backend.handle.rank) * geometry["local_sequence"]
        end = start + geometry["local_sequence"]
        def forward(padded):
            return original(*((padded,) + args[1:]), **kwargs) if args else original(**dict(kwargs, input=padded))
        local = tiled_linear(torch, value, forward, global_start=start, tile=tile)
        row.update(row_start=start, row_end=end, tile_rows=tile,
                   tile_intersections=[list(item) for item in tile_intersections(start, end, tile)],
                   output_shape=list(local.shape), output_stride=list(local.stride()),
                   output_storage_offset=local.storage_offset())
        audit["applied_calls"] += 1
        return local

    try:
        hooks.append(model.patch_embedding.register_forward_pre_hook(lambda m, a: geometry.clear()))
        hooks.append(model.patch_embedding.register_forward_hook(patch_shape, prepend=True))
        selected = [(f"blocks.{index}.{name}", module) for index, block in enumerate(model.blocks)
                    for name, module in block.named_modules() if isinstance(module, torch.nn.Linear)]
        selected.extend(("proj_out" + ("." + name if name else ""), module)
                        for name, module in model.proj_out.named_modules() if isinstance(module, torch.nn.Linear))
        if not selected or len({id(module) for _, module in selected}) != len(selected):
            raise RuntimeError("expected distinct token Linear modules")
        for name, module in selected:
            original = module.forward
            originals.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = lambda *a, name=name, module=module, original=original, **k: wrapped(
                name, module, original, *a, **k)
            audit["modules"].append(name)
        yield audit
    finally:
        for module, existed, original in reversed(originals):
            if existed:
                module.forward = original
            else:
                del module.forward
        for hook in reversed(hooks):
            hook.remove()
        audit.update(hooks_removed=True, forwards_restored=True)


@contextlib.contextmanager
def sp1_tile_variant(worker, variant, rank=0):
    """Apply the candidate recipe to ordinary SP1 without installing SP hooks."""
    backend = SimpleNamespace(worker=worker, _require_active=lambda: None,
        handle=SimpleNamespace(rank=rank, current=SimpleNamespace(degree=1, ranks=(rank,), epoch=0)))
    with global_tile_linear(backend, variant) as audit:
        yield audit


def pack_exchange(tensor,degree,*,inverse=False):
    batch,sequence,heads,dim=tensor.shape
    if inverse:
        if sequence%degree:
            raise ValueError("sequence length must divide SP degree")
        return tensor.reshape(batch,degree,sequence//degree,heads,dim).permute(1,0,2,3,4).contiguous()
    if heads%degree:
        raise ValueError("heads must divide SP degree")
    return tensor.reshape(batch,sequence,degree,heads//degree,dim).permute(2,0,1,3,4).contiguous()


def unpack_exchange(received,*,inverse=False):
    degree,batch,local_sequence,local_heads,dim=received.shape
    if inverse:
        return received.permute(1,2,0,3,4).reshape(batch,local_sequence,local_heads*degree,dim)
    return received.permute(1,0,2,3,4).reshape(batch,degree*local_sequence,local_heads,dim)


def _rotary(torch, tensor, rotary):
    if rotary is None:
        return tensor
    x1,x2=tensor.unflatten(-1,(-1,2)).unbind(-1)
    cos,sin=rotary[0][...,0::2],rotary[1][...,1::2]
    out=torch.empty_like(tensor)
    out[...,0::2]=x1*cos-x2*sin
    out[...,1::2]=x1*sin+x2*cos
    return out.type_as(tensor)


class Exchange:
    def __init__(self, torch, tensor, group, stream, *, inverse=False):
        self.torch,self.stream,self.inverse=torch,stream,inverse
        self.degree=torch.distributed.get_world_size(group)
        self.batch,self.sequence,self.heads,self.dim=tensor.shape
        packed=pack_exchange(tensor,self.degree,inverse=inverse)
        self.packed=packed
        self.received=torch.empty_like(packed)
        current=torch.cuda.current_stream(tensor.device)
        ready=torch.cuda.Event()
        ready.record(current)
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            self.work=torch.distributed.all_to_all_single(self.received,packed,group=group,async_op=True)
            # Enqueue the NCCL completion dependency on the communication stream.
            # A host async return alone does not establish safe output readiness.
            self.work.wait()
            self.done=torch.cuda.Event()
            self.done.record(stream)
            packed.record_stream(stream)
            self.received.record_stream(stream)

    def finish(self):
        torch=self.torch
        current=torch.cuda.current_stream(self.received.device)
        current.wait_event(self.done)
        self.received.record_stream(current)
        return unpack_exchange(self.received,inverse=self.inverse)


class WorkflowUlyssesProcessor:
    """Self-attention variants baseline, qkv_pipeline and heads_pipeline.

    A backend owns layout hooks and consensus. Cross attention delegates to the
    pinned processor because its K/V text sequence must remain replicated.
    """
    def __init__(self, backend, original, mode):
        if mode not in ("baseline","qkv_pipeline","heads_pipeline"):
            raise ValueError("unknown experiment attention variant")
        self.backend,self.original,self.mode=backend,original,mode
        self.stream=None

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,rotary_emb=None):
        backend=self.backend
        backend._require_active()
        if encoder_hidden_states is not None or backend.handle.current.degree==1:
            return self.original(attn,hidden_states,encoder_hidden_states,attention_mask,rotary_emb)
        if attention_mask is not None or attn.add_k_proj is not None:
            raise ValueError("workflow SP2 supports only unmasked Wan T2V self attention")
        import torch
        group=backend.groups.group(backend.handle.current.ranks)
        if self.stream is None:
            self.stream=torch.cuda.Stream(device=hidden_states.device)
        heads=attn.heads
        def projected(layer,norm=None):
            result=layer(hidden_states)
            result=norm(result) if norm is not None else result
            return result.unflatten(2,(heads,-1))
        def exchange(value,inverse=False):
            return Exchange(torch,value,group,self.stream,inverse=inverse)
        def attention(q,k,v):
            # Explicit Flash policy independent of process-global SDPA toggles.
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
                return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.0,is_causal=False).transpose(1,2)
        if self.mode=="qkv_pipeline":
            q=exchange(_rotary(torch,projected(attn.to_q,attn.norm_q),rotary_emb))
            k=exchange(_rotary(torch,projected(attn.to_k,attn.norm_k),rotary_emb))
            v=exchange(projected(attn.to_v))
            result=exchange(attention(q.finish(),k.finish(),v.finish()),inverse=True).finish()
        else:
            q=_rotary(torch,projected(attn.to_q,attn.norm_q),rotary_emb)
            k=_rotary(torch,projected(attn.to_k,attn.norm_k),rotary_emb)
            v=projected(attn.to_v)
            if self.mode=="baseline":
                q=exchange(q).finish()
                k=exchange(k).finish()
                v=exchange(v).finish()
                result=exchange(attention(q,k,v),inverse=True).finish()
            else:
                degree=backend.handle.current.degree
                groups=3
                local_heads=heads//degree
                if local_heads%groups:
                    raise ValueError("three head groups require divisible local heads")
                width=local_heads//groups
                def sliced(tensor,index):
                    return tensor.unflatten(2,(degree,local_heads))[:,:,:,index*width:(index+1)*width,:].flatten(2,3).contiguous()
                def launch(index):
                    return tuple(exchange(sliced(tensor,index)) for tensor in (q,k,v))
                pending=launch(0)
                outputs=[]
                for index in range(groups):
                    current=tuple(item.finish() for item in pending)
                    # Queue the next group's transport before current attention;
                    # the communicator and attention compute can now overlap.
                    next_pending=launch(index+1) if index+1<groups else None
                    outputs.append(exchange(attention(*current),inverse=True))
                    pending=next_pending
                chunks=[item.finish().unflatten(2,(degree,width)) for item in outputs]
                result=torch.stack(chunks,dim=3).flatten(2,4)
        result=result.flatten(2,3).type_as(hidden_states)
        return attn.to_out[1](attn.to_out[0](result))


@contextlib.contextmanager
def attention_variant(backend,mode):
    """Install experiment processors only for the lifetime of a test arm."""
    if mode == "cache_dit":
        yield dict(variant=mode, diagnostic_only=False, attention_path="cache_dit",
                   reference_flags=arithmetic_flags(backend.worker.torch),
                   candidate_flags=arithmetic_flags(backend.worker.torch))
        return
    if mode in DIAGNOSTIC_VARIANTS:
        context = (global_tile_linear(backend, mode) if mode in TILE_VARIANTS else
                   no_reduced_precision(backend) if mode == "no_reduced_precision" else
                   full_rows_linear(backend, mode))
        with context as audit:
            yield audit
        return
    modules=[]
    for module,_ in backend._processors:
        original=module.processor
        modules.append((module,original))
        module.set_processor(WorkflowUlyssesProcessor(backend,original,mode))
    try:
        yield dict(variant=mode, diagnostic_only=False, attention_path="workflow_"+mode,
                   reference_flags=arithmetic_flags(backend.worker.torch),
                   candidate_flags=arithmetic_flags(backend.worker.torch))
    finally:
        for module,original in modules:
            module.set_processor(original)
