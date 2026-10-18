"""Exact, cache-off Wan block continuations; imports never initialize CUDA.

Packets retain only the current forward's live tensors, not a history of
activations. The original RequestState and its checkpoint schema stay intact.
Arithmetic follows diffusers 0.36.0 WanTransformer3DModel (Apache-2.0).
"""

import dataclasses
import hashlib
import time

PACKET_SCHEMA = "ditserve.wan-block-packet.v1"
PACKET_TENSORS = ("hidden", "encoder", "temb", "projected")


@dataclasses.dataclass
class BlockState:
    base: object
    owner: object = dataclasses.field(repr=False)
    committed_step: int = 0
    branch: str = "cond"
    next_block: int = 0
    packet: object = None
    conditional: object = None
    cancelled: bool = False
    failed: bool = False
    closed: bool = False
    events: list = dataclasses.field(default_factory=list)


class WanBlockExecutor:
    """One shared resident Wan model, independent resumable request packets.

    Only this executor may advance an attached base state. Cancellation discards
    an unfinished step; callers must release rather than resume that request.
    Without ``parallel`` safe points are local to one GPU. With the attached
    Ulysses backend every rank of the request group runs the same quanta: the
    step opens before its first block, each quantum that stops inside the step
    is a collective safe point, and the solver commit closes the step.
    """

    def __init__(self, worker, parallel=None):
        self.worker, self.model, self.parallel = worker, worker.pipe.transformer, parallel
        self._validate_model()
        self._owner, self._states, self._advancing = object(), {}, False

    def _validate_model(self):
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock
        model = self.model
        if (type(model) is not WanTransformer3DModel or tuple(model.config.patch_size) != (1, 2, 2)
                or model.config.added_kv_proj_dim is not None or model.config.image_dim is not None
                or model.gradient_checkpointing or getattr(model, "_is_parallelized", False)
                or getattr(model, "peft_config", None) or getattr(model, "_hf_peft_config_loaded", False)):
            raise ValueError("block execution requires an unadapted Wan T2V transformer")
        if self.worker.torch.distributed.is_initialized() and self.parallel is None:
            raise ValueError("block execution in a process group requires the SP attachment's safe points")
        if any(type(block) is not WanTransformerBlock for block in model.blocks):
            raise ValueError("unknown Wan block implementation")
        # The SP attachment shards at the patch embedding and RoPE and gathers at the
        # output projection; any other hook would change the arithmetic unseen.
        sharding = {id(model.patch_embedding), id(model.rope), id(model.proj_out)} if self.parallel else set()
        for module in model.modules():
            if (module.training or getattr(module, "_hf_hook", None) is not None
                    or getattr(module, "_orig_mod", None) is not None
                    or getattr(module, "_compiled_call_impl", None) is not None
                    or (getattr(module, "_forward_hooks", {}) and id(module) not in sharding)
                    or getattr(module, "_forward_pre_hooks", {})):
                raise ValueError("training, module hooks, offload and compilation are unsupported")

    def attach(self, base):
        with self.worker._lock:
            self.worker._check(base, "gpu")
            if base.cache_enabled or base.spec["cache"] or base.contexts:
                raise ValueError("block execution is cache-off only")
            if id(base) in self._states:
                raise ValueError("request already has a block continuation")
            if base.finished or base.step_index >= base.spec["steps"]:
                raise ValueError("request has no remaining solver steps")
            state = BlockState(base, self._owner, committed_step=base.step_index)
            self._states[id(base)] = state
            return state

    def _check(self, state):
        if (state.owner is not self._owner or self._states.get(id(state.base)) is not state
                or state.cancelled or state.failed or state.closed):
            raise ValueError("inactive, failed, cancelled or foreign block continuation")
        self.worker._check(state.base, "gpu")
        if state.base.step_index != state.committed_step:
            raise ValueError("base solver state changed outside block executor")
        if state.base.cache_enabled or state.base.spec["cache"] or state.base.contexts:
            raise ValueError("cache policy changed during block execution")
        self.worker._update_policy(state.base)

    def _begin(self, state):
        base, model = state.base, self.model
        latent = base.latents.to(model.dtype)
        batch, _, frames, height, width = latent.shape
        pt, ph, pw = model.config.patch_size
        rotary = model.rope(latent)
        hidden = model.patch_embedding(latent).flatten(2).transpose(1, 2)
        timestep = base.scheduler.timesteps[base.step_index].expand(batch)
        embedding = base.prompt_embeds if state.branch == "cond" else base.negative_prompt_embeds
        temb, projected, encoder, image = model.condition_embedder(
            timestep, embedding, None, timestep_seq_len=None)
        if image is not None or temb.ndim != 2:
            raise ValueError("expanded timestep and image conditioning are unsupported")
        state.packet = dict(hidden=hidden, encoder=encoder, temb=temb,
                            projected=projected.unflatten(1, (6, -1)), rotary=rotary,
                            shape=(batch, frames // pt, height // ph, width // pw, pt, ph, pw))

    def _prediction(self, packet):
        model, hidden, temb = self.model, packet["hidden"], packet["temb"]
        shift, scale = (model.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift, scale = shift.to(hidden.device), scale.to(hidden.device)
        hidden = (model.norm_out(hidden.float()) * (1 + scale) + shift).type_as(hidden)
        hidden = model.proj_out(hidden)
        batch, frames, height, width, pt, ph, pw = packet["shape"]
        hidden = hidden.reshape(batch, frames, height, width, pt, ph, pw, -1)
        return hidden.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _finish_branch(self, state):
        base = state.base
        prediction = self._prediction(state.packet)
        state.packet, state.next_block = None, 0
        if state.branch == "cond" and base.spec["guidance_scale"] > 1:
            state.conditional, state.branch = prediction, "uncond"
            return False
        if state.branch == "uncond":
            prediction = prediction + base.spec["guidance_scale"] * (state.conditional - prediction)
        # The solver is touched exactly once, after both predictions exist.
        base.latents = base.scheduler.step(
            prediction, base.scheduler.timesteps[base.step_index], base.latents, return_dict=False)[0]
        base.step_index += 1
        base.needs_anchor = False
        state.committed_step = base.step_index
        state.branch, state.conditional = "cond", None
        if not bool(self.worker.torch.isfinite(base.latents).all().item()):
            raise RuntimeError("nonfinite latent after block solver commit")
        base.events.append(dict(kind="step", step_index=base.step_index - 1,
                                completed_steps=base.step_index, path="compute", executor="blocks"))
        return True

    def advance(self, state, blocks=1, *, observe=False, check_budget=lambda: None):
        """Execute at most blocks blocks, stopping no later than one solver commit.

        All launched GPU work completes before returning. Prelude, output
        projection and solver work belong to their adjacent quantum's latency.
        """
        if type(blocks) is not int or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        worker, torch = self.worker, self.worker.torch
        with worker._lock, torch.inference_mode(), worker._dit_attention_context():
            self._check(state)
            if self._advancing:
                raise RuntimeError("overlapping/reentrant GPU block execution")
            if state.base.step_index >= state.base.spec["steps"]:
                raise ValueError("request has no remaining solver steps")
            check_budget()
            step, initial_branch, initial_block = state.committed_step, state.branch, state.next_block
            events, completed, executed = [], False, 0
            worker._sync()
            started = time.perf_counter()
            self._advancing = True
            try:
                for _ in range(blocks):
                    check_budget()
                    if state.packet is None:
                        if self.parallel is not None and state.branch == "cond":
                            self.parallel.begin_step(state.base)
                        self._begin(state)
                    packet, index, branch = state.packet, state.next_block, state.branch
                    if observe:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                    packet["hidden"] = self.model.blocks[index](packet["hidden"], packet["encoder"],
                                                                 packet["projected"], packet["rotary"])
                    if observe:
                        end.record()
                        events.append((branch, index, start, end))
                    state.next_block += 1
                    executed += 1
                    if state.next_block == len(self.model.blocks):
                        completed = self._finish_branch(state)
                        if completed:
                            break
                worker._sync()
                if self.parallel is not None:
                    if completed:
                        self.parallel.end_step()
                    else:
                        self.parallel.quantum_point(dict(request_id=state.base.request_id, step=state.committed_step,
                                                         branch=state.branch, next_block=state.next_block,
                                                         blocks=executed))
                wall_ms = (time.perf_counter() - started) * 1000
                row = dict(kind="block_quantum", step_index=step, branch=initial_branch,
                           start_block=initial_block, next_block=state.next_block, next_branch=state.branch,
                           blocks=executed, step_completed=completed, ms=wall_ms,
                           block_cuda_ms=[dict(branch=b, block=i, ms=float(a.elapsed_time(z)))
                                          for b, i, a, z in events],
                           extra_state_bytes=self.extra_state_bytes(state))
                state.events.append(row)
                return row
            except BaseException:
                state.failed = True
                if self.parallel is not None:
                    self.parallel.handle.fail()
                raise
            finally:
                self._advancing = False

    def extra_state_bytes(self, state):
        """Unique storage retained by packet/prediction, excluding the base state."""
        torch, seen = self.worker.torch, set()
        def visit(value, count):
            if torch.is_tensor(value):
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr(), storage.nbytes())
                new = key not in seen
                seen.add(key)
                return storage.nbytes() if count and new else 0
            if isinstance(value, dict):
                return sum(visit(v, count) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(visit(v, count) for v in value)
            return 0
        visit(self.worker._payload(state.base), False)
        return int(visit((state.packet, state.conditional), True))

    def native_step(self, state, *, check_budget=lambda: None):
        """The unmodified worker baseline, admitted only at a solver boundary."""
        with self.worker._lock:
            self._check(state)
            if self._advancing or state.packet is not None or state.branch != "cond" or state.next_block:
                raise ValueError("native worker requires an idle solver boundary")
            check_budget()
            self._advancing = True
            try:
                event = (self.parallel or self.worker).step(state.base)
                state.committed_step = state.base.step_index
                record = dict(event, kind="native_step", step_completed=True, extra_state_bytes=0,
                              block_cuda_ms=[], blocks=len(self.model.blocks) *
                              (2 if state.base.spec["guidance_scale"] > 1 else 1))
                state.events.append(record)
                return record
            except BaseException:
                state.failed = True
                raise
            finally:
                self._advancing = False

    def cancel(self, state):
        with self.worker._lock:
            self._check(state)
            state.cancelled = True
            state.packet = state.conditional = None

    def close(self, state):
        with self.worker._lock:
            if state.owner is not self._owner or self._states.get(id(state.base)) is not state:
                raise ValueError("foreign or already closed continuation")
            self.worker._sync()
            state.packet = state.conditional = None
            state.closed = True
            del self._states[id(state.base)]

    def quantum(self, state, policy):
        if policy == "step":
            return len(self.model.blocks) * (2 if state.base.spec["guidance_scale"] > 1 else 1)
        if policy == "cfg":
            return len(self.model.blocks) - state.next_block
        if policy.startswith("block-"):
            return int(policy.split("-", 1)[1])
        raise ValueError("unknown preemption policy: " + policy)


def _digest(torch, value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def export_block_state(executor, state):
    """CPU copy of a continuation at any block boundary inside a solver step.

    The base solver state is untouched inside a step, so the step-boundary
    wire codec carries it unchanged. The packet adds the running hidden state,
    conditioning, RoPE and, in the unconditional branch, the conditional
    prediction. clone() keeps dense strides, so the resumed kernels see the
    same layouts; the caller releases the source after a successful write.
    """
    from .parallel import pack_cache_off_request
    worker, torch = executor.worker, executor.worker.torch
    with worker._lock, torch.inference_mode():
        executor._check(state)
        if executor._advancing:
            raise RuntimeError("cannot export during block execution")
        # Inside an SP step the packet holds this rank's token shard; only the
        # replicated step-boundary state can move to another group.
        if executor.parallel is not None and (state.packet is not None or state.conditional is not None
                                              or state.branch != "cond" or state.next_block):
            raise ValueError("SP block export is limited to solver-step boundaries")
        worker._sync()
        started = time.perf_counter()
        base, tensors = pack_cache_off_request(worker, state.base)
        layout = dict(committed_step=state.committed_step, branch=state.branch, next_block=state.next_block,
                      packet=state.packet is not None, conditional=state.conditional is not None)
        extra = {}
        if state.packet is not None:
            for name in PACKET_TENSORS:
                extra["packet." + name] = state.packet[name]
            for index, value in enumerate(state.packet["rotary"]):
                extra["packet.rotary.%d" % index] = value
            layout["shape"] = list(state.packet["shape"])
            layout["rotary"] = len(state.packet["rotary"])
        if state.conditional is not None:
            extra["conditional"] = state.conditional
        packet_bytes = 0
        for name, value in extra.items():
            tensors[name] = value.detach().cpu().clone()
            packet_bytes += value.numel() * value.element_size()
        manifest = dict(schema=PACKET_SCHEMA, base=base, layout=layout,
                        packet_sha256={name: _digest(torch, tensors[name]) for name in extra},
                        strides={name: list(tensors[name].stride()) for name in extra})
        return dict(manifest=manifest, tensors=tensors), dict(
            export_ms=(time.perf_counter() - started) * 1000, packet_bytes=packet_bytes,
            transfer_bytes=sum(value.numel() * value.element_size() for value in tensors.values()))


def import_block_state(executor, payload):
    """Rebuild an exported block boundary on this worker; returns (base, continuation)."""
    from .parallel import unpack_cache_off_request
    worker, torch = executor.worker, executor.worker.torch
    manifest, tensors = payload["manifest"], payload["tensors"]
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ValueError("unknown block packet schema")
    names = set(manifest["packet_sha256"])
    base_names = set(manifest["base"]["tensor_hashes"])
    if set(tensors) != base_names | names:
        raise ValueError("block packet tensor set mismatch")
    for name in names:
        if _digest(torch, tensors[name]) != manifest["packet_sha256"][name] or list(tensors[name].stride()) != manifest["strides"][name]:
            raise ValueError("block packet tensor checksum or layout mismatch: " + name)
    layout = manifest["layout"]
    with worker._lock, torch.inference_mode():
        started = time.perf_counter()
        base = unpack_cache_off_request(worker, manifest["base"], {name: tensors[name] for name in base_names})
        if base.request_id in worker._states:
            raise ValueError("destination already owns this request")
        worker._states[base.request_id] = base
        try:
            if base.step_index != layout["committed_step"]:
                raise ValueError("block packet and base solver step disagree")
            state = executor.attach(base)
            device = base.latents.device
            state.branch, state.next_block = layout["branch"], layout["next_block"]
            if layout["packet"]:
                packet = {name: tensors["packet." + name].to(device) for name in PACKET_TENSORS}
                packet["rotary"] = tuple(tensors["packet.rotary.%d" % index].to(device) for index in range(layout["rotary"]))
                packet["shape"] = tuple(layout["shape"])
                state.packet = packet
            if layout["conditional"]:
                state.conditional = tensors["conditional"].to(device)
            worker._sync()
        except BaseException:
            worker.release(base)
            worker._states.pop(base.request_id, None)
            raise
        return base, state, dict(import_ms=(time.perf_counter() - started) * 1000)
