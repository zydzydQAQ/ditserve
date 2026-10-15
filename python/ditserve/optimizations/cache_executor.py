"""Request-private, same-GPU continuations for the pinned paired_008 policy.

An interrupted/failed partial step is discarded, never restored through the
old whole-step global RNG/manager snapshot. No cache-dit or model monkeypatch.
"""

import dataclasses
import time

from ..research.cfg_split_prefix import CfgSplitPrefix
from .wan import BlockBackend


@dataclasses.dataclass
class CacheContinuation:
    base: object
    owner: object = dataclasses.field(repr=False)
    committed_step: int = 0
    phase: str = "prefix"
    branch: str = "cond"
    next_block: int = 0
    packets: dict = dataclasses.field(default_factory=dict)
    predictions: dict = dataclasses.field(default_factory=dict)
    probe: object = None
    reuse: object = None
    gates: list = dataclasses.field(default_factory=list)
    step_quanta: list = dataclasses.field(default_factory=list)
    cancelled: bool = False
    failed: bool = False
    closed: bool = False
    events: list = dataclasses.field(default_factory=list)


class PairedCacheExecutor:
    """At most K actual blocks, across CFG branches, one solver commit maximum."""

    def __init__(self, worker, backend=None, *, split=None):
        self.worker, self.model = worker, worker.pipe.transformer
        self.split = split or CfgSplitPrefix(worker)
        self.backend = backend or BlockBackend(worker)
        if self.backend.worker is not worker or self.backend.model is not self.model:
            raise ValueError("cache backend belongs to another worker/model")
        if self.split.worker is not worker:
            raise ValueError("split-prefix helper belongs to another worker")
        self._owner, self._states, self._advancing = object(), {}, False

    def attach(self, base):
        with self.worker._lock:
            self.worker._check(base, "gpu")
            self._validate(base)
            if id(base) in self._states:
                raise ValueError("request already has a cache continuation")
            if base.finished or base.step_index >= base.spec["steps"]:
                raise ValueError("request has no remaining solver steps")
            state = CacheContinuation(base, self._owner, committed_step=base.step_index)
            self._states[id(base)] = state
            return state

    def _validate(self, base):
        if (not base.cache_enabled or base.spec.get("cache") is not True or
                base.spec.get("threshold") != .08 or base.spec.get("max_continuous") != 1):
            raise ValueError("cache continuation requires unchanged paired_008")
        self.split.validate_state(base)
        self.worker._update_policy(base)

    def _check(self, state):
        if (state.owner is not self._owner or self._states.get(id(state.base)) is not state
                or state.failed or state.cancelled or state.closed):
            raise ValueError("inactive, failed, cancelled or foreign cache continuation")
        self.worker._check(state.base, "gpu")
        if state.base.step_index != state.committed_step:
            raise ValueError("base solver state changed outside cache executor")
        self._validate(state.base)

    def _packet(self, state):
        base, model = state.base, self.model
        hidden = base.latents.to(model.dtype)
        batch, _, frames, height, width = hidden.shape
        pt, ph, pw = model.config.patch_size
        rotary = model.rope(hidden)
        hidden = model.patch_embedding(hidden).flatten(2).transpose(1, 2)
        timestep = base.scheduler.timesteps[base.step_index].expand(batch)
        embedding = base.prompt_embeds if state.branch == "cond" else base.negative_prompt_embeds
        temb, projected, encoder, image = model.condition_embedder(
            timestep, embedding, None, timestep_seq_len=None)
        if image is not None or temb.ndim != 2:
            raise ValueError("unsupported image or expanded timestep")
        return dict(hidden=hidden, encoder=encoder, temb=temb,
                    projected=projected.unflatten(1, (6, -1)), rotary=rotary,
                    original_prefix=hidden,
                    shape=(batch, frames // pt, height // ph, width // pw, pt, ph, pw))

    def _prediction(self, packet):
        # Identical output arithmetic to CfgSplitPrefix._suffix.
        model, hidden, temb = self.model, packet["hidden"], packet["temb"]
        shift, scale = (model.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift, scale = shift.to(hidden.device), scale.to(hidden.device)
        hidden = (model.norm_out(hidden.float()) * (1 + scale) + shift).type_as(hidden)
        hidden = model.proj_out(hidden)
        batch, frames, height, width, pt, ph, pw = packet["shape"]
        hidden = hidden.reshape(batch, frames, height, width, pt, ph, pw, -1)
        return hidden.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _start_suffix(self, state):
        event = self.split._gate(state.packets[state.branch], state.branch)
        expected = state.probe[0 if state.branch == "cond" else 1]
        if event["native_reuse"] != expected["native_reuse"]:
            raise RuntimeError("split-prefix gate changed between isolated probe and commit")
        event.update(accepted_reuse=state.reuse, forced_full=not state.reuse)
        state.gates.append(event)
        packet, manager = state.packets[state.branch], self.worker.manager
        if state.reuse:
            manager.add_cached_step()
            packet["hidden"], packet["encoder"] = manager.apply_cache(
                packet["hidden"], packet["encoder"], prefix="blocks_Bn_residual",
                encoder_prefix="blocks_Bn_residual")
            state.next_block = 30
        else:
            manager.set_Fn_buffer(packet["residual"], prefix="blocks_Fn_residual")
            packet["original_middle"] = packet["hidden"]
            packet["original_encoder"] = packet["encoder"]
            state.next_block = 8
        state.phase = "suffix"

    def _finish_suffix(self, state):
        packet, manager = state.packets[state.branch], self.worker.manager
        if not state.reuse:
            packet["hidden"] = packet["hidden"].contiguous()
            residual = packet["hidden"] - packet.pop("original_middle")
            original_encoder = packet.pop("original_encoder")
            encoder_residual = None
            if packet["encoder"] is not None and original_encoder is not None:
                packet["encoder"] = packet["encoder"].contiguous()
                encoder_residual = packet["encoder"] - original_encoder
            manager.set_Bn_buffer(residual, prefix="blocks_Bn_residual")
            manager.set_Bn_encoder_buffer(encoder_residual, prefix="blocks_Bn_residual")
        state.predictions[state.branch] = self._prediction(packet)
        del state.packets[state.branch]
        if state.branch == "cond":
            state.branch, state.phase, state.next_block = "uncond", "commit_gate", 8
            return False
        base = state.base
        prediction = state.predictions["uncond"] + base.spec["guidance_scale"] * (
            state.predictions["cond"] - state.predictions["uncond"])
        base.latents = base.scheduler.step(prediction, base.scheduler.timesteps[base.step_index],
                                           base.latents, return_dict=False)[0]
        base.step_index += 1
        base.needs_anchor = False
        if not bool(self.worker.torch.isfinite(base.latents).all().item()):
            raise RuntimeError("nonfinite latent after paired cache solver commit")
        path = self.worker._path(base, state.committed_step)
        if path["path"] == "mixed":
            raise RuntimeError("paired continuation committed mixed cache branches")
        for context in base.contexts.values():
            if (context.executed_steps != base.step_index or
                    context.transformer_executed_steps != 2 * base.step_index):
                raise RuntimeError("cache counters diverged from solver")
        base.events.append(dict(kind="step", step_index=state.committed_step,
            completed_steps=base.step_index, **path,
            executor="paired_008_continuation", backend=self.backend.name,
            gates=list(state.gates), probe_gates=state.probe,
            fn_block_calls=dict(cond=8, uncond=8),
            mn_block_calls=dict(cond=0 if state.reuse else 22, uncond=0 if state.reuse else 22)))
        state.committed_step = base.step_index
        state.phase, state.branch, state.next_block = "prefix", "cond", 0
        state.packets.clear()
        state.predictions.clear()
        state.probe = state.reuse = None
        state.gates.clear()
        state.step_quanta.clear()
        return True

    def advance(self, state, blocks=4, *, observe=False, check_budget=lambda: None):
        if type(blocks) is not int or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        worker, torch = self.worker, self.worker.torch
        with worker._lock, torch.inference_mode(), worker._dit_attention_context():
            self._check(state)
            if self._advancing:
                raise RuntimeError("overlapping/reentrant cache execution")
            if state.base.step_index >= state.base.spec["steps"]:
                raise ValueError("request has no remaining solver steps")
            self.split._context_guard.validate()
            check_budget()
            initial = (state.committed_step, state.phase, state.branch, state.next_block)
            completed, executed, events = False, 0, []
            worker._sync()
            started = time.perf_counter()
            self._advancing = True
            try:
                with worker._activate(state.base):
                    worker.manager.set_context(self.split.blocks.cache_context)
                    while not completed:
                        check_budget()
                        if state.phase == "probe":
                            state.probe = self.split._probe(state.base, state.packets)
                            state.reuse = all(e["native_reuse"] for e in state.probe)
                            state.phase, state.branch = "commit_gate", "cond"
                        if state.phase == "commit_gate":
                            self._start_suffix(state)
                        if state.phase == "suffix" and state.next_block == 30:
                            completed = self._finish_suffix(state)
                            continue
                        if executed >= blocks:
                            break
                        if state.phase == "prefix" and state.branch not in state.packets:
                            state.packets[state.branch] = self._packet(state)
                        packet = state.packets[state.branch]
                        stop = 8 if state.phase == "prefix" else 30
                        width = 1 if observe else self.backend.width(state.next_block, blocks-executed, stop)
                        if observe:
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            begin.record()
                        packet["hidden"] = self.backend.run(state.next_block, width, packet["hidden"],
                            packet["encoder"], packet["projected"], packet["rotary"])
                        if observe:
                            end.record()
                            events.append((state.branch, state.next_block, begin, end))
                        executed += width
                        state.next_block += width
                        if state.phase == "prefix" and state.next_block == 8:
                            packet["residual"] = self.split.blocks._get_Fn_residual(
                                packet.pop("original_prefix"), packet["hidden"])
                            if state.branch == "cond":
                                state.branch, state.next_block = "uncond", 0
                            else:
                                state.phase = "probe"
                worker._sync()
                row = dict(kind="cache_block_quantum", step_index=initial[0],
                    phase=initial[1], branch=initial[2], start_block=initial[3],
                    next_phase=state.phase, next_branch=state.branch, next_block=state.next_block,
                    blocks=executed, step_completed=completed, backend=self.backend.name,
                    ms=(time.perf_counter()-started)*1000,
                    extra_state_bytes=self.extra_state_bytes(state),
                    block_cuda_ms=[dict(branch=b, block=i, ms=float(a.elapsed_time(z)))
                                   for b, i, a, z in events],
                    boundary_semantics="at_most_K_actual_blocks_cross_CFG_then_stop_at_solver_commit",
                    failure_semantics="discard_owned_request_no_global_rng_rollback")
                state.events.append(row)
                return row
            except BaseException:
                state.failed = True
                self._discard_packets(state)
                raise
            finally:
                self._advancing = False

    def extra_state_bytes(self, state):
        torch, seen = self.worker.torch, set()
        def visit(value, count):
            if torch.is_tensor(value):
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr(), storage.nbytes())
                new = key not in seen
                seen.add(key)
                return int(storage.nbytes()) if count and new else 0
            if isinstance(value, dict):
                return sum(visit(v, count) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(visit(v, count) for v in value)
            return 0
        visit(self.worker._payload(state.base), False)
        return visit((state.packets, state.predictions), True)

    @staticmethod
    def _discard_packets(state):
        state.packets.clear()
        state.predictions.clear()
        state.probe = state.reuse = None
        state.gates.clear()

    def cancel(self, state):
        with self.worker._lock:
            self._check(state)
            state.cancelled = True
            self._discard_packets(state)

    def close(self, state):
        with self.worker._lock:
            if state.owner is not self._owner or self._states.get(id(state.base)) is not state:
                raise ValueError("foreign or already closed continuation")
            self.worker._sync()
            self._discard_packets(state)
            state.closed = True
            del self._states[id(state.base)]

    def quantum(self, state, policy):
        if policy == "step":
            return 60
        if policy == "block-4":
            return 4
        raise ValueError("paired continuation supports only step and block-4")

