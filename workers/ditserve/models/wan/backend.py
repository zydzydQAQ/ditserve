"""JSON-facing Wan adapter. Scheduling and experiment gates live in C++."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import time

from ..base import Backend, artifact, file_hash, positive_int, progress


class WanBackend(Backend):
    operations = Backend.operations + ("reference", "encode_start", "encode_poll", "encode_cancel",
                                       "distributed", "transfer", "block_export", "block_import")

    def __init__(self, config):
        super().__init__(config)
        from .runtime import WanStepWorker
        from .prompt_cache import cache_options
        self.options = config.get("options", {})
        cache_config = cache_options(self.options)
        self.runtime = WanStepWorker(device=config.get("device", "cuda:0"),
            model_path=config.get("model_path"), attention=config.get("attention", "math"), **cache_config)
        from .linear import validate_linear_policy
        self.sp_linear_policy = validate_linear_policy(self.options.get("sp_linear_policy", "native"))
        self.execution = self.options.get("execution", "step")
        if self.execution not in ("step", "block"):
            raise ValueError("Wan execution must be step or block")
        self.executor = self.parallel = self.elastic = None
        self.continuations = {}
        self.encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prompt-encoder")
        self.tickets, self.ticket_counter = {}, 0
        self.ticket_limit = self.options.get("text_prefetch_depth")
        if self.ticket_limit is not None and (type(self.ticket_limit) is not int or not 0 <= self.ticket_limit <= 2):
            raise ValueError("text_prefetch_depth must be 0, 1 or 2")
        self.ticket_peak = 0
        self.owns_world = False

    def _progress(self, state):
        continuation = next((value for value in self.continuations.values() if value.base is state), None)
        result = dict(progress(state.step_index, state.spec["steps"]),
                      transfer_ready=self._transfer_ready(state, continuation))
        if continuation is not None:
            result.update(branch=continuation.branch, next_block=continuation.next_block,
                          block=continuation.next_block + (len(self.runtime.pipe.transformer.blocks)
                                                         if continuation.branch == "uncond" else 0))
        elastic = getattr(self, "elastic", None)
        if elastic is not None and state.request_id in elastic.records:
            result.update(elastic.describe(state.request_id))
        return result

    @staticmethod
    def _transfer_ready(state, continuation):
        if continuation is None:
            return True
        return (continuation.committed_step == state.step_index
                and continuation.packet is None and continuation.conditional is None
                and continuation.branch == "cond" and continuation.next_block == 0)

    def encode_start(self, request):
        spec = self._spec(request)
        self._reap_cancelled_tickets()
        if self.ticket_limit is not None and len(self.tickets) >= self.ticket_limit:
            raise ValueError("encoder ticket capacity exhausted")
        if any(existing == spec for existing, unused in self.tickets.values()):
            raise ValueError("request already has an encoder ticket")
        self.ticket_counter += 1
        ticket = "encode-%d" % self.ticket_counter
        self.tickets[ticket] = (spec, self.encoder.submit(self.runtime.encode_prompt, spec))
        self.ticket_peak = max(self.ticket_peak, len(self.tickets))
        return dict(ticket=ticket, ready=False, **self._ticket_stats())

    def _reap_cancelled_tickets(self):
        cancelled = getattr(self, "cancelled_tickets", set())
        for ticket in list(cancelled):
            if ticket not in self.tickets or self.tickets[ticket][1].done():
                self.tickets.pop(ticket, None)
                cancelled.discard(ticket)
        self.cancelled_tickets = cancelled

    def _ticket_stats(self):
        return dict(outstanding_tickets=len(self.tickets), ticket_capacity=self.ticket_limit,
                    peak_outstanding_tickets=self.ticket_peak)

    def encode_cancel(self, request):
        ticket = request["ticket"]
        self._reap_cancelled_tickets()
        if ticket not in self.tickets:
            return dict(ticket=ticket, cancelled=True, already_released=True, **self._ticket_stats())
        future = self.tickets[ticket][1]
        # A running CPU encoder continues to occupy its slot until completion.
        # Removing it now would permit an unbounded hidden executor queue.
        future.cancel()
        self.cancelled_tickets.add(ticket)
        self._reap_cancelled_tickets()
        return dict(ticket=ticket, cancelled=True, **self._ticket_stats())

    def encode_poll(self, request):
        ticket = request["ticket"]
        self._reap_cancelled_tickets()
        if ticket not in self.tickets:
            raise ValueError("unknown encode ticket")
        if ticket in self.cancelled_tickets:
            return dict(ticket=ticket, ready=False, cancelled=True, **self._ticket_stats())
        spec, future = self.tickets[ticket]
        if not future.done():
            return dict(ticket=ticket, ready=False, **self._ticket_stats())
        try:
            payload = future.result()
        except BaseException:
            del self.tickets[ticket]
            raise
        receipt = getattr(payload, "negative_prompt_cache", None)
        return dict(ticket=ticket, ready=True, model_ms=payload.cpu_prompt_encode_ms,
                    **({"negative_prompt_cache": deepcopy(receipt)} if receipt is not None else {}),
                    **self._ticket_stats())

    def _spec(self, request):
        from .runtime import normalize_spec
        value = dict(self.config.get("spec", {}), **request.get("spec", {}))
        value.setdefault("request_id", request.get("state_id"))
        return normalize_spec(value)

    def prepare(self, request):
        key = self.new_id(request)
        spec = self._spec(request)
        if self.elastic is not None and spec["request_id"] != key:
            raise ValueError("elastic state ID must equal request_id")
        ticket = request.get("encode_ticket")
        if ticket:
            if ticket not in self.tickets:
                raise ValueError("unknown encode ticket")
            encoded_spec, future = self.tickets[ticket]
            if not future.done():
                raise ValueError("encode ticket is not ready")
            if encoded_spec != spec:
                raise ValueError("encode ticket belongs to a different spec")
            if ticket in getattr(self, "cancelled_tickets", set()):
                raise ValueError("encode ticket was cancelled")
            try:
                state = self.runtime.prepare_encoded(spec, future.result())
            finally:
                del self.tickets[ticket]
        else:
            state = self.runtime.prepare(spec)
        try:
            if self.execution == "block":
                if self.executor is None:
                    from .block import WanBlockExecutor
                    self.executor = WanBlockExecutor(self.runtime)
                self.continuations[key] = self.executor.attach(state)
            self.states[key] = state
            if self.elastic is not None:
                self.elastic.register(key)
        except BaseException:
            if self.elastic is not None:
                self.elastic.records.pop(key, None)
            self.states.pop(key, None)
            if key in self.continuations:
                self.executor.close(self.continuations.pop(key))
            self.runtime.release(state)
            raise
        timing = next((event for event in reversed(state.events) if event.get("kind") == "prepare"), {})
        return dict(state_id=key, progress=self._progress(state), model_ms=state.prepare_ms,
                    **{name: deepcopy(timing[name]) for name in (
                        "cpu_prompt_encode_ms", "gpu_prepare_ms", "negative_prompt_cache") if name in timing})

    def advance(self, request):
        elastic = getattr(self, "elastic", None)
        execution = {}
        if elastic is not None:
            record, sequence, cached = elastic.before_advance(request)
            if cached is not None:
                return cached
            execution = dict(layout=record.handle, quantum_seq=sequence)
        state = self.state(request)
        key = request["state_id"]
        records = []
        if key in self.continuations:
            blocks = (2 * len(self.runtime.pipe.transformer.blocks) if request.get("quantum") == "step"
                      else positive_int(request.get("blocks", request.get("steps", 1))))
            try:
                records.append(self.executor.advance(self.continuations[key],
                    blocks=blocks,
                    observe=bool(request.get("observe", False)), **execution))
            except BaseException:
                if elastic is not None:
                    elastic.fail_world()
                raise
        else:
            for unused in range(min(positive_int(request.get("steps", 1)), state.spec["steps"] - state.step_index)):
                records.append((self.parallel or self.runtime).step(state))
        if not records:
            raise ValueError("request has no remaining work")
        result = dict(state_id=key, progress=self._progress(state), events=records,
                      model_ms=sum(record.get("ms", 0.) for record in records))
        if elastic is not None:
            elastic.after_advance(request, result)
        return result

    def _boundary(self, request):
        state = self.state(request)
        if not self._transfer_ready(state, self.continuations.get(request["state_id"])):
            raise ValueError("this operation requires a completed solver-step boundary")
        return state

    def block_export(self, request):
        """Write a block-boundary continuation to a file; release the source by default."""
        key = request["state_id"]
        if self.elastic is not None:
            raise ValueError("elastic requests use collective transfer, not local block_export")
        state = self.state(request)
        if key not in self.continuations:
            raise ValueError("block export requires a block continuation")
        from .block import export_block_state
        payload, timing = export_block_state(self.executor, self.continuations[key])
        destination = Path(request["path"]).resolve()
        started = time.perf_counter()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            self.runtime.torch.save(payload, stream)
        timing["write_ms"] = (time.perf_counter() - started) * 1000
        layout = payload["manifest"]["layout"]
        if request.get("release", True):
            self.executor.close(self.continuations.pop(key))
            self.runtime.release(state)
            del self.states[key]
        digest, size = file_hash(destination), destination.stat().st_size
        # Registered as a raw artifact so the run finalizer prunes it after use.
        return dict(timing, path=str(destination), sha256=digest, bytes=size, layout=layout,
                    released=bool(request.get("release", True)),
                    artifacts=[dict(path=str(destination), sha256=digest, bytes=size, kind="raw")])

    def block_import(self, request):
        if self.elastic is not None:
            raise ValueError("elastic requests use collective transfer, not local block_import")
        if self.execution != "block":
            raise ValueError("block import requires a block-execution worker")
        key = self.new_id(request)
        path = Path(request["path"])
        # Per-tensor hashes inside the manifest always verify the payload; a
        # file hash is checked too when the caller knows it.
        observed = file_hash(path)
        if request.get("sha256") and observed != request["sha256"]:
            raise ValueError("block packet hash mismatch")
        started = time.perf_counter()
        payload = self.runtime.torch.load(path, map_location="cpu", weights_only=True)
        read_ms = (time.perf_counter() - started) * 1000
        if self.executor is None:
            from .block import WanBlockExecutor
            self.executor = WanBlockExecutor(self.runtime)
        from .block import import_block_state
        state, continuation, timing = import_block_state(self.executor, payload)
        self.states[key], self.continuations[key] = state, continuation
        return dict(timing, read_ms=read_ms, state_id=key, progress=self._progress(state), sha256=observed,
                    layout=payload["manifest"]["layout"])

    def distributed(self, request):
        action = request.get("action", "init")
        if action == "init":
            elastic = request.get("elastic", False)
            if type(elastic) is not bool:
                raise ValueError("elastic must be boolean")
            if (self.parallel is not None or self.states or self.runtime._states
                    or (self.execution != "step" and not (elastic and self.execution == "block"))):
                raise ValueError("SP initialization requires a fresh worker and an explicit supported execution mode")
            if self.runtime.attention != "flash":
                raise ValueError("pinned Wan SP requires config.attention=flash")
            torch = self.runtime.torch
            world_size = request.get("world_size", 2)
            timeout = request.get("timeout_seconds", 30)
            if type(timeout) not in (int, float) or not 0 < timeout <= 60:
                raise ValueError("group timeout must be in (0, 60] seconds")
            if elastic and world_size != 2:
                raise ValueError("experimental elastic SP supports two ranks only")
            if self.sp_linear_policy != "native" and world_size != 2:
                raise ValueError("SP Linear repair supports a two-rank world only")
            if world_size not in (2, 4, 8):
                raise ValueError("aligned-group SP requires 2, 4 or 8 ranks")
            from datetime import timedelta
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", init_method=request["init_method"],
                    rank=request["rank"], world_size=world_size, timeout=timedelta(seconds=timeout))
                self.owns_world = True
            elif torch.distributed.get_world_size() != world_size or torch.distributed.get_rank() != request["rank"]:
                raise ValueError("existing process group differs from requested world/rank")
            from .parallel import LayoutHandle, build_fixed_groups
            from .sp import PinnedWanUlyssesBackend
            groups = build_fixed_groups(backend="nccl", timeout_seconds=timeout)
            handle = LayoutHandle(torch.distributed.get_rank(), request.get("ranks"))
            self.parallel = PinnedWanUlyssesBackend(handle, groups,
                linear_policy=self.sp_linear_policy).attach(self.runtime)
            if elastic:
                from .block import WanParallelBlockExecutor
                from .elastic import ElasticRequests
                self.executor = WanParallelBlockExecutor(self.runtime, self.parallel)
                self.elastic = ElasticRequests(self)
            return dict(self.parallel.capabilities, initialized=True, elastic=elastic,
                        sp_linear_policy=self.parallel.linear_policy,
                        distributed_request_transport=elastic,
                        elastic_step_boundary_degrees=[1, 2] if elastic else [],
                        transport="host_file_v1" if elastic else None)
        if self.parallel is None:
            raise ValueError("distributed init must complete first on every rank")
        if action == "prepare":
            return self.prepare(request)
        if action == "step":
            return self.advance(request)
        if action == "transition":
            if self.elastic is None:
                raise ValueError("transition requires experimental elastic initialization")
            return self.elastic.transition(request)
        if action == "commit":
            if self.elastic is not None:
                raise ValueError("elastic layouts change only through ownership transfer")
            layout = self.parallel.handle.commit(request["ranks"], expected_epoch=request["expected_epoch"])
            return dict(ranks=list(layout.ranks), epoch=layout.epoch, gpu_validated=False)
        raise ValueError("unknown distributed action")

    def transfer(self, request):
        if self.elastic is None:
            raise ValueError("transfer requires experimental elastic initialization")
        return self.elastic.dispatch(request)

    def reference(self, request):
        if self.parallel:
            raise ValueError("SP references require an independent unsharded worker")
        spec = self._spec(request)
        result = self.runtime.reference(spec)
        output = result.pop("output")
        latents = result.pop("latents")
        return dict(result, output=self.persist(output, request), **self._evidence(request, spec, latents))

    def finish(self, request):
        if self.elastic is not None and self.elastic.records[request["state_id"]].status != "active":
            raise ValueError("cannot decode a frozen elastic request")
        state = self._boundary(request)
        if request.get("decode_frames") is not None:
            # Preemptible decode: repeated finish calls advance latent-frame chunks.
            output, info = self.runtime.decode_chunks(state, positive_int(request["decode_frames"], "decode_frames"))
            if output is None:
                return dict(state_id=request["state_id"], progress=self._progress(state), decode=info, model_ms=info["ms"])
            return dict(state_id=request["state_id"], output=self._persist_checked(output, request), decode=info,
                        progress=self._progress(state), model_ms=info["ms"],
                        **self._evidence(request, state.spec, state.latents))
        output = self.runtime.finish(state)
        return dict(state_id=request["state_id"], output=self._persist_checked(output, request),
                    progress=self._progress(state), model_ms=state.events[-1]["ms"],
                    **self._evidence(request, state.spec, state.latents))

    def _persist_checked(self, output, request):
        mode = request.get("output_mode", "file")
        result = self.persist(output, dict(request, output_mode="digest") if mode == "check" else request)
        if mode in ("check", "digest"):
            result["output_mode"] = mode
            result["sha256"] = result["array_sha256"]
        return result

    def _evidence(self, request, spec, latents):
        from .runtime import MODEL_ID, REVISION
        generation = dict(spec, model=self.config.get("model_id", MODEL_ID), revision=self.config.get("revision", REVISION),
            model_path=str(self.runtime.model_path), model_integrity="local_snapshot_not_rehashed",
            dtype="float16", vae_dtype="float32", text_encoder_dtype="float32", text_encoder_device="cpu",
            fps=self.config.get("fps", 16), attention=self.runtime.attention, approximate=False)
        result = dict(backend="wan", measured=True, generation=generation)
        if request.get("save_latents"):
            destination = request.get("latent_path") or str(Path(self.output_path(request)).with_suffix(".latent.npy"))
            result["latent"] = artifact(latents, destination)
        elif request.get("output_mode") in ("check", "digest"):
            result["latent"] = artifact(latents, mode="digest")
            result["latent"]["sha256"] = result["latent"]["array_sha256"]
        return result

    def release(self, request):
        elastic = getattr(self, "elastic", None)
        if elastic is not None:
            record = elastic.records.get(request["state_id"])
            if record is not None and record.status not in ("active", "failed"):
                raise ValueError("cannot release source before transfer is finalized")
        return self._release_state(request["state_id"])

    def _release_state(self, key):
        request = dict(state_id=key)
        state = self.state(request)
        if key in self.continuations:
            self.executor.close(self.continuations.pop(key))
        self.runtime.release(state)
        del self.states[key]
        elastic = getattr(self, "elastic", None)
        if elastic is not None:
            elastic.records.pop(key, None)
        return dict(released=key)

    def close(self):
        try:
            self.encoder.shutdown(wait=True, cancel_futures=True)
            self.tickets.clear()
            for key in list(self.states):
                self._release_state(key)
            if self.parallel:
                self.parallel.close()
            if self.owns_world:
                self.runtime.torch.distributed.destroy_process_group()
            self.runtime._sync()
        finally:
            clear = getattr(self.runtime, "clear_negative_prompt_cache", None)
            if clear is not None:
                clear()
