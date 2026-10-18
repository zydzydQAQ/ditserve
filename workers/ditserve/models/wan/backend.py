"""JSON-facing Wan adapter. Scheduling and experiment gates live in C++."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from ..base import Backend, artifact, file_hash, positive_int, progress


class WanBackend(Backend):
    operations = Backend.operations + ("reference", "encode_start", "encode_poll", "distributed",
                                       "block_export", "block_import")

    def __init__(self, config):
        super().__init__(config)
        from .runtime import WanStepWorker
        self.runtime = WanStepWorker(device=config.get("device", "cuda:0"),
            model_path=config.get("model_path"), attention=config.get("attention", "math"))
        self.options = config.get("options", {})
        self.execution = self.options.get("execution", "step")
        if self.execution not in ("step", "block"):
            raise ValueError("Wan execution must be step or block")
        self.executor = self.parallel = None
        self.continuations = {}
        self.encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prompt-encoder")
        self.tickets, self.ticket_counter = {}, 0
        self.owns_world = False

    def _progress(self, state):
        continuation = next((value for value in self.continuations.values() if value.base is state), None)
        return dict(progress(state.step_index, state.spec["steps"]), transfer_ready=self._transfer_ready(state, continuation))

    @staticmethod
    def _transfer_ready(state, continuation):
        if continuation is None:
            return True
        return (continuation.committed_step == state.step_index
                and continuation.packet is None and continuation.conditional is None
                and continuation.branch == "cond" and continuation.next_block == 0)

    def encode_start(self, request):
        spec = self._spec(request)
        self.ticket_counter += 1
        ticket = "encode-%d" % self.ticket_counter
        self.tickets[ticket] = (spec, self.encoder.submit(self.runtime.encode_prompt, spec))
        return dict(ticket=ticket, ready=False)

    def encode_poll(self, request):
        ticket = request["ticket"]
        if ticket not in self.tickets:
            raise ValueError("unknown encode ticket")
        spec, future = self.tickets[ticket]
        if not future.done():
            return dict(ticket=ticket, ready=False)
        payload = future.result()
        return dict(ticket=ticket, ready=True, model_ms=payload.cpu_prompt_encode_ms)

    def _spec(self, request):
        from .runtime import normalize_spec
        value = dict(self.config.get("spec", {}), **request.get("spec", {}))
        value.setdefault("request_id", request.get("state_id"))
        return normalize_spec(value)

    def prepare(self, request):
        key = self.new_id(request)
        spec = self._spec(request)
        ticket = request.get("encode_ticket")
        if ticket:
            if ticket not in self.tickets:
                raise ValueError("unknown encode ticket")
            encoded_spec, future = self.tickets[ticket]
            if not future.done():
                raise ValueError("encode ticket is not ready")
            if encoded_spec != spec:
                raise ValueError("encode ticket belongs to a different spec")
            state = self.runtime.prepare_encoded(spec, future.result())
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
        except BaseException:
            if key in self.continuations:
                self.executor.close(self.continuations.pop(key))
            self.runtime.release(state)
            raise
        timing = next((event for event in reversed(state.events) if event.get("kind") == "prepare"), {})
        return dict(state_id=key, progress=self._progress(state), model_ms=state.prepare_ms,
                    **{name: timing[name] for name in ("cpu_prompt_encode_ms", "gpu_prepare_ms") if name in timing})

    def advance(self, request):
        state = self.state(request)
        key = request["state_id"]
        records = []
        if key in self.continuations:
            blocks = (2 * len(self.runtime.pipe.transformer.blocks) if request.get("quantum") == "step"
                      else positive_int(request.get("blocks", request.get("steps", 1))))
            records.append(self.executor.advance(self.continuations[key],
                blocks=blocks,
                observe=bool(request.get("observe", False))))
        else:
            for unused in range(min(positive_int(request.get("steps", 1)), state.spec["steps"] - state.step_index)):
                records.append((self.parallel or self.runtime).step(state))
        if not records:
            raise ValueError("request has no remaining work")
        return dict(state_id=key, progress=self._progress(state), events=records,
                    model_ms=sum(record.get("ms", 0.) for record in records))

    def _boundary(self, request):
        state = self.state(request)
        if not self._transfer_ready(state, self.continuations.get(request["state_id"])):
            raise ValueError("this operation requires a completed solver-step boundary")
        return state

    def block_export(self, request):
        """Write a block-boundary continuation to a file; release the source by default."""
        key = request["state_id"]
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
            if self.parallel is not None or self.states or self.execution != "step":
                raise ValueError("SP initialization requires a fresh whole-step worker")
            if self.runtime.attention != "flash":
                raise ValueError("pinned Wan SP requires config.attention=flash")
            torch = self.runtime.torch
            world_size = request.get("world_size", 2)
            if world_size not in (2, 4, 8):
                raise ValueError("aligned-group SP requires 2, 4 or 8 ranks")
            from datetime import timedelta
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", init_method=request["init_method"],
                    rank=request["rank"], world_size=world_size, timeout=timedelta(seconds=request.get("timeout_seconds", 30)))
                self.owns_world = True
            from .parallel import LayoutHandle, build_fixed_groups
            from .sp import PinnedWanUlyssesBackend
            groups = build_fixed_groups(backend="nccl", timeout_seconds=request.get("timeout_seconds", 30))
            handle = LayoutHandle(torch.distributed.get_rank(), request.get("ranks"))
            self.parallel = PinnedWanUlyssesBackend(handle, groups).attach(self.runtime)
            return dict(self.parallel.capabilities, initialized=True)
        if self.parallel is None:
            raise ValueError("distributed init must complete first on every rank")
        if action == "prepare":
            return self.prepare(request)
        if action == "step":
            return self.advance(request)
        if action == "commit":
            layout = self.parallel.handle.commit(request["ranks"], expected_epoch=request["expected_epoch"])
            return dict(ranks=list(layout.ranks), epoch=layout.epoch, gpu_validated=False)
        raise ValueError("unknown distributed action")

    def reference(self, request):
        if self.parallel:
            raise ValueError("SP references require an independent unsharded worker")
        spec = self._spec(request)
        result = self.runtime.reference(spec)
        output = result.pop("output")
        latents = result.pop("latents")
        return dict(result, output=self.persist(output, request), **self._evidence(request, spec, latents))

    def finish(self, request):
        state = self._boundary(request)
        if request.get("decode_frames") is not None:
            # Preemptible decode: repeated finish calls advance latent-frame chunks.
            output, info = self.runtime.decode_chunks(state, positive_int(request["decode_frames"], "decode_frames"))
            if output is None:
                return dict(state_id=request["state_id"], progress=self._progress(state), decode=info, model_ms=info["ms"])
            return dict(state_id=request["state_id"], output=self.persist(output, request), decode=info,
                        progress=self._progress(state), model_ms=info["ms"],
                        **self._evidence(request, state.spec, state.latents))
        output = self.runtime.finish(state)
        return dict(state_id=request["state_id"], output=self.persist(output, request),
                    progress=self._progress(state), model_ms=state.events[-1]["ms"],
                    **self._evidence(request, state.spec, state.latents))

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
        return result

    def release(self, request):
        state = self.state(request)
        key = request["state_id"]
        if key in self.continuations:
            self.executor.close(self.continuations.pop(key))
        self.runtime.release(state)
        del self.states[key]
        return dict(released=key)

    def close(self):
        self.encoder.shutdown(wait=True, cancel_futures=True)
        self.tickets.clear()
        super().close()
        if self.parallel:
            self.parallel.close()
        if self.owns_world:
            self.runtime.torch.distributed.destroy_process_group()
        self.runtime._sync()
