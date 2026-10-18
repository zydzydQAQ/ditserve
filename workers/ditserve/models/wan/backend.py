"""JSON-facing Wan adapter. Scheduling and experiment gates live in C++."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from ..base import Backend, artifact, file_hash, positive_int, progress


class WanBackend(Backend):
    operations = Backend.operations + ("snapshot", "restore", "inventory", "set_cache", "reference",
        "encode_start", "encode_poll", "decode_start", "decode_poll", "migration_release", "migration_abort", "local_roundtrip", "distributed",
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
        self.continuations, self.pending = {}, {}
        self.encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prompt-encoder")
        self.tickets, self.ticket_counter = {}, 0
        from .migration import MigrationEndpoint
        self.endpoint = MigrationEndpoint(self.runtime,
            config.get("device_id", config.get("gpu_uuid", str(self.runtime.device))))
        self.owns_world = False
        self.decoder, self.decode_ticket, self.decode_state, self.decode_result = None, None, None, None
        self.decode_request = None

    def decode_start(self, request):
        state = self._boundary(request)
        if self.decode_ticket is not None:
            raise ValueError("collect the previous decode before starting another")
        if self.parallel:
            raise ValueError("asynchronous VAE overlap is supported only for the SP1 worker")
        if self.decoder is None:
            from .async_decode import AsyncDecoder
            self.decoder = AsyncDecoder(self.runtime)
        self.decoder.submit(state)
        self.decode_ticket = "decode-%s-%s" % (request["state_id"], request["id"])
        self.decode_state = request["state_id"]
        self.decode_result = None
        self.decode_request = dict(request)
        return dict(ticket=self.decode_ticket, ready=False, state_id=self.decode_state)

    def decode_poll(self, request):
        if request.get("ticket") != self.decode_ticket or self.decode_ticket is None:
            raise ValueError("unknown decode ticket")
        decoded = self.decoder.collect(wait=bool(request.get("wait", False)))
        if decoded is None:
            return dict(ticket=self.decode_ticket, ready=False)
        output, event = decoded
        key = self.decode_state
        state = self.states[key]
        state.finished = True
        state.events.append(event)
        ticket, self.decode_ticket, self.decode_state = self.decode_ticket, None, None
        request = dict(self.decode_request, **request)
        request["state_id"] = key
        self.decode_request = None
        return dict(ticket=ticket, ready=True, state_id=key, progress=self._progress(state),
                    output=self.persist(output, request),
                    model_ms=event["ms"], event=event, **self._evidence(request, state.spec, state.latents, self._paths(state)))

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
        key = request.get("state_id", request.get("snapshot_id"))
        if key == self.decode_state:
            raise ValueError("state is owned by an outstanding async decode")
        continuation = self.continuations.get(key)
        if not self._transfer_ready(state, continuation):
            raise ValueError("transfer requires a completed solver-step boundary")
        return state

    def snapshot(self, request):
        state = self._boundary(request)
        key = request["state_id"]
        if request.get("target_device"):
            if key in self.continuations or self.parallel:
                raise ValueError("heterogeneous migration requires the SP1 whole-step adapter")
            migration_id = request["migration_id"]
            action = request.get("action", {"cold": "cold_migrate", "hot": "full_migrate", "keep": "full_migrate"}[request.get("mode", "hot")])
            manifest, tensors, timing = self.endpoint.pause_export(state, action=action,
                target_device=request["target_device"], migration_id=migration_id,
                epoch=request.get("epoch", 0), certificate=request.get("certificate"))
            destination = Path(request["path"]).resolve()
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    self.runtime.torch.save(dict(manifest=manifest, tensors=tensors), stream)
            except BaseException:
                self.endpoint.abort(migration_id)
                raise
            return dict(snapshot_id=migration_id, path=str(destination), sha256=file_hash(destination),
                        bytes=destination.stat().st_size, phase="paused", manifest=manifest, **timing)
        mode = request.get("mode", "hot")
        from .state import state_inventory, assert_transfer_supported
        with self.runtime._lock:
            inventory = state_inventory(state, torch_module=self.runtime.torch)
            assert_transfer_supported(inventory)
            if mode in ("hot", "cold") and any(row.get("supported_timestep_views") for row in inventory["storages"]):
                from .storage import snapshot_timestep_views
                result = snapshot_timestep_views(self.runtime, state, mode, inventory)
            else:
                result = self.runtime.snapshot(state, mode)
        return dict(result, snapshot_id=key, state_id=key)

    def restore(self, request):
        if request.get("path"):
            key = self.new_id(request)
            if key in self.pending.values():
                raise ValueError("state_id already reserved by a prepared migration")
            path = Path(request["path"])
            if not request.get("sha256") or file_hash(path) != request["sha256"]:
                raise ValueError("migration snapshot hash mismatch")
            payload = self.runtime.torch.load(path, map_location="cpu", weights_only=True)
            ack = self.endpoint.prepare(payload["manifest"], payload["tensors"])
            self.pending[ack["migration_id"]] = key
            return dict(state_id=key, snapshot_id=ack["migration_id"], phase="prepared", acknowledgement=ack,
                        runnable=False)
        if request.get("receipt"):
            migration_id = request["migration_id"]
            if migration_id not in self.pending:
                raise ValueError("unknown prepared migration")
            state = self.endpoint.commit(migration_id, request["receipt"])
            key = self.pending.pop(migration_id)
            self.states[key] = state
            return dict(state_id=key, progress=self._progress(state), phase="committed", runnable=True)
        state = self.state(request)
        if state.pending_snapshot and state.pending_snapshot.get("research_storage_codec"):
            from .storage import restore_timestep_views
            with self.runtime._lock:
                result = restore_timestep_views(self.runtime, state)
        else:
            result = self.runtime.restore(state)
        return dict(result, state_id=request.get("state_id", request.get("snapshot_id")),
                    progress=self._progress(state))

    def block_export(self, request):
        """Write a block-boundary continuation to a file; release the source by default."""
        key = request["state_id"]
        state = self.state(request)
        if key not in self.continuations:
            raise ValueError("block export requires a block continuation")
        if key == self.decode_state:
            raise ValueError("state is owned by an outstanding async decode")
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
        return dict(timing, path=str(destination), sha256=file_hash(destination), bytes=destination.stat().st_size,
                    layout=layout, released=bool(request.get("release", True)))

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

    def migration_release(self, request):
        migration_id = request["migration_id"]
        self.endpoint.acknowledge(migration_id, request["acknowledgement"])
        receipt = self.endpoint.release_source(migration_id)
        for key, state in list(self.states.items()):
            if state.request_id == receipt["request_id"]:
                del self.states[key]
        return dict(receipt=receipt, phase="released")

    def migration_abort(self, request):
        self.endpoint.abort(request["migration_id"])
        self.pending.pop(request["migration_id"], None)
        return dict(aborted=request["migration_id"])

    def inventory(self, request):
        from .state import state_inventory
        return state_inventory(self.state(request), torch_module=self.runtime.torch)

    def set_cache(self, request):
        state = self._boundary(request)
        if request["state_id"] in self.continuations:
            raise ValueError("block continuations are cache-off and cannot change cache policy")
        from .runtime import normalize_spec
        spec = dict(state.spec)
        for key in ("cache", "threshold", "max_continuous"):
            if key in request:
                spec[key] = request[key]
        state.spec = normalize_spec(spec)
        self.runtime._update_policy(state)
        return dict(cache=state.cache_enabled, threshold=state.spec["threshold"],
                    max_continuous=state.spec["max_continuous"], qualified=False)

    def local_roundtrip(self, request):
        state = self._boundary(request)
        if request["state_id"] in self.continuations or self.parallel:
            raise ValueError("roundtrip requires SP1 whole-step state")
        from .migration import local_roundtrip
        state, result = local_roundtrip(self.runtime, state, action=request.get("action", "full_migrate"),
            migration_id=request["migration_id"], certificate=request.get("certificate"))
        self.states[request["state_id"]] = state
        return dict(result, progress=self._progress(state), qualified=False)

    def distributed(self, request):
        action = request.get("action", "init")
        if action == "init":
            if self.parallel is not None or self.states or self.execution != "step":
                raise ValueError("SP initialization requires a fresh whole-step worker")
            if self.runtime.attention != "flash":
                raise ValueError("pinned Wan SP requires config.attention=flash")
            torch = self.runtime.torch
            world_size = request.get("world_size", 4)
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
        if action == "cache_off_migrate":
            key = request["state_id"]
            state, event = self.parallel.migrate_cache_off(self.states.get(key), request["source_ranks"],
                request["target_ranks"], request_id=request["request_id"], step_index=request["step_index"],
                expected_epoch=request["expected_epoch"])
            if state is None:
                self.states.pop(key, None)
            else:
                self.states[key] = state
            return dict(event, state_id=key if state is not None else None, gpu_validated=False)
        raise ValueError("unknown distributed action")

    def reference(self, request):
        if self.decode_ticket is not None:
            raise ValueError("official reference uses VAE; collect async decode first")
        if self.parallel:
            raise ValueError("SP references require an independent unsharded worker")
        spec = self._spec(request)
        result = self.runtime.reference(spec)
        output = result.pop("output")
        latents = result.pop("latents")
        paths = result.pop("path_history")
        return dict(result, output=self.persist(output, request),
                    **self._evidence(request, spec, latents, paths))

    def finish(self, request):
        if self.decode_ticket is not None:
            raise ValueError("VAE already owns an asynchronous decode; use decode_poll")
        state = self._boundary(request)
        if request.get("decode_frames") is not None:
            # Preemptible decode: repeated finish calls advance latent-frame chunks.
            output, info = self.runtime.decode_chunks(state, positive_int(request["decode_frames"], "decode_frames"))
            if output is None:
                return dict(state_id=request["state_id"], progress=self._progress(state), decode=info, model_ms=info["ms"])
            return dict(state_id=request["state_id"], output=self.persist(output, request), decode=info,
                        progress=self._progress(state), model_ms=info["ms"],
                        **self._evidence(request, state.spec, state.latents, self._paths(state)))
        output = self.runtime.finish(state)
        return dict(state_id=request["state_id"], output=self.persist(output, request),
                    progress=self._progress(state), model_ms=state.events[-1]["ms"],
                    **self._evidence(request, state.spec, state.latents, self._paths(state)))

    @staticmethod
    def _paths(state):
        keys = ("step_index", "cache_contexts", "reused_forwards", "path_scope", "path")
        return [{key: event[key] for key in keys if key in event}
                for event in state.events if event.get("kind") == "step"]

    def _evidence(self, request, spec, latents, paths):
        from .runtime import REVISION
        generation = dict(spec, model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", revision=REVISION,
            model_path=str(self.runtime.model_path), model_integrity="local_snapshot_not_rehashed",
            dtype="float16", vae_dtype="float32", text_encoder_dtype="float32", text_encoder_device="cpu",
            fps=self.config.get("fps", 16), attention=self.runtime.attention,
            cache_enabled=spec["cache"], cache_threshold=spec["threshold"],
            approximate=bool(spec["cache"] or any(row.get("reused_forwards", 0) for row in paths)))
        result = dict(backend="wan", measured=True, generation=generation, path_history=paths)
        if request.get("save_latents"):
            destination = request.get("latent_path") or str(Path(self.output_path(request)).with_suffix(".latent.npy"))
            result["latent"] = artifact(latents, destination)
        return result

    def release(self, request):
        state = self.state(request)
        key = request["state_id"]
        if key == self.decode_state:
            raise ValueError("collect asynchronous decode before releasing its state")
        if key in self.continuations:
            self.executor.close(self.continuations.pop(key))
        self.runtime.release(state)
        del self.states[key]
        return dict(released=key)

    def close(self):
        if self.decoder:
            self.decoder.close()
            self.decode_ticket = self.decode_state = None
        self.encoder.shutdown(wait=True, cancel_futures=True)
        self.tickets.clear()
        for migration_id in list(self.pending):
            self.endpoint.abort(migration_id)
        self.pending.clear()
        for migration_id, row in list(self.endpoint._sources.items()):
            if row["phase"] in ("paused", "acknowledged"):
                self.endpoint.abort(migration_id)
        super().close()
        if self.parallel:
            self.parallel.close()
        if self.owns_world:
            self.runtime.torch.distributed.destroy_process_group()
        self.runtime._sync()
